
import torch

from dataset import Dataset
from graph import Mesh, NodeType, interpolate_field, cells_to_edges, find_edge, find_edges_with
from utils import BATCH, batch_dicts, get_triangle_aspect_ratio, get_triangle_sarea
from remesher.core import get_sizing_field_tensor, closest_SDP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from collections import namedtuple
from typing import Any


EPS = 1e-3
AR_THRESHOLD = 5 # aspect ratio threshold

MeshState = namedtuple(
    "MeshState",
    ["edges", "opposites", "modified_edge_mask", "node_mask", "mesh_pos", "world_pos", "S_i"]
)


def plot_mesh(ax, mesh: Mesh, color="w", alpha=1.0):
    triangles = mesh["cells"][BATCH]
    type_mask = (mesh["node_type"][BATCH][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()

    triang = mtri.Triangulation(mesh["mesh_pos"][BATCH][:,0], mesh["mesh_pos"][BATCH][:,1], triangles, ~type_mask)
    ax.triplot(triang, linewidth=1, color=color, alpha=alpha)
    ax.set_aspect('equal')
    ax.set_ylim(1, 0)
    ax.set_axis_off()

def plot_mesh_state(ax, ms: MeshState, color="w", alpha=1.0):
    edge_mask = ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
    for e_id in torch.nonzero(edge_mask):
        e_id = e_id.item()
        ax.plot(ms.mesh_pos[ms.edges[e_id]][:, 0], ms.mesh_pos[ms.edges[e_id]][:, 1], color, alpha=alpha)
    ax.set_aspect('equal')
    ax.set_ylim(1, 0)
    ax.set_axis_off()

def plot_ellipse(ax, S: torch.Tensor, pos=(0,0), c="r") -> None:
    eigenvalues, eigenvectors = np.linalg.eig(S.numpy())
    axis_size = 2 / np.sqrt(eigenvalues)
    angle = np.rad2deg(np.atan2(eigenvectors[1,0], eigenvectors[0,0]))
    plt_ellipse = PltEllipse(
        pos, axis_size[0], axis_size[1], angle=angle,
        facecolor="none", edgecolor=c, linewidth=2
    )
    ax.scatter(*pos, c=c)
    ax.add_patch(plt_ellipse)

# TODO: not that much used
def is_border(ms: MeshState, edge_id) -> bool:
    return (ms.opposites[edge_id][-1] == -1).item()

def get_splittable_edges_mask(ms: MeshState) -> torch.Tensor:
    """ Find splittable edges """
    S_ij = 0.5 * (ms.S_i[ms.edges[:, 0]] + ms.S_i[ms.edges[:, 1]])
    u_ij = ms.mesh_pos[ms.edges[:, 1]] - ms.mesh_pos[ms.edges[:, 0]]
    edge_size = torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij)
    return (edge_size > 1 + EPS)

def get_flippable_edges_mask(ms: MeshState) -> torch.Tensor:
    """ Find flippable edges """
    i, j = ms.edges[:, 0], ms.edges[:, 1]
    k, l = ms.opposites[:, 0], ms.opposites[:, 1]

    border_mask = (l == -1)

    u_ik = ms.mesh_pos[i] - ms.mesh_pos[k]
    u_jk = ms.mesh_pos[j] - ms.mesh_pos[k]
    u_il = ms.mesh_pos[i] - ms.mesh_pos[l]
    u_jl = ms.mesh_pos[j] - ms.mesh_pos[l]
    S_A = 0.5 * (ms.S_i[i] + ms.S_i[j] + ms.S_i[k] + ms.S_i[l])

    ujk_x_uik = u_jk[:, 0] * u_ik[:, 1] - u_jk[:, 1] * u_ik[:, 0]
    uil_x_ujl = u_il[:, 0] * u_jl[:, 1] - u_il[:, 1] * u_jl[:, 0]
    uil_SA_ujl = torch.einsum("ei,eij,ej->e", u_il, S_A, u_jl)
    ujk_SA_uik = torch.einsum("ei,eij,ej->e", u_jk, S_A, u_ik)
    flippable = ujk_x_uik*uil_SA_ujl + ujk_SA_uik*uil_x_ujl < 0
    flippable &= ~border_mask

    return flippable

def get_maximal_independent_set(edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Find a maximal independent set """
    maximal_independent_set = []
    node_set = set()
    for e_id in torch.nonzero(mask):
        if edges[e_id,0].item() not in node_set and edges[e_id, 1].item() not in node_set:
            maximal_independent_set.append(e_id)
            node_set.add(edges[e_id, 0].item())
            node_set.add(edges[e_id, 1].item())

    return torch.Tensor(maximal_independent_set).type(torch.long)

# TODO: add a stopping contidition (max_iter ?)
def flip(ms: MeshState, new_edge_mask: torch.Tensor) -> MeshState:
    edge_mask = new_edge_mask
    edge_mask &= ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
    while True:
        flippable_mask = get_flippable_edges_mask(ms) & edge_mask
        maximal_independent_set = get_maximal_independent_set(ms.edges, flippable_mask)
        if len(maximal_independent_set) == 0: break

        def correct_opposites(nodes: tuple[Any,Any], old, new) -> None:
            e_id = find_edge(nodes[0], nodes[1], updated_edges)
            updated_opposites[e_id][updated_opposites[e_id] == old] = new

        updated_edges = ms.edges.clone()
        updated_opposites = ms.opposites.clone()
        for edge_id in maximal_independent_set:
            i, j = updated_edges[edge_id]
            k, l = updated_opposites[edge_id]

            if find_edge(k, l, updated_edges) != -1: continue
            if get_triangle_aspect_ratio(ms.mesh_pos, i, k, l) > AR_THRESHOLD: continue
            if get_triangle_aspect_ratio(ms.mesh_pos, j, k, l) > AR_THRESHOLD: continue

            # check if a triangle would be inverted
            tris = torch.stack([
                torch.stack([ms.mesh_pos[i], ms.mesh_pos[k], ms.mesh_pos[j]], dim=0),
                torch.stack([ms.mesh_pos[i], ms.mesh_pos[k], ms.mesh_pos[l]], dim=0),
                torch.stack([ms.mesh_pos[j], ms.mesh_pos[i], ms.mesh_pos[k]], dim=0),
                torch.stack([ms.mesh_pos[j], ms.mesh_pos[l], ms.mesh_pos[k]], dim=0)
            ], dim=0)
            tris_orientation = torch.sign(get_triangle_sarea(tris))
            if tris_orientation[0] != tris_orientation[1] or tris_orientation[2] != tris_orientation[3]: continue

            correct_opposites((i, k), j, l)
            correct_opposites((i, l), j, k)
            correct_opposites((j, k), i, l)
            correct_opposites((j, l), i, k)

            updated_edges[edge_id]     = torch.Tensor([k, l])
            updated_opposites[edge_id] = torch.Tensor([i, j])

        edge_mask[maximal_independent_set] = False  # edges checked

        ms = ms._replace(
            edges     = updated_edges,
            opposites = updated_opposites,
        )

    return ms

# TODO: add a stopping contidition (max_iter ?)
def split(ms: MeshState) -> MeshState:
    from copy import deepcopy
    temp = deepcopy(ms)
    """ Split all possible edges in `ms` """

    def add_edge(nodes: tuple[Any,Any], opposites: tuple[Any,Any]) -> None:
        new_edges_buf.append(nodes)
        new_opposites_buf.append(opposites)

    def correct_opposites(nodes: tuple[Any,Any], old, new) -> None:
        e_id = find_edge(nodes[0], nodes[1], updated_edges)
        updated_opposites[e_id][updated_opposites[e_id] == old] = new

    def split_edge(i, j, k, l, m) -> bool:
        # check if new triangles would have a bad aspect ratio
        if get_triangle_aspect_ratio(updated_mesh_pos, i, m, k) > AR_THRESHOLD: return False
        if get_triangle_aspect_ratio(updated_mesh_pos, j, m, k) > AR_THRESHOLD: return False
        if l != -1:
            if get_triangle_aspect_ratio(updated_mesh_pos, i, m, l) > AR_THRESHOLD: return False
            if get_triangle_aspect_ratio(updated_mesh_pos, j, m, l) > AR_THRESHOLD: return False

        # add new edges and their corresponding opposite nodes
        add_edge((i, m), (k, l))
        add_edge((j, m), (k, l))
        add_edge((k, m), (i, j))
        if l != -1:
            add_edge((l, m), (i, j))

        # correct opposite nodes of adjacent edges
        correct_opposites((i, k), j, m)
        correct_opposites((j, k), i, m)
        if l != -1:
            correct_opposites((i, l), j, m)
            correct_opposites((j, l), i, m)

        return True

    edge_mask = ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
    modified_edge_mask = torch.zeros(len(ms.edges)).bool()
    while True:
        invalid_edge_mask = get_splittable_edges_mask(ms) & edge_mask
        maximal_independent_set = get_maximal_independent_set(ms.edges, invalid_edge_mask)
        if len(maximal_independent_set) == 0: break

        # interpolate values
        i_, j_ = ms.edges[maximal_independent_set][:,0], ms.edges[maximal_independent_set][:,1]
        m_mesh_pos  = 0.5 * (ms.mesh_pos[i_] + ms.mesh_pos[j_])
        m_world_pos = 0.5 * (ms.world_pos[i_] + ms.world_pos[j_])
        m_S_i       = 0.5 * (ms.S_i[i_] + ms.S_i[j_])
        m_S_i       = torch.stack([closest_SDP(S) for S in torch.unbind(m_S_i, dim=0)], dim=0)
        updated_node_mask = torch.concat([ms.node_mask.clone(), torch.Tensor([True]*len(maximal_independent_set))])
        updated_mesh_pos  = torch.concat([ms.mesh_pos.clone(),  m_mesh_pos])
        updated_world_pos = torch.concat([ms.world_pos.clone(), m_world_pos])
        updated_S_i       = torch.concat([ms.S_i.clone(),       m_S_i])

        # split edges
        updated_edges = ms.edges.clone()
        updated_opposites = ms.opposites.clone()
        new_edges_buf = []
        new_opposites_buf = []
        removed_edges, kept_edges = [], []
        for it, edge_id in enumerate(maximal_independent_set):
            i, j = i_[it], j_[it]
            k, l = updated_opposites[edge_id].unbind()
            m = len(ms.node_mask) + it
            if split_edge(i, j, k, l, m):
                removed_edges.append(edge_id)
            else:
                kept_edges.append(edge_id)

        removed_edges = torch.Tensor(removed_edges).long()
        kept_edges    = torch.Tensor(kept_edges).long()

        new_edges     = torch.Tensor(new_edges_buf)
        new_opposites = torch.Tensor(new_opposites_buf)
        
        removed_edges_mask = torch.zeros(len(updated_edges)).bool()
        removed_edges_mask[removed_edges] = True

        updated_edges = updated_edges[~removed_edges_mask]
        updated_edges = torch.concat([updated_edges, new_edges]).long()
        updated_opposites = updated_opposites[~removed_edges_mask]
        updated_opposites = torch.concat([updated_opposites, new_opposites]).long()
        modified_edge_mask = modified_edge_mask[~removed_edges_mask]
        modified_edge_mask = torch.concat([modified_edge_mask, torch.ones(len(new_edges)).bool()])

        edge_mask[kept_edges] = False
        edge_mask = edge_mask[~removed_edges_mask]
        edge_mask = torch.concat([edge_mask, torch.ones(len(new_edges))]).bool()

        ms = MeshState(
            edges = updated_edges.long(),
            opposites = updated_opposites.long(),
            modified_edge_mask = modified_edge_mask,
            node_mask = updated_node_mask.bool(),
            mesh_pos = updated_mesh_pos,
            world_pos = updated_world_pos,
            S_i = updated_S_i
        )

        new_edge_mask = torch.zeros(len(updated_edges)).bool()
        if len(new_edges) > 0:
            new_edge_mask[-len(new_edges):] = True
        ms = flip(ms, new_edge_mask.clone())
        
    return ms

def collapse(ms: MeshState) -> MeshState:
    """ Collapse all possible edges in `ms` """
    h: float = 0.0  # "close to invalid" constant

    # TODO: check the size of the new 
    def check_face(prev_tri: list, new_tri: list, prev_edges: list, new_edges: list[list]) -> bool:
        boundary_condition = not is_border(ms, prev_edges[0]) and not is_border(ms, prev_edges[1])
        if not boundary_condition: return False

        aspect_ratio = get_triangle_aspect_ratio(ms.mesh_pos, *new_tri).item()
        aspect_condition = aspect_ratio < AR_THRESHOLD
        if not aspect_condition: return False

        tris = torch.stack([ms.mesh_pos[prev_tri], ms.mesh_pos[new_tri]], dim=0)
        tris_orientation = torch.sign(get_triangle_sarea(tris))
        not_inverted_condition = tris_orientation[0].item() == tris_orientation[1].item()
        if not not_inverted_condition: return False

        # FIXME: this removes almost every edges
        edges = tris = torch.stack([
            torch.Tensor(new_edges[0]),
            torch.Tensor(new_edges[1])
        ], dim=0).long()
        S_ij = 0.5 * (ms.S_i[edges[:, 0]] + ms.S_i[edges[:, 1]])
        u_ij = ms.mesh_pos[edges[:, 1]] - ms.mesh_pos[edges[:, 0]]
        edge_size = torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij)
        edge_size_condition = bool(torch.any(edge_size > 1 - h).item())
        if edge_size_condition: return False

        return True
    
    def check_collapisble(i, j, k, l) -> bool:
        ij = find_edge(i, j, ms.edges)
        
        # find all edges from `j` (without `i,j`)
        edges_j = find_edges_with(j, ms.edges)
        edges_j = edges_j[edges_j != ij]

        # find all nodes that are at the end of the edges found previously
        adjacent_nodes = ms.edges[edges_j].flatten()
        adjacent_nodes = adjacent_nodes[adjacent_nodes != i]
        adjacent_nodes = adjacent_nodes[adjacent_nodes != j]
        
        # check all edges for "forbidden" cases
        checked_tris = set()
        for e_id in edges_j:
            nodes = ms.edges[e_id]
    
            adjacent_id = nodes[nodes != j]
            for node_id in adjacent_nodes[adjacent_nodes != adjacent_id]:
                adjacent_edge = find_edge(adjacent_id, node_id, ms.edges)
                if adjacent_edge == -1: continue

                tri, order = torch.sort(torch.Tensor([j, node_id, adjacent_id]))
                tri = tri.long().tolist()

                if tuple(tri) not in checked_tris:
                    checked_tris.add(tuple(tri))

                    new_tri = torch.Tensor([i, node_id, adjacent_id])[order].long().tolist()
                    if not check_face(tri, new_tri, [e_id, adjacent_edge], [[i, node_id], [i, adjacent_id]]):
                        return False

        return True

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_mesh_state(ax1, ms, "w")

    # edges_j = find_edges_with(1198, ms.edges)
    # print(edges_j)
    # print(ms.edges[edges_j])
    # exit(2)

    i = -1
    while torch.count_nonzero(ms.modified_edge_mask) > 0:
        edge_id = torch.nonzero(ms.modified_edge_mask)[0,0]
        ms.modified_edge_mask[edge_id] = False

        deleted_edges_buf = []
        #! deleted_nodes = []   #? need to shift every node idx in the edge map
        added_edges_buf = []
        #! added_opposites_buf = []    # TODO
        # TODO: update opposites on the "ring" around j

        nodes = [*ms.edges[edge_id], *ms.opposites[edge_id]]
        collapsible = check_collapisble(*nodes)
        if not collapsible:
            nodes[0], nodes[1] = nodes[1], nodes[0] # swap `i` and`j`
            collapsible = check_collapisble(*nodes)

        if collapsible:
            i += 1
            # if i != 2: continue
            if i > 0: break

            ji = find_edge(nodes[1], nodes[0], ms.edges)
            jk = find_edge(nodes[1], nodes[2], ms.edges)
            jl = find_edge(nodes[1], nodes[3], ms.edges)

            ax1.plot(ms.mesh_pos[ms.edges[ji], 0], ms.mesh_pos[ms.edges[ji], 1], "g")
            ax1.scatter(ms.mesh_pos[nodes[0], 0], ms.mesh_pos[nodes[0], 1], color="c", alpha=1, zorder=100)


            from_j = find_edges_with(nodes[1], ms.edges)
            from_j = from_j[(from_j != jk) & (from_j != jl) & (from_j != ji)]

            deleted_edges_buf.append(ji)
            deleted_edges_buf.append(jk)
            ax1.plot(ms.mesh_pos[ms.edges[jk], 0], ms.mesh_pos[ms.edges[jk], 1], "r")
            deleted_edges_buf.append(jl)
            ax1.plot(ms.mesh_pos[ms.edges[jl], 0], ms.mesh_pos[ms.edges[jl], 1], "r")
            for e in from_j:
                deleted_edges_buf.append(e)
                ax1.plot(ms.mesh_pos[ms.edges[e], 0], ms.mesh_pos[ms.edges[e], 1], "b")
    
                adjacent_node = ms.edges[e][ms.edges[e] != nodes[1]]
                added_edges_buf.append([nodes[0], adjacent_node])

            deleted_edge_mask = torch.ones(len(ms.edges)).bool()
            deleted_edge_mask[torch.Tensor(deleted_edges_buf).long()] = False

            updated_modified_edge_mask = torch.concat([ms.modified_edge_mask[deleted_edge_mask], torch.zeros(len(added_edges_buf)).bool()])
            updated_edges = torch.concat([ms.edges[deleted_edge_mask], torch.Tensor(added_edges_buf).long()])
            #! updated_opposites = torch.concat([ms.opposites[deleted_edge_mask], torch.Tensor(added_opposites_buf).long()])

            ms = ms._replace(
                edges = updated_edges,
                #! opposites = ...,
                modified_edge_mask = updated_modified_edge_mask
            )

            # break   #!
    
    plot_mesh_state(ax2, ms, color="gray")
    plt.show()
    
    return ms


ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 40

"""
COMPUTATION
"""
mesh1 = batch_dicts([ds[traj_len*traj + time_idx + 0]])
mesh2 = batch_dicts([ds[traj_len*traj + time_idx + 1]])

normal1_mask = mesh1["node_type"][BATCH].flatten() == NodeType.NORMAL
normal2_mask = mesh2["node_type"][BATCH].flatten() == NodeType.NORMAL

sizing_field2 = get_sizing_field_tensor(mesh2)
S_i = interpolate_field(mesh1, mesh2, sizing_field2)

edges, opposites = cells_to_edges(mesh1)
edges = edges[:edges.shape[0]//2]  # only use one way edges (same computation for the others)
opposites = opposites[:opposites.shape[0]//2]  # only use one way edges (same computation for the others)

ms = MeshState(
    edges = edges,
    opposites = opposites,
    modified_edge_mask = [],
    node_mask = normal1_mask,
    mesh_pos = mesh1["mesh_pos"][BATCH],
    world_pos = mesh1["world_pos"][BATCH],
    S_i = S_i
)

split_ms = split(ms)

collapse_ms = collapse(split_ms)
exit(1)

"""
DISPLAY
"""
plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2, ax3, ax4 = fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144)

ax1.set_title("Starting Mesh")
plot_mesh(ax1, mesh1)

ax2.set_title("Splitted Edges")
plot_mesh(ax2, mesh1, alpha=0.5)
plot_mesh_state(ax2, split_ms, color="r", alpha=0.5)
ax2.scatter(split_ms.mesh_pos[1198,0], split_ms.mesh_pos[1198,1], color="y")

ax3.set_title("Collapsed Edges")
plot_mesh(ax3, mesh1, "gray")

ax4.set_title("Target Mesh")
# plot_mesh(ax4, mesh1, alpha=0.5)
# plot_mesh_state(ax4, split_ms, "r", alpha=0.5)
plot_mesh(ax4, mesh2)


fig.tight_layout()
plt.show()
