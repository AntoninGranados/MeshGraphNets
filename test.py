
import torch

from dataset import Dataset
from graph import Mesh, NodeType, interpolate_field, cells_to_edges, find_edge
from utils import BATCH, batch_dicts, get_triangle_sarea
from remesher.core import get_sizing_field_tensor, closest_SDP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from collections import namedtuple
from typing import Any

def plot_mesh(ax, mesh: Mesh, color="w"):
    triangles = mesh["cells"][BATCH]
    type_mask = (mesh["node_type"][BATCH][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()

    triang = mtri.Triangulation(mesh["mesh_pos"][BATCH][:,0], mesh["mesh_pos"][BATCH][:,1], triangles, ~type_mask)
    ax.triplot(triang, linewidth=1, color=color)
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


EPS = 1e-3
AR_THRESHOLD = 1e3 # aspect ratio threshold

MeshState = namedtuple(
    "MeshState",
    ["edges", "opposites", "node_mask", "mesh_pos", "world_pos", "S_i"]
)

def get_splittable_edges_mask(ms: MeshState) -> torch.Tensor:
    """ Find splittable edges """
    S_ij = 0.5 * (ms.S_i[ms.edges[:, 0]] + ms.S_i[ms.edges[:, 1]])
    u_ij = ms.mesh_pos[ms.edges[:, 1]] - ms.mesh_pos[ms.edges[:, 0]]
    edge_size = torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij)
    return (edge_size > 1 + EPS)

def get_maximal_independent_set(ms: MeshState, mask: torch.Tensor) -> torch.Tensor:
    """ Find a maximal independent set """

    maximal_independent_set = []
    node_set = set()
    for e_id in torch.nonzero(mask):
        if ms.edges[e_id,0].item() not in node_set and ms.edges[e_id, 1].item() not in node_set:
            maximal_independent_set.append(e_id)
            node_set.add(ms.edges[e_id, 0].item())
            node_set.add(ms.edges[e_id, 1].item())

    return torch.Tensor(maximal_independent_set).type(torch.long)

# TODO: add a stopping contidition (max_iter ?)
# TODO: flip edge at the end of each iteration
def split(ms: MeshState) -> MeshState:
    """ Split all possible edges in `ms` """

    while True:
        edge_mask = ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
        invalid_edges = get_splittable_edges_mask(ms) & edge_mask
        maximal_independent_set = get_maximal_independent_set(ms, invalid_edges)
        if len(maximal_independent_set) == 0: return ms

        # interpolate values
        i_, j_ = ms.edges[maximal_independent_set][:,0], ms.edges[maximal_independent_set][:,1]
        m_mesh_pos  = 0.5 * (ms.mesh_pos[i_] + ms.mesh_pos[j_])
        m_world_pos = 0.5 * (ms.world_pos[i_] + ms.world_pos[j_])
        m_S_i       = 0.5 * (ms.S_i[i_] + ms.S_i[j_])
        m_S_i       = torch.stack([closest_SDP(S) for S in torch.unbind(m_S_i, dim=0)], dim=0)
        new_node_mask = torch.concat([ms.node_mask.clone(), torch.Tensor([True]*len(maximal_independent_set))])
        new_mesh_pos  = torch.concat([ms.mesh_pos.clone(),  m_mesh_pos])
        new_world_pos = torch.concat([ms.world_pos.clone(), m_world_pos])
        new_S_i       = torch.concat([ms.S_i.clone(),       m_S_i])

        # split edges
        def add_edge(nodes: tuple[Any,Any], opposites: tuple[Any,Any]) -> None:
            new_edges_buf.append(nodes)
            new_opposites_buf.append(opposites)

        def correct_opposites(nodes: tuple[Any,Any], old, new) -> None:
            e_id = find_edge(nodes[0], nodes[1], new_edges)
            new_opposites[e_id][new_opposites[e_id] == old] = new

        def get_aspect_ratio(i, j, k) -> torch.Tensor:
            a = torch.norm(new_mesh_pos[i].unsqueeze(0) - new_mesh_pos[j].unsqueeze(0))
            b = torch.norm(new_mesh_pos[j].unsqueeze(0) - new_mesh_pos[k].unsqueeze(0))
            c = torch.norm(new_mesh_pos[k].unsqueeze(0) - new_mesh_pos[i].unsqueeze(0))

            return a*b*c / ((b+c-a)*(c+a-b)*(a+b-c))
        
        def split_edge(i, j, k, l, m) -> None:
            # check if new triangles would have a bad aspect ratio
            if get_aspect_ratio(i, m, k) > AR_THRESHOLD or get_aspect_ratio(j, m, k) > AR_THRESHOLD: return
            if l != -1:
                if get_aspect_ratio(i, m, l) > AR_THRESHOLD or get_aspect_ratio(j, m, l) > AR_THRESHOLD: return

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


        new_edges = ms.edges.clone()
        new_opposites = ms.opposites.clone()
        new_edges_buf = []
        new_opposites_buf = []
        for it, e_id in enumerate(maximal_independent_set):
            i, j = i_[it], j_[it]
            k, l = new_opposites[e_id].unbind()
            m = len(ms.node_mask) + it
            split_edge(i, j, k, l, m)

        new_edges     = torch.concat([new_edges, torch.Tensor(new_edges_buf)]).type(torch.long)
        new_opposites = torch.concat([new_opposites, torch.Tensor(new_opposites_buf)]).type(torch.long)

        invalid_edges = torch.zeros(len(new_edges)).type(torch.bool)
        invalid_edges[maximal_independent_set] = True
        new_edges = new_edges[~invalid_edges]
        new_opposites = new_opposites[~invalid_edges]

        ms = MeshState(
            edges = new_edges.type(torch.long),
            opposites = new_opposites.type(torch.long),
            node_mask = new_node_mask.type(torch.bool),
            mesh_pos = new_mesh_pos,
            world_pos = new_world_pos,
            S_i = new_S_i
        )

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
    node_mask = normal1_mask,
    mesh_pos = mesh1["mesh_pos"][BATCH],
    world_pos = mesh1["world_pos"][BATCH],
    S_i = S_i
)

new_ms = split(ms)
new_edge_count = len(new_ms.edges) - len(ms.edges)  #! wrong computation (does not account for removed edges)

# exit(1)

"""
DISPLAY
"""
plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2, ax3, ax4 = fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144)

ax1.set_title("Starting Mesh")
plot_mesh(ax1, mesh1)

ax2.set_title("Splitted Edges")
edge_mask = new_ms.node_mask[new_ms.edges[:, 0]] | new_ms.node_mask[new_ms.edges[:, 1]]
for e_id in torch.nonzero(edge_mask):
    e_id = e_id.item()
    ax2.plot(new_ms.mesh_pos[new_ms.edges[e_id]][:, 0], new_ms.mesh_pos[new_ms.edges[e_id]][:, 1], "c", linewidth=0.5, zorder=-1)
plot_mesh(ax2, mesh1)

ax3.set_title("Collapsed Edges")
plot_mesh(ax3, mesh1, "gray")

ax4.set_title("Target Mesh")
plot_mesh(ax4, mesh1, "r")
plot_mesh(ax4, mesh2)


fig.tight_layout()
plt.show()
