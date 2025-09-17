
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

def plot_mesh(ax, mesh: Mesh):
    triangles = mesh["cells"][BATCH]
    type_mask = (mesh["node_type"][BATCH][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()

    triang = mtri.Triangulation(mesh["mesh_pos"][BATCH][:,0], mesh["mesh_pos"][BATCH][:,1], triangles, ~type_mask)
    ax.triplot(triang, linewidth=1, color="w")
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

MeshState = namedtuple(
    "MeshState",
    ["edges", "opposites", "node_mask", "mesh_pos", "world_pos", "S_i"]
)

def splittable_edges(ms: MeshState) -> torch.Tensor:
    normal_edge_mask = ms.node_mask[ms.edges[:,0]] | ms.node_mask[ms.edges[:,1]]

    S_ij = 0.5 * (ms.S_i[ms.edges[:,0]] + ms.S_i[ms.edges[:,1]])
    u_ij = ms.mesh_pos[ms.edges[:,0]] - ms.mesh_pos[ms.edges[:,1]]

    metric = torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij)
    return (metric > 1 + EPS) & normal_edge_mask

def maximal_independent_set(edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eligible_indices = torch.nonzero(mask).squeeze()
    independent_set = []
    used_vertices = set()
    
    for idx in eligible_indices:
        i, j = edges[idx].tolist()
        if i not in used_vertices and j not in used_vertices:
            independent_set.append(idx.item())
            used_vertices.add(i)
            used_vertices.add(j)
    
    return torch.Tensor(independent_set).type(torch.long)

def split_edge_set(ms: MeshState, independent_set: torch.Tensor) -> MeshState:
    edges_to_split = ms.edges[independent_set]

    # interpolate values
    new_mesh_pos = torch.concat([
        ms.mesh_pos,
        0.5 * (ms.mesh_pos[edges_to_split[:,0]] + ms.mesh_pos[edges_to_split[:,1]])
    ], dim=0)
    
    new_world_pos = torch.concat([
        ms.world_pos,
        0.5 * (ms.world_pos[edges_to_split[:,0]] + ms.world_pos[edges_to_split[:,1]])
    ], dim=0)

    interpolated_S_i = 0.5 * (ms.S_i[edges_to_split[:,0]] + ms.S_i[edges_to_split[:,1]])
    interpolated_S_i = torch.stack([closest_SDP(e) for e in interpolated_S_i.unbind(dim=0)], dim=0)
    new_S_i = torch.concat([ms.S_i, interpolated_S_i], dim=0)

    new_edges = ms.edges.clone()
    new_opposites = ms.opposites.clone()
    new_node_mask = ms.node_mask.clone()
    for e_id in independent_set:
        i, j = ms.edges[e_id].unbind()
        k, l = ms.opposites[e_id].unbind()
        m = len(new_node_mask)

        new_node_mask = torch.concat([new_node_mask, torch.Tensor([True])]).type(torch.bool)
        
        # remove the splitted edge
        new_edges = torch.concat([new_edges[:e_id], new_edges[e_id+1:]], dim=0)
        new_opposites = torch.concat([new_opposites[:e_id], new_opposites[e_id+1:]], dim=0)

        # compute new edges
        new_edges = torch.concat([new_edges, torch.Tensor([[m, i]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[k, l]])], dim=0)

        new_edges = torch.concat([new_edges, torch.Tensor([[m, j]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[k, l]])], dim=0)

        new_edges = torch.concat([new_edges, torch.Tensor([[m, k]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[i, j]])], dim=0)

        if l != -1:   # l = -1 if the edge is on the border and have only one "opposite" point
            new_edges = torch.concat([new_edges, torch.Tensor([[m, l]])], dim=0)
            new_opposites = torch.concat([new_opposites, torch.Tensor([[i, j]])], dim=0)

        # compute new opposites
        ik = find_edge(i, k, new_edges)
        new_opposites[ik][new_opposites[ik] == j] = m

        jk = find_edge(j, k, new_edges)
        new_opposites[jk][new_opposites[jk] == i] = m

        if l != -1:
            il = find_edge(i, l, new_edges)
            new_opposites[il][new_opposites[il] == j] = m

            jl = find_edge(j, l, new_edges)
            new_opposites[jl][new_opposites[jl] == i] = m

    return MeshState(
        edges = new_edges.type(torch.long),
        opposites = new_opposites.type(torch.long),
        node_mask = new_node_mask.type(torch.bool),
        mesh_pos = new_mesh_pos,
        world_pos = new_world_pos,
        S_i = new_S_i
    )

def split_edges(ms: MeshState) -> MeshState:
    while True:
        splittable_mask = splittable_edges(ms)
        independent_set = maximal_independent_set(ms.edges, splittable_mask)
        print(len(independent_set))
        if len(independent_set) == 0: break
        
        print(ms.edges.shape)
        ms = split_edge_set(ms, independent_set)
        print(ms.edges.shape)
        
        # new_faces = get_faces_from_edges(ms, independent_set)
        # ms = flip_edges(ms, active_faces=new_faces)
    
    return ms

"""
def split_edges(ms: MeshState) -> MeshState:
    normal_edge_mask = ms.node_mask[ms.edges[:,0]] | ms.node_mask[ms.edges[:,1]]

    # find the edges to split
    S_ij = 0.5 * (ms.S_i[ms.edges[:,0]] + ms.S_i[ms.edges[:,1]])
    u_ij = ms.mesh_pos[ms.edges[:,0]] - ms.mesh_pos[ms.edges[:,1]]

    metric = torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij)
    metric[~normal_edge_mask] = float("nan")

    invalid_edges_mask = metric > 1 + EPS
    invalid_edges = ms.edges[invalid_edges_mask]
    # sort edges by metric
    argsort_metric = torch.argsort(metric[invalid_edges_mask], descending=True)

    # TODO: interpolate S_i when creating `m`
    new_mesh_pos = torch.concat([
        ms.mesh_pos,
        0.5 * (ms.mesh_pos[invalid_edges[:,0]] + ms.mesh_pos[invalid_edges[:,1]])
    ], dim=0)
    new_node_mask = torch.concat([ms.node_mask, torch.Tensor([True]*len(invalid_edges))]).type(torch.bool)
    new_world_pos = torch.concat([
        ms.world_pos,
        0.5 * (ms.world_pos[invalid_edges[:,0]] + ms.world_pos[invalid_edges[:,1]])
    ], dim=0)

    interpolated_S_i = 0.5 * (ms.S_i[invalid_edges[:,0]] + ms.S_i[invalid_edges[:,1]])
    interpolated_S_i = torch.stack([closest_SDP(e) for e in interpolated_S_i.unbind(dim=0)], dim=0)
    new_S_i = torch.concat([ms.S_i, interpolated_S_i], dim=0)

    new_edges = ms.edges
    new_opposites = ms.opposites
    for e_id in argsort_metric:
        i, j = invalid_edges[e_id].unbind()
        k, l = new_opposites[invalid_edges_mask][e_id].unbind()
        m = len(ms.node_mask) + e_id # offset in `new_nodes_pos`
        
        # compute new edges
        new_edges = torch.concat([new_edges, torch.Tensor([[m, i]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[k, l]])], dim=0)

        new_edges = torch.concat([new_edges, torch.Tensor([[m, j]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[k, l]])], dim=0)

        new_edges = torch.concat([new_edges, torch.Tensor([[m, k]])], dim=0)
        new_opposites = torch.concat([new_opposites, torch.Tensor([[i, j]])], dim=0)

        if l != -1:   # l = -1 if the edge is on the border and have only one "opposite" point
            new_edges = torch.concat([new_edges, torch.Tensor([[m, l]])], dim=0)
            new_opposites = torch.concat([new_opposites, torch.Tensor([[i, j]])], dim=0)
            invalid_edges_mask = torch.concat([invalid_edges_mask, torch.Tensor([False])]).type(torch.bool)

        for _ in range(3):
            invalid_edges_mask = torch.concat([invalid_edges_mask, torch.Tensor([False])]).type(torch.bool)
        

        # compute new opposites
        ik = find_edge(i, k, new_edges)
        new_opposites[ik][new_opposites[ik] == j] = m

        jk = find_edge(j, k, new_edges)
        new_opposites[jk][new_opposites[jk] == i] = m

        if l != -1:
            il = find_edge(i, l, new_edges)
            new_opposites[il][new_opposites[il] == j] = m

            jl = find_edge(j, l, new_edges)
            new_opposites[jl][new_opposites[jl] == i] = m

    return MeshState(
        edges = new_edges.type(torch.long),
        opposites = new_opposites.type(torch.long),
        node_mask = new_node_mask.type(torch.bool),
        mesh_pos = new_mesh_pos,
        world_pos = new_world_pos,
        S_i = new_S_i
    )
"""

ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 35

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

split_ms = split_edges(ms)

exit(-1)

normal_edge_mask = split_ms.node_mask[split_ms.edges[:,0]] | split_ms.node_mask[split_ms.edges[:,1]]
border_mask = split_ms.opposites[:,1] == -1
mask = normal_edge_mask & (~border_mask)
i = split_ms.edges[:,0]
j = split_ms.edges[:,1]
k = split_ms.opposites[:,0]
l = split_ms.opposites[:,1]
M_avg = 0.25 * (split_ms.S_i[i] + split_ms.S_i[j] + split_ms.S_i[k] + split_ms.S_i[l])

u_ik = split_ms.mesh_pos[i] - split_ms.mesh_pos[k]
u_il = split_ms.mesh_pos[i] - split_ms.mesh_pos[l]
u_jk = split_ms.mesh_pos[j] - split_ms.mesh_pos[k]
u_jl = split_ms.mesh_pos[j] - split_ms.mesh_pos[l]

ujk_x_uik = u_jk[:, 0] * u_ik[:, 1] - u_jk[:, 1] * u_ik[:, 0]
uil_x_ujl = u_il[:, 0] * u_jl[:, 1] - u_il[:, 1] * u_jl[:, 0]
uil_M_ujl = torch.einsum("ei,eij,ej->e", u_il, M_avg, u_jl)
ujk_M_uik = torch.einsum("ei,eij,ej->e", u_jk, M_avg, u_ik)

delaunay_cond = (ujk_x_uik * uil_M_ujl) + (ujk_M_uik * uil_x_ujl) < -EPS
delaunay_cond[~mask] = False
print(torch.count_nonzero(delaunay_cond))

new_edges = split_ms.edges.clone()
new_opposites = split_ms.opposites.clone()
for e_id in torch.nonzero(delaunay_cond):
    i_, j_ = i[e_id].item(), j[e_id].item()
    # k_, l_ = new_opposites[find_edge(i_, j_, split_ms.edges)].unbind()
    k_, l_ = k[e_id].item(), l[e_id].item()

    new_edges[e_id] = torch.Tensor([k_, l_]).type(torch.long)
    new_opposites[e_id] = torch.Tensor([i_, j_]).type(torch.long)

    ik = find_edge(i_, k_, split_ms.edges)
    new_opposites[ik][new_opposites[ik] == j_] = l_

    jk = find_edge(j_, k_, new_edges)
    new_opposites[jk][new_opposites[jk] == i_] = l_

    il = find_edge(i_, l_, new_edges)
    new_opposites[il][new_opposites[il] == j_] = k_

    jl = find_edge(j_, l_, new_edges)
    new_opposites[jl][new_opposites[jl] == i_] = k_

exit(0)

"""
flip1_ms = MeshState(
    edges = new_edges,
    opposites = new_opposites,
    node_mask = split_ms.node_mask,
    mesh_pos = split_ms.mesh_pos,
    world_pos = split_ms.world_pos,
    S_i = split_ms.S_i
)
"""

"""
# compute common nodes between the meshes
nodes_dist = torch.cdist(mesh1["mesh_pos"][0], mesh2["mesh_pos"][0])
nodes_dist[:,~normal2_mask] = float("inf")

common_nodes = (nodes_dist < EPS).any(dim=1) & normal1_mask
delete_nodes = ~common_nodes & normal1_mask

# compute node pairs (mesh1_i <-> mesh2_j | nan if not NORMAL or not in mesh2)
node_pairs = torch.argmin(nodes_dist, dim=1)
node_pairs = np.where(
    common_nodes,
    node_pairs,
    float("nan"),
)

# compute the containing triangle for each point
sorted_cells = mesh2["cells"][BATCH]
cell_mask = normal2_mask[sorted_cells[:,0]] | normal2_mask[sorted_cells[:,1]] | normal2_mask[sorted_cells[:,2]]
sorted_cells = sorted_cells[cell_mask]
tris, bary = find_enclosing_triangle(mesh1["mesh_pos"][BATCH][delete_nodes], mesh2["mesh_pos"][BATCH], sorted_cells)
containing_tris = sorted_cells[tris]
"""

"""
DISPLAY
"""
plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2, ax3, ax4 = fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144)

ax1.set_title("Starting Mesh")
plot_mesh(ax1, mesh1)

"""
c = np.where(common_nodes[normal1_mask], "g", "r")
ax1.scatter(mesh1["mesh_pos"][0,:,0][normal1_mask], mesh1["mesh_pos"][0,:,1][normal1_mask], c=c, alpha=0.8)
idx = 0
ax1.scatter(mesh1["mesh_pos"][BATCH][delete_nodes][idx,0],    mesh1["mesh_pos"][BATCH][delete_nodes][idx,1], c="c")
ax1.scatter(mesh2["mesh_pos"][BATCH][containing_tris[idx],0], mesh2["mesh_pos"][BATCH][containing_tris[idx],1], c="c", marker="x", s=40)
"""


ax2.set_title("Splited Edges")
plot_mesh(ax2, mesh1)
for i in range(len(edges), len(split_ms.edges)):
     ax2.plot(split_ms.mesh_pos[split_ms.edges[i],0], split_ms.mesh_pos[split_ms.edges[i],1], c="r", linewidth=1, alpha=0.6)


ax3.set_title("Mesh")
for e in split_ms.edges[normal_edge_mask]:
     ax3.plot(split_ms.mesh_pos[e,0], split_ms.mesh_pos[e,1], c="w", linewidth=1)
# for e in split_ms.edges[mask][delaunay_cond]:
#      ax3.plot(split_ms.mesh_pos[e,0], split_ms.mesh_pos[e,1], c="r", linewidth=1)
for e in new_edges[delaunay_cond]:
     ax3.plot(split_ms.mesh_pos[e,0], split_ms.mesh_pos[e,1], c="r", linewidth=0.5)

ax3.set_aspect('equal')
ax3.set_ylim(1, 0)
ax3.set_axis_off()

ax4.set_title("Target Mesh")
plot_mesh(ax4, mesh2)
for e in split_ms.edges[normal_edge_mask]:
     ax4.plot(split_ms.mesh_pos[e,0], split_ms.mesh_pos[e,1], c="c", linewidth=1, alpha=0.5)

"""
# display ellipses from the sizing field
for i in range(3):  # ellipse around the selected node
    plot_ellipse(
        ax2, sizing_field2[containing_tris[idx][i]],
        mesh2["mesh_pos"][BATCH][containing_tris[idx][i]].unbind(), "c"
    )
for i in np.arange(len(normal1_mask))[delete_nodes]:    # ellipse for deleted nodes
    S = S_i[i]
    plot_ellipse(
        ax2, S,
        mesh1["mesh_pos"][BATCH][i].unbind(), "r"
    )
"""

fig.tight_layout()
plt.show()
