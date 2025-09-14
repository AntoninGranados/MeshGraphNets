from torch.utils.data import DataLoader
from torch_geometric.nn import radius
import torch

from dataset import Dataset
from graph import Mesh, NodeType, cells_to_edges, compute_world_edges
from miniellipse import miniellipse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from tqdm import tqdm

def plot_mesh(ax, mesh: Mesh):
    triangles = mesh["cells"][0]
    type_mask = (mesh["node_type"][0][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][0][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][0][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()

    triang = mtri.Triangulation(mesh["mesh_pos"][0][:,0], mesh["mesh_pos"][0][:,1], triangles, ~type_mask)
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

def get_triangle_sarea(tris: torch.Tensor) -> torch.Tensor:
    """Compute the signed area of each triangle in a list of list"""
    BA = tris[:,:,1] - tris[:,:,0]
    CB = tris[:,:,2] - tris[:,:,1]
    det = BA[:,:,0] * CB[:,:,1] - BA[:,:,1] * CB[:,:,0]
    return det/2

def get_barycentric_coord(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    ABC = torch.stack([A, B, C], dim=1)
    PBC = torch.stack([P, B, C], dim=1)
    APC = torch.stack([A, P, C], dim=1)
    ABP = torch.stack([A, B, P], dim=1)
    tris = torch.stack([ABC, PBC, APC, ABP], dim=1)
    sareas = get_triangle_sarea(tris)
    sareaABC = sareas[:,0].unsqueeze(-1)

    # if the triangle is flat, return the parmameter if the linear interpolation between A and B
    return torch.where(
        sareaABC == 0,  # flat triangle (division by 0)
        torch.stack([torch.norm(P-A, dim=-1)/torch.norm(B-A, dim=-1), torch.norm(P-B, dim=-1)/torch.norm(B-A, dim=-1), torch.zeros(P.shape[0])], dim=-1),
        sareas[:,1:] / sareaABC,
    )

# https://stackoverflow.com/questions/78369381/group-pytorch-feature-tensors-according-to-labels-by-concatenation
def aggregate(x, index):
    index_count = torch.bincount(index)

    fill_count = index_count.max() - index_count
    fill_nan = torch.ones_like(x[0]).repeat(int(fill_count.sum().item()),1) * float("nan")
    fill_index = torch.arange(0, fill_count.shape[0]).repeat_interleave(fill_count)

    index_ = torch.cat([index, fill_index], dim = 0)
    x_ = torch.cat([x, fill_nan], dim = 0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], int(index_count.max().item()), -1)
    return x_

# TODO: handle batch size > 1
# TODO: do not compute for nodes where NodeType != NORMAL
def get_sizing_field_tensor(mesh: Mesh):
    edges = cells_to_edges(mesh)

    senders, receivers = edges[0,:], edges[1,:]
    u_ij = mesh["mesh_pos"][0][receivers] - mesh["mesh_pos"][0][senders]
    aggregated_u_ij = aggregate(u_ij, senders)

    def compute_S(edges: torch.Tensor) -> torch.Tensor:
        cpu_edges = edges.detach().cpu()
        cpu_edges = edges[~edges.isnan().any(dim=-1)]
        S = miniellipse(cpu_edges.tolist())
        return torch.Tensor(S)

    return torch.stack([
        compute_S(edges) for edges in torch.unbind(aggregated_u_ij, dim=0)
    ], dim=0)



ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 30

mesh1 = {
    k: v.unsqueeze(0)
    for k, v in ds[traj_len*traj+time_idx].items()
}
mesh2 = {
    k: v.unsqueeze(0)
    for k, v in ds[traj_len*traj+time_idx+1].items()
}

normal1_mask = mesh1["node_type"][0].flatten() == NodeType.NORMAL
normal2_mask = mesh2["node_type"][0].flatten() == NodeType.NORMAL

plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

plot_mesh(ax1, mesh1)
# compute common nodes between the meshes
nodes_dist = torch.cdist(mesh1["mesh_pos"][0], mesh2["mesh_pos"][0])
nodes_dist[:,~normal2_mask] = float("inf")

common_nodes = (nodes_dist < 1e-3).any(dim=1) & normal1_mask
delete_nodes = ~common_nodes & normal1_mask

# compute node pairs (mesh1_i <-> mesh2_j | nan if not NORMAL or not in mesh2)
node_pairs = torch.argmin(nodes_dist, dim=1)
node_pairs = np.where(
    common_nodes,
    node_pairs,
    float("nan"),
)

c = np.where(common_nodes[normal1_mask], "g", "r")
ax1.scatter(mesh1["mesh_pos"][0,:,0][normal1_mask], mesh1["mesh_pos"][0,:,1][normal1_mask], c=c, alpha=0.8)

# compute the 3 closest nodes for each deleted node (not in mesh2) and its barycentric coordinates
rm_nodes_dist = torch.where(
    normal2_mask.unsqueeze(0),
    nodes_dist[delete_nodes], float("inf")
)
closest = torch.topk(rm_nodes_dist, 3, largest=False, dim=1).indices
rm_nodes_pos = mesh1["mesh_pos"][0][delete_nodes]
lambdas = get_barycentric_coord(
    rm_nodes_pos,
    mesh2["mesh_pos"][0,closest[:,0]],
    mesh2["mesh_pos"][0,closest[:,1]],
    mesh2["mesh_pos"][0,closest[:,2]]
)
temp_idx = 0
ax1.scatter(rm_nodes_pos[temp_idx, 0], rm_nodes_pos[temp_idx, 1], c="c")
ax1.scatter(mesh2["mesh_pos"][0][closest[temp_idx],0], mesh2["mesh_pos"][0][closest[temp_idx],1], c="c", marker="x")


plot_mesh(ax2, mesh2)
sizing_field2 = get_sizing_field_tensor(mesh2)
# display an ellipse from the sizing field
interpolated_sizing_filed = torch.sum(sizing_field2[closest] * lambdas.unsqueeze(-1).unsqueeze(-1), dim=1)
for i in range(rm_nodes_pos.shape[0]):
    plot_ellipse(
        ax2, interpolated_sizing_filed[i],
        rm_nodes_pos[i].unbind(), "r"
    )
sizing_field1 = np.zeros((normal1_mask.shape[0], 2, 2))
sizing_field1[common_nodes] = sizing_field2[0]
fig.tight_layout()
plt.show()


"""
ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 3

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

frame_count = traj_len
for i in range(frame_count):
    mesh = ds[i+traj_len*traj]
    # for i in range(10):
    #     print(i if i not in NodeType else NodeType(i).name, torch.any(mesh["node_type"] == i).item())
    
    world_pos = mesh["world_pos"]
    triangles = mesh["cells"]
    node_type = mesh["node_type"]
    batch_mesh = {k: v.unsqueeze(0) for k, v in mesh.items()}
    neighbours = compute_world_edges(batch_mesh, ds.meta)

    ax.cla()
    ax.plot_trisurf(world_pos[:,0], world_pos[:,1], world_pos[:,2], triangles=triangles, alpha=0.15, color="w", edgecolor="w", linewidth=0.3)
    
    # c = np.where(node_type==NodeType.NORMAL, "k", np.where(node_type==NodeType.OBSTACLE, "b", "g")).flatten()
    # s = np.where(node_type==NodeType.NORMAL, 0, np.where(node_type==NodeType.OBSTACLE, 0.1, 4)).flatten()
    c = np.where(node_type==NodeType.HANDLE, "g", "k").flatten()
    s = np.where(node_type==NodeType.HANDLE, 10, 0).flatten()
    ax.scatter(world_pos[:,0], world_pos[:,1], world_pos[:,2], c=c, s=s, alpha=0.8)

    for n in neighbours:
        # world_edge = world_pos[neighbours[np.random.randint(0, neighbours.shape[0])]]
        world_edge = world_pos[n]
        ax.plot(world_edge[:,0], world_edge[:,1], world_edge[:,2], c="r", linewidth=0.4)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.7, 0.3)
    ax.set_zlim(-0.9, 0.1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(f"World Edges\nFrame: {i+1:>4}/{frame_count}")
    # plt.show()
    plt.draw()
    plt.pause(0.05)

    # plt.savefig(Path("img", f"frame_{i:03}.png"), dpi=500)
"""
