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
from typing import Callable

# TODO: handle batch size > 1
BATCH = 0   #! where we use only the first batch and should be handle corretly for multiple batch

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

def get_triangle_sarea(tris: torch.Tensor) -> torch.Tensor:
    """Compute the signed area of each triangle in a list"""
    select = lambda tensor, dim, idx : torch.index_select(tensor, dim, torch.Tensor([idx]).type(torch.int32))

    BA = select(tris,-2,1) - select(tris,-2,0)
    CB = select(tris,-2,2) - select(tris,-2,1)
    det = select(BA,-1,0) * select(CB,-1,1) - select(BA,-1,1) * select(CB,-1,0)
    det = det.reshape(tris.shape[:-2])
    return det/2

def get_barycentric_coord(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Compute the barycentric coordinates of point `P` with repect to the triangle `ABC` (each argument can be a list of multiple points)"""
    ABC = torch.stack([A, B, C], dim=-2)
    PBC = torch.stack([P, B, C], dim=-2)
    APC = torch.stack([A, P, C], dim=-2)
    ABP = torch.stack([A, B, P], dim=-2)
    tris = torch.stack([ABC, PBC, APC, ABP], dim=-3)
    sareas = get_triangle_sarea(tris)
    sareaABC = sareas[:,0].unsqueeze(-1)

    # if the triangle is flat, return the parmameter if the linear interpolation between A and B
    return torch.where(
        sareaABC == 0,  # flat triangle (division by 0)
        torch.stack([
            torch.norm(P-A, dim=-1)/torch.norm(B-A, dim=-1),
            torch.norm(P-B, dim=-1)/torch.norm(B-A, dim=-1),
            torch.zeros(P.shape[0])
        ], dim=-1),
        sareas[:,1:] / sareaABC,
    )

# TODO: handle batch size > 1
def interpolate_field(from_mesh: Mesh, targ_mesh: Mesh, field: torch.Tensor, interpolation_filter: Callable[[torch.Tensor], torch.Tensor]|None=None) -> torch.Tensor:
    """Interpolates the field `field` comming from the target mesh `targ_mesh` using the nodes in `from_mesh`"""
    from_normal_mask = from_mesh["node_type"][BATCH].flatten() == NodeType.NORMAL
    targ_normal_mask = targ_mesh["node_type"][BATCH].flatten() == NodeType.NORMAL

    nodes_dist = torch.cdist(from_mesh["mesh_pos"][BATCH], targ_mesh["mesh_pos"][BATCH])
    nodes_dist[:,~targ_normal_mask] = float("inf")

    common_nodes_mask = (nodes_dist < 1e-3).any(dim=-1) & from_normal_mask
    delete_nodes_mask = ~common_nodes_mask & from_normal_mask

    # compute node pairs (from_mesh_i <-> targ_mesh_j | nan if not NORMAL or not in the target mesh)
    node_pairs = torch.argmin(nodes_dist, dim=-1)
    node_pairs = np.where(
        common_nodes_mask,
        node_pairs,
        float("nan")
    )

    # compute the 3 closest nodes for each deleted node (not in mesh2) and its barycentric coordinates
    delete_nodes_dist = torch.where(
        targ_normal_mask.unsqueeze(0),
        nodes_dist[delete_nodes_mask],
        float("inf")
    )
    closest = torch.topk(delete_nodes_dist, 3, largest=False, dim=1).indices
    delete_nodes_pos = mesh1["mesh_pos"][0][delete_nodes]
    lambdas = get_barycentric_coord(
        delete_nodes_pos,
        targ_mesh["mesh_pos"][BATCH][closest[:,0]],
        targ_mesh["mesh_pos"][BATCH][closest[:,1]],
        targ_mesh["mesh_pos"][BATCH][closest[:,2]]
    )
    lambdas_shape = (-1, 3, *[1]*(len(field.shape)-1))
    lambdas = lambdas.reshape(lambdas_shape)

    # fill the interpolated field
    new_field = torch.zeros((from_normal_mask.shape[0], *field.shape[1:]))
    new_field[common_nodes_mask] = field[node_pairs[common_nodes_mask]]

    interpolatied_field = torch.sum(field[closest] * lambdas, dim=1)
    if interpolation_filter is not None:
        interpolatied_field = torch.stack([
            interpolation_filter(e) for e in interpolatied_field.unbind(dim=BATCH)
        ], dim=BATCH)
    new_field[delete_nodes_mask] = interpolatied_field

    return new_field

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

def closest_SDP(S: torch.Tensor) -> torch.Tensor:
    """Compute the closest symetric definite matrix if `S` is not one"""
    eigenvalues = torch.linalg.eigvals(S)
    if torch.all(torch.real(eigenvalues) >= 0):
        return S
    
    X = 0.5 * (S + S.T)
    eigenvalues, eigenvectors = torch.linalg.eig(X)
    eigenvalues = torch.real(eigenvalues)
    eigenvectors = torch.real(eigenvectors)
    eigenvalues = torch.maximum(eigenvalues, torch.zeros_like(eigenvalues))
    eigenvalues = torch.minimum(eigenvalues, torch.ones_like(eigenvalues)*10/0.05)   #! hardcoded heuristic
    Z = torch.diag(eigenvalues) + torch.eye(2)*10/0.05  #! hardcoded heuristic
    return eigenvectors @ Z @ eigenvectors.T


# TODO: handle batch size > 1
# TODO: do not compute for nodes where NodeType != NORMAL
def get_sizing_field_tensor(mesh: Mesh):
    edges = cells_to_edges(mesh)

    senders, receivers = edges[0,:], edges[1,:]
    u_ij = mesh["mesh_pos"][BATCH][receivers] - mesh["mesh_pos"][BATCH][senders]
    aggregated_u_ij = aggregate(u_ij, senders)

    def compute_S(edges: torch.Tensor) -> torch.Tensor:
        cpu_edges = edges.detach().cpu()
        cpu_edges = edges[~edges.isnan().any(dim=-1)]
        S = miniellipse(cpu_edges.tolist())
        S = torch.Tensor(S)
        S = closest_SDP(S)

        return S

    return torch.stack([
        compute_S(edges) for edges in torch.unbind(aggregated_u_ij, dim=0)
    ], dim=0)



ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 48

mesh1 = {
    k: v.unsqueeze(0)
    for k, v in ds[traj_len*traj+time_idx].items()
}
mesh2 = {
    k: v.unsqueeze(0)
    for k, v in ds[traj_len*traj+time_idx+1].items()
}

normal1_mask = mesh1["node_type"][BATCH].flatten() == NodeType.NORMAL
normal2_mask = mesh2["node_type"][BATCH].flatten() == NodeType.NORMAL

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
delete_nodes_dist = torch.where(
    normal2_mask.unsqueeze(0),
    nodes_dist[delete_nodes], float("inf")
)
closest = torch.topk(delete_nodes_dist, 3, largest=False, dim=1).indices
temp_idx = 8
ax1.scatter(mesh1["mesh_pos"][BATCH][delete_nodes][temp_idx,0], mesh1["mesh_pos"][BATCH][delete_nodes][temp_idx,1], c="c")
ax1.scatter(mesh2["mesh_pos"][BATCH][closest[temp_idx],0],      mesh2["mesh_pos"][BATCH][closest[temp_idx],1], c="c", marker="x")


plot_mesh(ax2, mesh2)
sizing_field2 = get_sizing_field_tensor(mesh2)

plot_ellipse(
    ax2, sizing_field2[closest[temp_idx][0]],
    mesh2["mesh_pos"][BATCH][closest[temp_idx][0]].unbind(), "c"
)
plot_ellipse(
    ax2, sizing_field2[closest[temp_idx][1]],
    mesh2["mesh_pos"][BATCH][closest[temp_idx][1]].unbind(), "c"
)
plot_ellipse(
    ax2, sizing_field2[closest[temp_idx][2]],
    mesh2["mesh_pos"][BATCH][closest[temp_idx][2]].unbind(), "c"
)

interpolated_sizing_filed = interpolate_field(mesh1, mesh2, sizing_field2)
# display an ellipse from the sizing field
for i in np.arange(len(normal1_mask))[delete_nodes]:
    S = interpolated_sizing_filed[i]
    plot_ellipse(
        ax2, S,
        mesh1["mesh_pos"][BATCH][i].unbind(), "r"
    )

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
