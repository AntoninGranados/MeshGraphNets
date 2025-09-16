from torch.utils.data import DataLoader
from torch_geometric.nn import radius
import torch

from dataset import Dataset
from graph import Mesh, NodeType, interpolate_field
from utils import BATCH, batch_dicts, find_enclosing_triangle
from remesher.core import get_sizing_field_tensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from tqdm import tqdm
from typing import Callable

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


ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 40

mesh1 = batch_dicts([ds[traj_len*traj + time_idx + 0]])
mesh2 = batch_dicts([ds[traj_len*traj + time_idx + 1]])

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

# compute the containing triangle for each point
sorted_cells = mesh2["cells"][BATCH]
cell_mask = normal2_mask[sorted_cells[:,0]] | normal2_mask[sorted_cells[:,1]] | normal2_mask[sorted_cells[:,2]]
sorted_cells = sorted_cells[cell_mask]
tris, bary = find_enclosing_triangle(mesh1["mesh_pos"][BATCH][delete_nodes], mesh2["mesh_pos"][BATCH], sorted_cells)
containing_tris = sorted_cells[tris]
idx = 0
ax1.scatter(mesh1["mesh_pos"][BATCH][delete_nodes][idx,0], mesh1["mesh_pos"][BATCH][delete_nodes][idx,1], c="c")
ax1.scatter(mesh2["mesh_pos"][BATCH][containing_tris[idx],0],      mesh2["mesh_pos"][BATCH][containing_tris[idx],1], c="c", marker="x", s=40)

plot_mesh(ax2, mesh2)
sizing_field2 = get_sizing_field_tensor(mesh2)
interpolated_sizing_filed = interpolate_field(mesh1, mesh2, sizing_field2)

# display ellipses from the sizing field
for i in range(3):  # ellipse around the selected node
    plot_ellipse(
        ax2, sizing_field2[containing_tris[idx][i]],
        mesh2["mesh_pos"][BATCH][containing_tris[idx][i]].unbind(), "c"
    )
for i in np.arange(len(normal1_mask))[delete_nodes]:    # ellipse for deleted nodes
    S = interpolated_sizing_filed[i]
    plot_ellipse(
        ax2, S,
        mesh1["mesh_pos"][BATCH][i].unbind(), "r"
    )

fig.tight_layout()
plt.show()
