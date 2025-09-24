
import torch

from dataset import Dataset
from graph import Mesh, NodeType, interpolate_field, compute_world_edges
from utils import BATCH, batch_dicts
from remesher.core import MeshState, get_sizing_field_tensor
from remesher.remesher import Remesher

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from collections import namedtuple
from typing import Any


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

def plot_mesh_state(ax, ms: MeshState, color="w", alpha=1.0, linewidth=1.0):
    edge_mask = ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
    for e_id in torch.nonzero(edge_mask):
        e_id = e_id.item()
        ax.plot(ms.mesh_pos[ms.edges[e_id]][:, 0], ms.mesh_pos[ms.edges[e_id]][:, 1], color, alpha=alpha, linewidth=linewidth)
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

"""
COMPUTATION
"""
mesh1 = batch_dicts([ds[traj_len*traj + time_idx + 0]])
mesh2 = batch_dicts([ds[traj_len*traj + time_idx + 1]])

# compute_world_edges(mesh1, ds.meta)

normal1_mask = mesh1["node_type"][BATCH].flatten() == NodeType.NORMAL
normal2_mask = mesh2["node_type"][BATCH].flatten() == NodeType.NORMAL

sizing_field2 = get_sizing_field_tensor(mesh2)
S_i = interpolate_field(mesh1, mesh2, sizing_field2)


"""
S_norm = torch.sqrt(torch.sum(torch.square(ms.S_i), dim=(1,2)))
plt.style.use('dark_background')
fig, ax = plt.subplots()
plot_mesh_state(ax, ms, "w", linewidth=0.1)
col = ax.tripcolor(ms.mesh_pos[:,0][ms.node_mask], ms.mesh_pos[:,1][ms.node_mask], S_norm[ms.node_mask], cmap="inferno", vmin=0, vmax=5000)
# plt.colorbar(col)
plt.show()
"""

remesher = Remesher()
remeshed_mesh = remesher(mesh1, S_i)

"""
DISPLAY
"""
plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

ax1.set_title("Starting Mesh")
plot_mesh(ax1, mesh1)

ax2.set_title("Computed Edges")
plot_mesh(ax2, remeshed_mesh)
# plot_mesh_state(ax2, collapse_ms, color="w")

ax3.set_title("Target Mesh")
plot_mesh(ax3, remeshed_mesh, color="r", alpha=0.5)
# plot_mesh_state(ax3, collapse_ms, "r", alpha=0.5)
plot_mesh(ax3, mesh2, alpha=0.5)


fig.tight_layout()
plt.show()
