
import torch

from dataset import Dataset, move_batch_to
from graph import Mesh, NodeType, interpolate_field, compute_world_edges, generate_graph
from utils import BATCH, batch_dicts
from remesher.core import MeshState, get_sizing_field_tensor, to_meshstate
from remesher.remesher import Remesher
from network.model import Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse as PltEllipse
from pathlib import Path
from collections import namedtuple
from typing import Any
import json
from tqdm import tqdm
import sys


def plot_mesh(ax, mesh: Mesh, color="w", alpha=1.0, linewidth=1.0):
    triangles = mesh["cells"][BATCH]

    type_mask = (mesh["node_type"][BATCH][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask = ~type_mask

    triang = mtri.Triangulation(mesh["mesh_pos"][BATCH][:,0], mesh["mesh_pos"][BATCH][:,1], triangles, type_mask)
    ax.triplot(triang, linewidth=linewidth, color=color, alpha=alpha)
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


with open(Path(".", "hyperparam.json"), "r") as file:
        hyper = json.loads(file.read())

ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")
loader = torch.utils.data.DataLoader(ds, batch_size = 1, shuffle = True)

device = torch.device("mps")
checkpoint_path = sorted(list(Path(hyper["training"]["checkpoint-dir"]).glob("*.pt")))[-1]
last_checkpoint = torch.load(checkpoint_path, map_location=device)

model = Model(device)
model.load_state_dict(last_checkpoint['model_state_dict'])
model.to(device)
model.eval()

traj_len = ds.meta["trajectory_length"]
traj = 0
time_idx = 0

"""
COMPUTATION
"""

mesh1 = batch_dicts([ds[traj_len*traj + time_idx + 30]])
mesh2 = batch_dicts([ds[traj_len*traj + time_idx + 100]])
normal1_mask = mesh1["node_type"][BATCH].flatten() == NodeType.NORMAL
# normal2_mask = mesh2["node_type"][BATCH].flatten() == NodeType.NORMAL

sizing_field1 = get_sizing_field_tensor(mesh1)
sizing_field2 = get_sizing_field_tensor(mesh2)
# S_i = interpolate_field(mesh1, mesh2, sizing_field2)


ms1 = to_meshstate(mesh1, sizing_field1)
ms2 = to_meshstate(mesh2, sizing_field2)

plt.style.use('dark_background')
fig = plt.figure()
fig.set_size_inches((3*5, 4))
fig.subplots_adjust(left=0.02, bottom=0, right=0.98, top=0.75)
plt.rcParams["font.family"] = "monospace"

f = 0
for i in range(5):
    f += i * 20

    ax = fig.add_subplot(1, 5, i+1)
    mesh = batch_dicts([ds[traj_len*traj + time_idx + f]])
    sizing_field = get_sizing_field_tensor(mesh)
    ms = to_meshstate(mesh, sizing_field)

    S_norm = torch.sqrt(torch.sum(torch.square(ms.S_i), dim=(1,2)))
    S_norm /= torch.max(S_norm)
    plot_mesh_state(ax, ms, "w", linewidth=0.1)
    col = ax.tripcolor(
        ms.mesh_pos[:,0][ms.node_mask],
        ms.mesh_pos[:,1][ms.node_mask],
        S_norm[ms.node_mask],
        cmap="inferno"
    )

    ax.set_title(f"Frame {f}", color="gray", fontsize=12)
    ax.patch.set_edgecolor("w")
    ax.patch.set_facecolor("k")
    ax.patch.set_alpha(0.2)
    ax.patch.set_linewidth(1)

fig.suptitle("Magnitude of the Sizing Field (normalized)\nacross multitple frames", fontsize=16, y=0.95)
fig.text(0.5, 0.75, f"Simulation n°{traj} of the sphere_dynamic/valid set", ha="center", fontsize=12, color="white")
fig.patch.set_facecolor('k')
fig.patch.set_alpha(0.2)

plt.savefig("sizing_field.png", dpi=500)

exit(0)

# remesher = Remesher()
# remeshed_mesh = remesher(mesh1, S_i)

# world_edges = compute_world_edges(mesh1, ds.meta)[BATCH]
# graph = generate_graph(mesh1, ds.meta)

"""
DISPLAY
"""
plt.style.use('dark_background')
fig = plt.figure()
ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")

types = mesh1["node_type"][BATCH]
triangles = mesh1["cells"][BATCH]
# collider_mask = (types[triangles[...,0]] != NodeType.NORMAL) & (types[triangles[...,1]] != NodeType.NORMAL) & (types[triangles[...,2]] != NodeType.NORMAL)
# collider_mask = collider_mask.squeeze(-1)

ax1.plot_trisurf(
    mesh1["target|world_pos"][BATCH][:,0],
    mesh1["target|world_pos"][BATCH][:,1],
    mesh1["target|world_pos"][BATCH][:,2],
    triangles=triangles,
    color=(0.,0.,0.,0.), 
    edgecolor=(1.,0.5,0.5,0.3),
    linewidth=0.5
)
ax1.set_aspect('equal')
ax1.set_axis_off()


# ====

ax2.plot_trisurf(
    mesh1["world_pos"][BATCH][:,0],
    mesh1["world_pos"][BATCH][:,1],
    mesh1["world_pos"][BATCH][:,2],
    triangles=triangles,
    color=(0.,0.,0.,0.), 
    edgecolor=(1.,0.5,0.5,0.3),
    linewidth=0.5
)
ax2.set_aspect('equal')
ax2.set_axis_off()

# plot_mesh(ax2, mesh1, color="r", alpha=0.5)
# plot_mesh(ax2, mesh2, alpha=0.5)

fig.tight_layout()
plt.show()
