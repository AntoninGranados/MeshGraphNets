
import torch
from torch_geometric.data import HeteroData

from dataset import Dataset, move_batch_to
from graph import Mesh, NodeType, interpolate_field, compute_world_edges, generate_graph, compute_mesh_edges
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
from copy import deepcopy


def plot_mesh(ax, mesh: Mesh, color="w", alpha=1.0, linewidth=1.0):
    triangles = mesh["cells"][BATCH]

    type_mask = (mesh["node_type"][BATCH][triangles[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][triangles[:,2]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask = ~type_mask

    triang = mtri.Triangulation(mesh["mesh_pos"][BATCH][:,0], mesh["mesh_pos"][BATCH][:,1], triangles)
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


def generate_uniform_mesh(src_mesh: Mesh, N: int = 5) -> Mesh:
    uniform_mesh = {k: v.clone() for k,v in src_mesh.items()}

    node_mask = (src_mesh["node_type"][BATCH] != NodeType.OBSTACLE).squeeze(-1)
    cell_mask = node_mask[src_mesh["cells"][BATCH][...,0]] | node_mask[src_mesh["cells"][BATCH][...,1]] | node_mask[src_mesh["cells"][BATCH][...,2]]

    uniform_mesh["node_type"] = uniform_mesh["node_type"][BATCH][~node_mask]
    uniform_mesh["mesh_pos"] = uniform_mesh["mesh_pos"][BATCH][~node_mask]
    uniform_mesh["cells"] = uniform_mesh["cells"][BATCH][~cell_mask]

    c = N-1

    mesh_pos = []
    node_type = []
    for j in range(N):
        for i in range(N):
            mesh_pos.append([i/c, j/c])
            if j == 0 and (i == 0 or i == c):
                node_type.append([NodeType.HANDLE])
            else:
                node_type.append([NodeType.NORMAL])

    uniform_mesh["mesh_pos"] = torch.concat([uniform_mesh["mesh_pos"], torch.Tensor(mesh_pos)])
    uniform_mesh["node_type"] = torch.concat([uniform_mesh["node_type"], torch.Tensor(node_type)])

    node_idx = lambda i, j : j*N+i + torch.count_nonzero(~node_mask)

    cells = []
    for i in range(N):
        for j in range(N):
            border = (i == 0) | (j == 0)
            even = ((i + j) % 2 == 0)

            A = node_idx(i-1, j-1)
            B = node_idx(  i, j-1)
            C = node_idx(i-1,   j)
            D = node_idx(  i,   j)

            if not border:
                if even:
                    cells.append([A,B,D])
                    cells.append([A,D,C])
                else:
                    cells.append([A,B,C])
                    cells.append([B,D,C])
    uniform_mesh["cells"] = torch.concat([uniform_mesh["cells"], torch.Tensor(cells)]).long()

    uniform_mesh = batch_dicts([uniform_mesh])
    new_world_pos = interpolate_field(uniform_mesh, src_mesh, src_mesh["world_pos"])
    new_world_pos[0,torch.nonzero(~node_mask).squeeze(-1)] = src_mesh["world_pos"][0,~node_mask]
    uniform_mesh["world_pos"] = new_world_pos

    new_prev_world_pos = interpolate_field(uniform_mesh, src_mesh, src_mesh["prev|world_pos"])
    new_prev_world_pos[0,torch.nonzero(~node_mask).squeeze(-1)] = src_mesh["prev|world_pos"][0,~node_mask]
    uniform_mesh["prev|world_pos"] = new_prev_world_pos

    return uniform_mesh



"""
#! ====================== Uniform Mesh from the Ground Truth dataset ======================
with open(Path(".", "hyperparam.json"), "r") as file:
        hyper = json.loads(file.read())

ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")
loader = torch.utils.data.DataLoader(ds, batch_size = 1, shuffle = True)

plt.style.use('dark_background')
fig = plt.figure()
fig.set_size_inches((6, 4))
fig.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.70)
plt.rcParams["font.family"] = "monospace"

traj_len = ds.meta["trajectory_length"]
traj = 3
time_idx = 0

max_frame = 200
for i in tqdm(range(max_frame), "Computing frames", file=sys.stdout):
    fig.clf()
    
    # Ground Truth
    mesh = batch_dicts([ds[traj_len*traj + time_idx + i]])
    world_pos = mesh["world_pos"][BATCH]
    triangles = mesh["cells"][BATCH]
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_trisurf(world_pos[:,0], world_pos[:,1], world_pos[:,2], triangles=triangles, color="white", alpha=1)
    ax1.set_xlim(-1, 0)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(-1.5, -0.5)
    ax1.set_axis_off()
    ax1.set_aspect('equal')

    ax1.set_title(f"Ground Truth", color="gray", fontsize=12)
    ax1.patch.set_edgecolor("w")
    ax1.patch.set_facecolor("k")
    ax1.patch.set_alpha(0.2)
    ax1.patch.set_linewidth(1)

    # Uniform mesh
    uniform_mesh = generate_uniform_mesh(mesh, N=30)
    world_pos = uniform_mesh["world_pos"][BATCH]
    triangles = uniform_mesh["cells"][BATCH]
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_trisurf(world_pos[:,0], world_pos[:,1], world_pos[:,2], triangles=triangles, color="white", alpha=1)
    ax2.set_xlim(-1, 0)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(-1.5, -0.5)
    ax2.set_axis_off()
    ax2.set_aspect('equal')

    ax2.set_title(f"Uniform Mesh", color="gray", fontsize=12)
    ax2.patch.set_edgecolor("w")
    ax2.patch.set_facecolor("k")
    ax2.patch.set_alpha(0.2)
    ax2.patch.set_linewidth(1)

    node_count_1 = len(mesh["node_type"][0])
    node_count_2 = len(uniform_mesh["node_type"][0])

    fig.suptitle("Ground Thruth to Uniform Mesh", fontsize=16, y=0.95)
    fig.text(0.5, 0.85, f"Frame {i+1:>3}/{max_frame} — Node count ratio {int(100 * node_count_2 / node_count_1): >3}%", ha="center", fontsize=12, color="white")
    fig.patch.set_facecolor('k')
    fig.patch.set_alpha(0.2)

    # plt.savefig(f"img/frame_{i:03}.png", dpi=500)
    plt.draw()
    plt.pause(0.1)
"""


"""
#! ====================== ROLLOUT ON UNIFORM MESH ======================
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
traj = 4 
time_idx = 0

pred_mesh = generate_uniform_mesh(batch_dicts([ds[traj_len*traj + time_idx]]), N=30)
prev_mesh = {k: v.clone().to(device) for k,v in pred_mesh.items()}

plt.style.use('dark_background')
fig = plt.figure()
fig.set_size_inches((6, 4))
fig.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.75)
plt.rcParams["font.family"] = "monospace"

max_frame = 300
for i in tqdm(range(max_frame), "Computing frames", file=sys.stdout):

    # target
    mesh = batch_dicts([ds[traj_len*traj + time_idx + i]])

    # prediction
    obstacle = torch.nonzero(prev_mesh["node_type"] == NodeType.OBSTACLE, as_tuple=True)[1]
    prev_mesh["world_pos"][:,obstacle] = mesh["world_pos"].clone().to(device)[:,obstacle]
    with torch.no_grad():
        pred_mesh_i = model(prev_mesh, ds.meta)
    prev_mesh = pred_mesh_i
    mesh_i_cpu = {k: v.detach().cpu() for k,v in prev_mesh.items()}
    torch.mps.empty_cache()

    fig.clf()

    # display prediction
    world_pos = mesh_i_cpu["world_pos"][BATCH]
    triangles = mesh_i_cpu["cells"][BATCH]
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_trisurf(world_pos[:,0], world_pos[:,1], world_pos[:,2], triangles=triangles, color="white", alpha=1)
    ax1.set_xlim(-1, 0)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(-1.5, -0.5)
    ax1.set_axis_off()
    ax1.set_aspect('equal')

    ax1.set_title(f"Prediction", color="white", fontsize=12)
    ax1.patch.set_edgecolor("w")
    ax1.patch.set_facecolor("k")
    ax1.patch.set_alpha(0.2)
    ax1.patch.set_linewidth(1)

    # display target
    world_pos = mesh["world_pos"][BATCH]
    triangles = mesh["cells"][BATCH]
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_trisurf(world_pos[:,0], world_pos[:,1], world_pos[:,2], triangles=triangles, color="white", alpha=1)
    ax2.set_xlim(-1, 0)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(-1.5, -0.5)
    ax2.set_axis_off()
    ax2.set_aspect('equal')

    ax2.set_title(f"Target", color="white", fontsize=12)
    ax2.patch.set_edgecolor("w")
    ax2.patch.set_facecolor("k")
    ax2.patch.set_alpha(0.2)
    ax2.patch.set_linewidth(1)

    fig.suptitle("Sphere Dynamic", fontsize=16, y=0.95)
    fig.text(0.5, 0.85, f"Frame {i+1:>3}/{max_frame}", ha="center", fontsize=12, color="white")
    fig.patch.set_facecolor('k')
    fig.patch.set_alpha(0.2)

    plt.savefig(f"img/frame_{i:03}.png", dpi=400)
    plt.draw()
    plt.pause(0.1)
"""

with open(Path(".", "hyperparam.json"), "r") as file:
        hyper = json.loads(file.read())

ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid")
mesh = batch_dicts([ds[100]])

node = "node"
mesh_edge = (node, "mesh", node)
world_edge = (node, "world", node)

# Node edges
mesh_pos  = mesh["mesh_pos"][BATCH]
world_pos = mesh["world_pos"][BATCH]
prev_world = mesh["prev|world_pos"][BATCH]
node_type = mesh["node_type"][BATCH].squeeze(-1).long()

data = HeteroData()
data[node].mesh_pos = mesh_pos
data[node].world_pos = world_pos
data[node].type = node_type.unsqueeze(-1)

velocities = world_pos - prev_world
types_onehot = torch.nn.functional.one_hot(node_type, NodeType.COUNT).to(velocities.dtype)
data[node].x = torch.cat([velocities, types_onehot.to(velocities.dtype)], dim=-1)

# Mesh edges
mesh_edges, opposites = compute_mesh_edges(mesh)
mesh_edges = mesh_edges[BATCH]
opposites = opposites[BATCH]  

senders, receivers = mesh_edges[...,0].long(), mesh_edges[...,1].long()
edge_index = torch.stack([senders, receivers], dim=0)
data[mesh_edge].edge_index = edge_index

rel_world_pos = world_pos[senders] - world_pos[receivers]
rel_mesh_pos  = mesh_pos[senders] - mesh_pos[receivers]
mesh_edge_attr = torch.cat([
    rel_world_pos,
    torch.norm(rel_world_pos, dim=-1, keepdim=True),
    rel_mesh_pos,
    torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
], dim=-1)
data[mesh_edge].x = mesh_edge_attr
data[mesh_edge].opposites = opposites

# World edges
world_edges = compute_world_edges(mesh, meta=ds.meta, edges=mesh_edges.unsqueeze(0))
world_edges = world_edges[BATCH]
senders, receivers = world_edges[...,0], world_edges[...,1]
edge_index = torch.stack([senders, receivers], dim=0)
data[world_edge].edge_index = edge_index

rel_world_pos = (torch.index_select(world_pos, 0, senders) -
                    torch.index_select(world_pos, 0, receivers))
world_edge_features = torch.concat([
    rel_world_pos,
    torch.norm(rel_world_pos, dim=-1, keepdim=True),
], dim=-1)
data[world_edge].x = world_edge_features

# Remove unused edges (edges connected two non-normal nodes)
active_edges = torch.nonzero(((data[node].type[data[mesh_edge].edge_index]).squeeze(-1) == NodeType.NORMAL).any(0)).squeeze(-1)
data = data.edge_subgraph({mesh_edge: active_edges})

# latent edge features
v_i = data[node].x[data[mesh_edge].edge_index[0]]
v_j = data[node].x[data[mesh_edge].edge_index[1]]
features = torch.concat([ v_i, v_j, data[mesh_edge].x], dim=-1)

# latent node features
features = [data[node].x]

mesh_shape = (data[node].x.shape[0], data[mesh_edge].x.shape[1])
mesh_sum_eij = torch.zeros(mesh_shape).index_add_(0, data[mesh_edge].edge_index[1], data[mesh_edge].x)
features.append(mesh_sum_eij)

if world_edge in data.edge_types:
    world_shape = (data[node].x.shape[0], data[world_edge].x.shape[1])
    world_sum_eij = torch.zeros(world_shape).index_add_(0, data[world_edge].edge_index[1], data[world_edge].x)
    features.append(world_sum_eij)

features = torch.concat(features, dim=-1)
print(features.shape)
