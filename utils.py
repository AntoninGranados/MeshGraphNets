import torch
import numpy as np
from pathlib import Path
from enum import IntEnum
from torch_geometric.data import HeteroData

class NodeType(IntEnum):
    NORMAL = 0
    HANDLES = 1

# Define HeteroData component names
NODE = 'node'
MESH = (NODE, 'mesh', NODE)

def get_device() -> torch.device:
    if torch.cuda.is_available():   # Nvidia GPU
        return torch.device('cuda')
    elif torch.mps.is_available():  # MacOS (Metal Performance Shaders)
        return torch.device('mps')
    return torch.device('cpu')

def save_epoch(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    parsed_path = str(path).replace('[e]', f'{epoch:03}')
    print(f"Saving checkpoint to `{parsed_path}`")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, parsed_path)

def lr_lambda(step, hyper):
    """
    Exponential decay learning rate
    """
    decay_steps = hyper['training']['steps'] * hyper['training']['decay-steps']
    decay = pow(hyper['training']['decay-rate'], step / decay_steps)
    min_lr = hyper['training']['min-lr'] / hyper['training']['start-lr']
    return max(decay, min_lr)

def faces_to_bidirectional_edges(faces: np.ndarray):
    sorted_faces = np.sort(faces, axis=-1)
    face_edges = np.concatenate([
        sorted_faces[:, [0, 1]],
        sorted_faces[:, [1, 2]],
        sorted_faces[:, [0, 2]],    # use (0,2) instead of (2,0) to preserve node index order
    ], axis=0)
    unique_edges = np.unique(face_edges, axis=0)
    bidirectional_edges = np.concatenate([unique_edges, unique_edges[:, ::-1]], axis=0)
    return bidirectional_edges

def compute_dihedral_angle(sample: HeteroData, positions: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the dihedral angle linked to each edge that is not on the mesh border.
    If the `positions` of the node is set to `None`, the mesh_pos will be used (ie. the resting position)
    """
    mesh_pos: torch.Tensor = sample[NODE].mesh_pos if positions is None else positions
    faces: torch.Tensor = sample.face_index

    face_vertices = mesh_pos[faces]
    normals = torch.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0], dim=1)
    normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    normals = normals / normal_norm.clamp(min=torch.finfo(normals.dtype).eps)

    num_faces = faces.shape[0]
    face_ids = torch.arange(num_faces, device=faces.device, dtype=torch.long).repeat_interleave(3)

    sorted_faces, _ = torch.sort(faces, dim=1)
    face_edges = torch.cat((
        sorted_faces[:, [0, 1]],
        sorted_faces[:, [1, 2]],
        sorted_faces[:, [0, 2]],
    ), dim=0)
    _, inverse, counts = torch.unique(face_edges, dim=0, return_inverse=True, return_counts=True)

    order = torch.argsort(inverse, stable=True)
    sorted_faces_for_edges = face_ids[order]

    start_indices = torch.empty(counts.numel() + 1, device=faces.device, dtype=torch.long)
    start_indices[0] = 0
    torch.cumsum(counts, dim=0, out=start_indices[1:])

    first_face = sorted_faces_for_edges[start_indices[:-1]]
    second_face = torch.full_like(first_face, -1)
    mask_two = counts > 1   # edges that are not on the border (ie. that connect 2 faces) 
    second_face[mask_two] = sorted_faces_for_edges[start_indices[:-1][mask_two] + 1]

    angles = torch.zeros(first_face.size(0), device=faces.device, dtype=mesh_pos.dtype)
    if mask_two.any():
        dots = (normals[first_face[mask_two]] * normals[second_face[mask_two]]).sum(dim=1)
        angles[mask_two] = torch.acos(dots.clamp(-1.0, 1.0))

    return angles.repeat(2).float()
