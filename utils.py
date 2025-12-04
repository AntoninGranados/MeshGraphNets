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
    degenerate = normal_norm.squeeze(-1) < torch.finfo(normals.dtype).eps
    normals = normals / normal_norm.clamp(min=torch.finfo(normals.dtype).eps)
    normals[degenerate] = 0.0
    # normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    num_faces = faces.shape[0]
    face_ids = torch.arange(num_faces, dtype=torch.long).to(faces.device).repeat_interleave(3)

    sorted_faces, _ = torch.sort(faces, dim=1)
    face_edges = torch.cat((
        sorted_faces[:, [0, 1]],
        sorted_faces[:, [1, 2]],
        sorted_faces[:, [0, 2]],
    ), dim=0)
    num_nodes = mesh_pos.shape[0]
    edge_keys = face_edges[:, 0] * num_nodes + face_edges[:, 1]

    order = torch.argsort(edge_keys, stable=True)
    sorted_keys = edge_keys[order]
    sorted_faces_for_edges = face_ids[order]

    change = torch.ones_like(sorted_keys, dtype=torch.bool)
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    start_indices = change.nonzero(as_tuple=False).flatten()

    counts = torch.empty(start_indices.numel(), dtype=torch.long).to(faces.device)
    counts[:-1] = start_indices[1:] - start_indices[:-1]
    counts[-1] = sorted_keys.numel() - start_indices[-1]

    first_face = sorted_faces_for_edges[start_indices]
    second_face = torch.full_like(first_face, -1)
    mask_two = counts > 1   # edges that are not on the border (ie. that connect 2 faces) 
    second_face[mask_two] = sorted_faces_for_edges[start_indices[mask_two] + 1]

    angles = torch.zeros(first_face.size(0), dtype=mesh_pos.dtype).to(faces.device)
    if mask_two.any():
        dots = (normals[first_face[mask_two]] * normals[second_face[mask_two]]).sum(dim=1)
        dots = dots.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angles[mask_two] = torch.acos(dots)

    return angles.repeat(2).float()

def inv_mps(matrices: torch.Tensor) -> torch.Tensor:
    """
    There are issues with torch.linalg.inv with the MPS backend (the gradient might be NaN)
    """
    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    valid = det > 1e-12

    inv = torch.zeros_like(matrices)
    inv[valid, 0, 0] = matrices[valid, 1, 1]
    inv[valid, 1, 1] = matrices[valid, 0, 0]
    inv[valid, 0, 1] = -matrices[valid, 0, 1]
    inv[valid, 1, 0] = -matrices[valid, 1, 0]
    inv[valid] = inv[valid] / det[valid].unsqueeze(-1).unsqueeze(-1)

    return inv

def btrace(matrices: torch.Tensor) -> torch.Tensor:
    """
    Compute the trace of each matrix in the batch `matrices` of shape [B, N, N].
    The result is a tensor of shape [B]
    """
    return matrices.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def compute_green_strain(sample: HeteroData, positions: torch.Tensor) -> torch.Tensor:
    """
    Compute the Green strain tensor for each triangular face using rest `mesh_pos` and current `positions`.
    Returns a tensor of shape [num_faces, 3, 3].
    """
    mesh_pos: torch.Tensor = sample[NODE].mesh_pos
    faces: torch.Tensor = sample.face_index

    rest_vertices = mesh_pos[faces]          # [nb faces, 3, 3]
    deformed_vertices = positions[faces]     # [nb faces, 3, 3]

    rest_edges = torch.stack((
        rest_vertices[:, 1] - rest_vertices[:, 0],
        rest_vertices[:, 2] - rest_vertices[:, 0]
    ), dim=2)                                # [nb faces, 3, 2]
    deformed_edges = torch.stack((
        deformed_vertices[:, 1] - deformed_vertices[:, 0],
        deformed_vertices[:, 2] - deformed_vertices[:, 0]
    ), dim=2)                                # [nb faces, 3, 2]

    # Pseudo-inverse of rest_edges (closed-form 2x2 inverse to stay on-device)
    gram = torch.bmm(rest_edges.transpose(1, 2), rest_edges)  # [nb faces, 2, 2]
    gram += 1e-6 * torch.eye(2, dtype=gram.dtype, device=gram.device)
    gram_inv = inv_mps(gram)
    rest_pinv = torch.bmm(gram_inv, rest_edges.transpose(1, 2))  # [nb faces, 2, 3]

    F = torch.bmm(deformed_edges, rest_pinv)  # [nb faces, 3, 3]   # Deformation gradient
    C = torch.bmm(F.transpose(1, 2), F) # Cauchy-Green deformation tensor
    
    I = torch.eye(3, dtype=C.dtype, device=C.device)
    return 0.5 * (C - I)
