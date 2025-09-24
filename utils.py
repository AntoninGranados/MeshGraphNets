import torch

from typing import Any

# TODO: handle batch size > 1
BATCH = 0   #! where we use only the first batch and should be handle corretly for multiple batch

def batch_dicts(inputs: list[dict[Any, torch.Tensor]]):
    batch = {k: v.unsqueeze(0) for k, v in inputs[0].items()}
    for i in range(1, len(inputs)):
        batch = {
            k: torch.concatenate([
                v.unsqueeze(0), inputs[i][k]
            ], dim=0) for k, v in batch.items()
        }
    return batch

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

"""
OPERATIONS ON TRIANGLES
"""
def get_triangle_sarea(tris: torch.Tensor) -> torch.Tensor:
    """Compute the signed area of each triangle in a list"""
    select = lambda tensor, dim, idx : torch.index_select(tensor, dim, torch.Tensor([idx]).type(torch.long))
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

# TODO: remove `pos` and directly pass the position in `A`, `B` and `C`
def get_triangle_aspect_ratio(pos, A, B, C) -> torch.Tensor:
    a = torch.norm(pos[A].unsqueeze(0) - pos[B].unsqueeze(0))
    b = torch.norm(pos[B].unsqueeze(0) - pos[C].unsqueeze(0))
    c = torch.norm(pos[C].unsqueeze(0) - pos[A].unsqueeze(0))
    s = (a+b+c)/2

    return a*b*c / (8 * (s-a)*(s-b)*(s-c))

def find_enclosing_triangle(P: torch.Tensor, pos: torch.Tensor, tris: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each point in `P` (N,2), compute :
        - `enclosing_triangle` (N,1): the index of the triangle enclosing it in `tris` (F,3) using the vertex coordinates `pos` (M,2)
        - `barycentric_coords` (N,3): the barycentrix coordinates of the point inside of its cooresponding triangle
    """
    epsilon = 1e-5
    A = pos[tris[:, 0]]
    B = pos[tris[:, 1]]
    C = pos[tris[:, 2]]

    # bboxes
    mins = torch.min(torch.stack([A, B, C], dim=1), dim=1).values
    maxs = torch.max(torch.stack([A, B, C], dim=1), dim=1).values

    N = P.shape[0]
    face_idx = torch.full((N,), -1, dtype=torch.long)
    barycentric = torch.zeros((N, 3))

    for pi in range(N):
        p = P[pi]
        # find candidate faces whose bbox contains P
        mask = (mins[:,0] - epsilon <= p[0]) & (p[0] <= maxs[:,0] + epsilon) & \
               (mins[:,1] - epsilon <= p[1]) & (p[1] <= maxs[:,1] + epsilon)
        candidate_tris = torch.nonzero(mask).flatten()

        a, b, c = A[candidate_tris], B[candidate_tris], C[candidate_tris]
        lambdas = get_barycentric_coord(p.tile((candidate_tris.shape[0], 1)), a, b, c)
        correct_tris = (get_triangle_sarea(torch.stack([a, b, c], dim=1)) >= epsilon) & torch.all(lambdas >= 0, dim=1)
        
        found_tri = torch.count_nonzero(correct_tris) > 0
        face_idx[pi] = -1 if not found_tri else candidate_tris[correct_tris][0]
        barycentric[pi] = torch.Tensor([float("nan")]*3) if not found_tri else lambdas[correct_tris][0]

    return face_idx, barycentric
