import torch

from graph import Mesh, cells_to_edges
from remesher.miniellipse import miniellipse
from utils import aggregate, BATCH


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
    edges, _ = cells_to_edges(mesh)

    senders, receivers = edges[:,0], edges[:,1]
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

