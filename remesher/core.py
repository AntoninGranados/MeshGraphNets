import torch

from graph import Mesh, NodeType, cells_to_edges
from remesher.miniellipse import miniellipse
from utils import aggregate, batch_dicts, BATCH

from collections import namedtuple


EPS = 1e-4
AR_THRESHOLD = 100 # aspect ratio threshold

"""
MESH STATE OPERATIONS
"""
# TODO: store an edge map {(min(a,b), max(a,b)): edge_id} to speed up the find_edge calls
# This class is used to store the state of the mesh during remeshing operations
MeshState = namedtuple(
    "MeshState",
    ["edges", "opposites", "modified_edge_mask", "node_mask", "mesh_pos", "world_pos", "S_i"]
)

def to_meshstate(mesh: Mesh, sizing_field: torch.Tensor) -> MeshState:
    edges, opposites = cells_to_edges(mesh)
    edges, opposites = edges[BATCH], opposites[BATCH]
    edges = edges[:edges.shape[0]//2]  # only use one way edges (same computation for the others)
    opposites = opposites[:opposites.shape[0]//2]  # only use one way edges (same computation for the others)

    return MeshState(
        edges = edges,
        opposites = opposites,
        modified_edge_mask = torch.zeros(len(edges)).bool(),
        node_mask = mesh["node_type"][BATCH].flatten() == NodeType.NORMAL,
        mesh_pos = mesh["mesh_pos"][BATCH],
        world_pos = mesh["world_pos"][BATCH],
        S_i = sizing_field[BATCH]
    )

# TODO
def from_meshstate(ms: MeshState) -> Mesh:
    mesh = Mesh()
    mesh["cells"] = torch.Tensor()
    mesh["node_type"] = torch.Tensor()
    mesh["mesh_pos"] = torch.Tensor()
    mesh["world_pos"] = torch.Tensor()

    return batch_dicts([mesh])

"""
SIZING FIELD OPERATIONS
"""
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
    edges = edges[BATCH]

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
    ], dim=0).unsqueeze(0)

