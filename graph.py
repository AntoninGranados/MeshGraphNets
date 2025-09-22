from torch_geometric.nn import radius
import torch

from utils import find_enclosing_triangle, BATCH

from enum import IntEnum
from collections import namedtuple
from typing import Callable

# node_type, world_pos, cells, mesh_pos
Mesh = dict[str, torch.Tensor]

EdgeSet = namedtuple("EdgeSet", ["name", "edge_features", "senders", "receivers"])
MultiGraph = namedtuple("Graph", ["node_features", "edge_sets"])

# TODO: fix the dataset to only use the node types we need
class NodeType(IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    # AIRFOIL = 2
    HANDLE = 3
    # INFLOW = 4
    # OUTFLOW = 5
    # WALL_BOUNDARY = 6
    COUNT = 9   #! should be 7 but should be kept for backward compatibility

"""
EDGE OPERATIONS
"""
def find_edge(a, b, edges) -> int:
    candidate = torch.nonzero(((edges[:,0] == a) & (edges[:,1] == b)) | ((edges[:,0] == b) & (edges[:,1] == a)))
    if len(candidate) == 0:
        return -1
    return int(candidate.item())

def find_edges_with(a, edges) -> torch.Tensor:
    candidate = torch.nonzero((edges[:,0] == a) | (edges[:,1] == a)).squeeze()
    return candidate

def cells_to_edges(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the graph edges from the cells (triangles) and the opposites nodes for each edge
    """

    cells = mesh["cells"][BATCH]
    unpacked_edges = torch.concat([
        torch.stack([cells[:,0], cells[:,1]], dim=1),
        torch.stack([cells[:,1], cells[:,2]], dim=1),
        torch.stack([cells[:,2], cells[:,0]], dim=1)
    ], dim=0).long()
    unpacked_opposite = torch.concat([
        cells[:,2],
        cells[:,0],
        cells[:,1]
    ], dim=0).long()

    # sort nodes
    nodes_a, nodes_b = torch.max(unpacked_edges, dim=1).values, torch.min(unpacked_edges, dim=1).values
    # pack nodes
    max_node = torch.max(cells) + 1
    packed = torch.add(nodes_a, nodes_b * max_node)
    # remove duplicated nodes
    unique_packed_edges, inverse = torch.unique(packed, return_inverse=True)
    nodes_a, nodes_b = unique_packed_edges % max_node, unique_packed_edges // max_node

    edges = torch.concat([
        torch.stack([nodes_a, nodes_b], dim=-1),
        torch.stack([nodes_b, nodes_a], dim=-1)
        ], dim=0
    )

    # compute opposite nodes
    opposites = -torch.ones((unique_packed_edges.shape[0], 2), device=unpacked_edges.device).long()
    counts = torch.bincount(inverse, minlength=unique_packed_edges.shape[0])  # should be 1 (if edge on the border) or 2

    positions = torch.zeros_like(inverse)
    positions[torch.cumsum(counts[inverse], dim=0) - 1] = 1

    opposites[inverse, positions] = unpacked_opposite
    opposites = torch.concat([opposites, opposites])

    return edges, opposites
    

# TODO: handle batch size > 1
def compute_world_edges(mesh: Mesh, meta: dict, edges: torch.Tensor|None = None) -> torch.Tensor:
    if edges is None:
        edges, _ = cells_to_edges(mesh)

    neighbours = radius(mesh["world_pos"][BATCH], mesh["world_pos"][BATCH], r=meta["collision_radius"]).rot90()

    edge_mask = ~(edges[:,None]==neighbours).all(dim=-1).any(dim=0)
    type_mask = (mesh["node_type"][BATCH][neighbours[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][neighbours[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()

    return neighbours[edge_mask & type_mask]

"""
MESH OPERATIONS
"""
def generate_graph(mesh: Mesh, meta: dict) -> MultiGraph:
    # compute node features
    velocities = torch.subtract(mesh["world_pos"], mesh["prev|world_pos"])
    types = torch.nn.functional.one_hot(mesh["node_type"].squeeze(-1).long(), NodeType.COUNT)
    node_features = torch.concat([
        velocities,
        types
    ], dim=-1)
    
    # compute mesh edge set
    mesh_edges, _ = cells_to_edges(mesh)
    senders, receivers = mesh_edges[:,0], mesh_edges[:,0]

    rel_world_pos = (torch.index_select(mesh["world_pos"], 1, senders) -
                        torch.index_select(mesh["world_pos"], 1, receivers))
    rel_mesh_pos = (torch.index_select(mesh["mesh_pos"], 1, senders) -
                        torch.index_select(mesh["mesh_pos"], 1, receivers))
    mesh_edge_features = torch.concat([
        rel_world_pos,
        torch.norm(rel_world_pos, dim=-1, keepdim=True),
        rel_mesh_pos,
        torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
    ], dim=-1)

    mesh_edge_set = EdgeSet(
        name="mesh",
        edge_features=mesh_edge_features,
        senders=senders,
        receivers=receivers
    )
    
    #! compute world edge set
    """
    world_edges = compute_world_edges(mesh, meta=meta, edges=mesh_edges)
    senders, receivers = world_edges[:,0], world_edges[:,0]

    rel_world_pos = (torch.index_select(mesh["world_pos"], 1, senders) -
                        torch.index_select(mesh["world_pos"], 1, receivers))
    world_edge_features = torch.concat([
        rel_world_pos,
        torch.norm(rel_world_pos, dim=-1, keepdim=True),
    ], dim=-1)

    world_edge_set = EdgeSet(
        name="world",
        edge_features=world_edge_features,
        senders=senders,
        receivers=receivers
    )
    """

    return MultiGraph(
        node_features=node_features,
        edge_sets=[
            mesh_edge_set,
            #! world_edge_set
        ]
    )

# TODO: handle batch size > 1
# TODO: find better names for `from_mesh` and `targ_mesh`
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
    node_pairs = torch.where(
        common_nodes_mask,
        node_pairs,
        -1
    )

    # compute the enclosing triangle for each point
    sorted_cells = targ_mesh["cells"][BATCH]
    cell_mask = targ_normal_mask[sorted_cells[:,0]] | targ_normal_mask[sorted_cells[:,1]] | targ_normal_mask[sorted_cells[:,2]]
    sorted_cells = sorted_cells[cell_mask]
    tris, lambdas = find_enclosing_triangle(
        from_mesh["mesh_pos"][BATCH][delete_nodes_mask],
        targ_mesh["mesh_pos"][BATCH],
        sorted_cells
    )

    enclosing_tris = sorted_cells[tris]
    lambdas_shape = (-1, 3, *[1]*(len(field.shape)-1))
    lambdas = lambdas.reshape(lambdas_shape)

    # fill the interpolated field
    new_field = torch.zeros((from_normal_mask.shape[0], *field.shape[1:]))
    new_field[common_nodes_mask] = field[node_pairs[common_nodes_mask]]

    interpolated_field = torch.sum(field[enclosing_tris] * lambdas, dim=1)
    if interpolation_filter is not None:
        interpolated_field = torch.stack([
            interpolation_filter(e) for e in interpolated_field.unbind(dim=0)
        ], dim=0)
    new_field[delete_nodes_mask] = interpolated_field

    return new_field
