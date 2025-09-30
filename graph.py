from torch_geometric.nn import radius
import torch

from utils import find_enclosing_triangle, BATCH

from enum import IntEnum
from collections import namedtuple
from typing import Callable
import sys

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

def compute_mesh_edges(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor]:
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

    comparaison = inverse.unsqueeze(0) == torch.arange(len(unique_packed_edges)).unsqueeze(1).to(inverse.device)
    counts = torch.count_nonzero(comparaison, dim=1)

    edge_indices, opposites1 = torch.nonzero(comparaison[counts == 1], as_tuple=True)
    opposites[edge_indices, 0] = unpacked_opposite[opposites1]
    
    edge_indices, opposites2 = torch.nonzero(comparaison[counts == 2], as_tuple=True)
    edge_indices = edge_indices.reshape((-1,2))[:,0]
    opposites2 = opposites2.reshape((-1,2))
    opposites[edge_indices] = opposites2

    opposites = torch.concat([opposites, opposites])

    return edges.unsqueeze(0), opposites.unsqueeze(0)

# TODO: handle batch size > 1
def compute_world_edges(mesh: Mesh, meta: dict, edges: torch.Tensor|None = None) -> torch.Tensor:
    if edges is None:
        edges, _ = compute_mesh_edges(mesh)
    edges = edges[BATCH]

    world_pos = mesh["world_pos"][BATCH]
    if sys.platform == "darwin":
        world_pos = world_pos.to("cpu")

    neighbors = radius(world_pos, world_pos, r=meta["collision_radius"]).t()

    if sys.platform == "darwin":
        neighbors = neighbors.to(mesh["world_pos"].device)

    # only allow world edges where there are no mesh edges
    edge_mask = ~(edges.unsqueeze(1) == neighbors.unsqueeze(0)).all(dim=-1).any(dim=0)

    # only allow world edges where at least one node is `NORMAL`
    type_mask = (mesh["node_type"][BATCH][neighbors[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][BATCH][neighbors[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()

    return neighbors[edge_mask & type_mask].unsqueeze(0)

"""
MESH OPERATIONS
"""
# TODO: handle batch size > 1
def generate_graph(mesh: Mesh, meta: dict) -> MultiGraph:
    # compute node features
    velocities = mesh["world_pos"] - mesh["prev|world_pos"]
    types = torch.nn.functional.one_hot(mesh["node_type"].squeeze(-1).long(), NodeType.COUNT)
    node_features = torch.concat([
        velocities,
        types
    ], dim=-1)
    
    # compute mesh edge set
    mesh_edges, _ = compute_mesh_edges(mesh)
    senders, receivers = mesh_edges[...,0][BATCH], mesh_edges[...,1][BATCH]
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

    world_edges = compute_world_edges(mesh, meta=meta, edges=mesh_edges)
    senders, receivers = world_edges[...,0][BATCH], world_edges[...,1][BATCH]

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

    return MultiGraph(
        node_features=node_features,
        edge_sets=[
            mesh_edge_set,
            world_edge_set
        ]
    )

# TODO: handle batch size > 1
def interpolate_field(dst_mesh: Mesh, src_mesh: Mesh, field: torch.Tensor, interpolation_filter: Callable[[torch.Tensor], torch.Tensor]|None=None) -> torch.Tensor:
    """ Interpolates the field `field` from source mesh `src_mesh` to destination mesh `dst_mesh` using barycentric interpolation """

    dst_node_mask = dst_mesh["node_type"][BATCH].flatten() == NodeType.NORMAL
    dst_node_mask |= dst_mesh["node_type"][BATCH].flatten() == NodeType.HANDLE
    src_node_mask = src_mesh["node_type"][BATCH].flatten() == NodeType.NORMAL
    src_node_mask |= src_mesh["node_type"][BATCH].flatten() == NodeType.HANDLE


    # TODO: compute only the distance between normal/handle nodes then between collider nodes (would need to keep track of the indices)
    nodes_dist = torch.cdist(dst_mesh["mesh_pos"][BATCH], src_mesh["mesh_pos"][BATCH])

    # find the closest nodes for each collider nodes to set the field value for them
    #! this is just to not keep the with a zero field but they still should not be uses as in mesh spece they could be overlaping
    # non_normal_node_present = bool(((torch.count_nonzero(~dst_node_mask) > 0) & (torch.count_nonzero(~src_node_mask) > 0)).item())
    # if non_normal_node_present:
    #     closest_collider = torch.argmin(nodes_dist[:,~src_node_mask][~dst_node_mask], dim=-1)

    nodes_dist[:,~src_node_mask] = float("inf")
    common_nodes_mask = (nodes_dist < 1e-3).any(dim=-1) & dst_node_mask
    removed_nodes_mask = ~common_nodes_mask & dst_node_mask

    # compute node pairs (dst_mesh_i <-> src_mesh_j | nan if not NORMAL or not in the target mesh)
    node_pairs = torch.argmin(nodes_dist, dim=-1)
    node_pairs = torch.where(
        common_nodes_mask,
        node_pairs,
        -1
    )

    # compute the enclosing triangle for each point
    sorted_cells = src_mesh["cells"][BATCH]
    cell_mask = src_node_mask[sorted_cells[:,0]] | src_node_mask[sorted_cells[:,1]] | src_node_mask[sorted_cells[:,2]]
    sorted_cells = sorted_cells[cell_mask]
    tris, lambdas = find_enclosing_triangle(
        dst_mesh["mesh_pos"][BATCH][removed_nodes_mask],
        src_mesh["mesh_pos"][BATCH],
        sorted_cells
    )

    enclosing_tris = sorted_cells[tris]
    lambdas_shape = (-1, 3, *[1]*(len(field[BATCH].shape)-1))
    lambdas = lambdas.reshape(lambdas_shape)
    if torch.isnan(lambdas).any():
        print("lambdas is NaN"); exit(1)

    # fill the interpolated field
    new_field = torch.zeros((dst_node_mask.shape[0], *field[BATCH].shape[1:]))

    new_field[common_nodes_mask] = field[BATCH][node_pairs[common_nodes_mask]]
    interpolated_field = torch.sum(field[BATCH][enclosing_tris] * lambdas, dim=1)
    if interpolation_filter is not None:
        interpolated_field = torch.stack([
            interpolation_filter(e) for e in interpolated_field.unbind(dim=0)
        ], dim=0)
    new_field[removed_nodes_mask] = interpolated_field

    # if non_normal_node_present:
    #     new_field[~dst_node_mask] = field[BATCH][closest_collider]

    return new_field.unsqueeze(0)
