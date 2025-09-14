from torch_geometric.nn import radius
import torch

from enum import IntEnum
from collections import namedtuple

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
    COUNT = 9

# TODO: change edges shape from (2, ...) to (..., 2)
def cells_to_edges(mesh: Mesh) -> torch.Tensor:
    """ Compute the graph edges from the cells (triangles) """

    cells = mesh["cells"][0]    #! assumes same cells across the whole batch
    dirty_edges = torch.concat([
        torch.stack([cells[:,0], cells[:,1]], dim=1),
        torch.stack([cells[:,1], cells[:,2]], dim=1),
        torch.stack([cells[:,2], cells[:,0]], dim=1)
    ], dim=0)

    # sort nodes
    nodes_a, nodes_b = torch.max(dirty_edges, dim=1).values, torch.min(dirty_edges, dim=1).values
    # pack nodes
    max_node = torch.max(cells) + 1
    packed = torch.add(nodes_a, nodes_b * max_node)
    # remove duplicated nodes
    nodes = torch.unique(packed)
    nodes_a, nodes_b = nodes % max_node, nodes // max_node
    
    return torch.concat([
        torch.stack([nodes_a, nodes_b]),
        torch.stack([nodes_b, nodes_a])
        ], dim=-1
    )

# TODO: handle batch size > 1
def compute_world_edges(mesh: Mesh, meta: dict, edges: torch.Tensor|None = None) -> torch.Tensor:
    if edges is None:
        edges = cells_to_edges(mesh).rot90()

    #! torch.cdist might work here ?
    neighbours = radius(mesh["world_pos"][0], mesh["world_pos"][0], r=meta["collision_radius"]).rot90()

    edge_mask = ~(edges[:,None]==neighbours).all(dim=-1).any(dim=0)
    type_mask = (mesh["node_type"][0][neighbours[:,0]]==NodeType.NORMAL).any(dim=-1).flatten()
    type_mask |= (mesh["node_type"][0][neighbours[:,1]]==NodeType.NORMAL).any(dim=-1).flatten()

    return neighbours[edge_mask & type_mask]

def generate_graph(mesh: Mesh) -> MultiGraph:
    # compute node features
    velocities = torch.subtract(mesh["world_pos"], mesh["prev|world_pos"])
    types = torch.nn.functional.one_hot(mesh["node_type"].squeeze(-1).type(torch.long), NodeType.COUNT)
    node_features = torch.concat([
        velocities,
        types
    ], dim=-1)
    
    # compute mesh edge sets
    edges = cells_to_edges(mesh)
    senders, receivers = edges[0,:], edges[1,:]

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

    return MultiGraph(
        node_features=node_features,
        edge_sets=[
            mesh_edge_set
        ]
    )