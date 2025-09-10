import torch

from enum import IntEnum
from collections import namedtuple

Mesh = dict[str, torch.Tensor]

EdgeSet = namedtuple("EdgeSet", ["name", "edge_features", "senders", "receivers"])
Graph = namedtuple("Graph", ["node_features", "edge_sets"])

# TODO: fix the dataset to only use the node types we need
class NodeType(IntEnum):
    NORMAL = 0
    # OBSTACLE = 1
    # AIRFOIL = 2
    HANDLE = 3
    # INFLOW = 4
    # OUtorchLOW = 5
    # WALL_BOUNDARY = 6
    COUNT = 9

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
