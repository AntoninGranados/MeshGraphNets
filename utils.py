from enum import IntEnum

class NodeType(IntEnum):
    NORMAL = 0
    HANDLES = 1

# Define HeteroData component names
NODE = "node"
MESH = (NODE, "mesh", NODE)
