import torch
from enum import IntEnum

class NodeType(IntEnum):
    NORMAL = 0
    HANDLES = 1

# Define HeteroData component names
NODE = "node"
MESH = (NODE, "mesh", NODE)

def get_device() -> torch.device:
    if torch.cuda.is_available():   # Nvidia GPU
        return torch.device("cuda")
    elif torch.mps.is_available():  # MacOS (Metal Performance Shaders)
        return torch.device("mps")
    return torch.device("cpu")
