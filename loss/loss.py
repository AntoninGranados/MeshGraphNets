import torch
from torch_geometric.data import HeteroData

from utils import *

class Loss():
    def __init__(self):
        pass

    def __call__(self, sample: HeteroData, prediction: HeteroData) -> torch.Tensor:
        raise NotImplementedError
