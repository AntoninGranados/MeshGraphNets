import torch
from torch_geometric.data import HeteroData

from network.model import Model
from utils import *

class Loss():
    def __init__(self):
        pass

    def compute(self, sample: HeteroData, prediction: HeteroData) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self, model: Model, sample: HeteroData) -> torch.Tensor:
        prediction = model.forward_pass(sample)
        return self.compute(sample, prediction)
