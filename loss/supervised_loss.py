from compatibility import override

import torch
from torch_geometric.data import HeteroData

from loss.loss import Loss
from utils import *

class SupervisedLoss(Loss):
    def __init__(self):
        self._loss_fn = torch.nn.MSELoss()
        self._loss_terms = torch.empty((1,))

    def get_loss_terms(self) -> torch.Tensor:
        return self._loss_terms
    
    @override
    def compute(self, sample: HeteroData, prediction: HeteroData) -> torch.Tensor:
        pred_acc = prediction[NODE].features

        pos = sample[NODE].world_pos
        prev_pos = sample[NODE].prev_world_pos
        targ_pos = sample[NODE].next_world_pos
        target_acc = targ_pos - 2*pos + prev_pos

        normal_mask = sample[NODE].type == NodeType.NORMAL
        normal_mask = normal_mask.squeeze(-1)

        loss = self._loss_fn(pred_acc[normal_mask], target_acc[normal_mask])
        self._loss_terms = torch.detach(loss)
        return loss
