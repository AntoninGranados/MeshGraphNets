from typing import override

import torch
from torch_geometric.data import HeteroData

from network.model import Model
from loss.loss import Loss
from utils import *

class RolloutLoss(Loss):
    def __init__(self, step_loss_fn: Loss, rollout_steps: int):
        assert rollout_steps > 0, "The number of rollout steps need to be strictly positive"
        self.step_loss_fn = step_loss_fn
        self.rollout_steps = rollout_steps

    @override
    def __call__(self, model: Model, sample: HeteroData) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=sample[NODE].world_pos.device)
        current_sample = sample.clone()

        for step in range(self.rollout_steps):
            prediction = model.forward_pass(current_sample, accumulate_stats=step==0)
            step_loss = self.step_loss_fn.compute(current_sample, prediction)
            total_loss += step_loss

            # Update current_sample for next step
            current_sample = model.integrate_pos(current_sample, prediction)

        total_loss /= self.rollout_steps
        return total_loss
