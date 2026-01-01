from compatibility import override

import torch
from torch_geometric.data import HeteroData

from network.model import Model
from loss.loss import Loss
from utils import *

class RolloutLoss(Loss):
    def __init__(self, step_loss_fn: Loss, rollout_steps: int):
        assert rollout_steps > 0, "The number of rollout steps need to be strictly positive"
        self._step_loss_fn = step_loss_fn
        self._rollout_steps = rollout_steps
        self._loss_terms = torch.empty_like(step_loss_fn.get_loss_terms())

    def get_loss_terms(self) -> torch.Tensor:
        return self._loss_terms

    @override
    def __call__(self, model: Model, sample: HeteroData) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=sample[NODE].world_pos.device)
        current_sample = sample.clone()

        self._loss_terms = torch.zeros_like(self._loss_terms)

        for step in range(self._rollout_steps):
            prediction = model.forward_pass(current_sample, accumulate_stats=step==0)
            step_loss = self._step_loss_fn.compute(current_sample, prediction)
            self._loss_terms += self._step_loss_fn.get_loss_terms()
            total_loss += step_loss

            # Update current_sample for next step
            current_sample = model.integrate_pos(current_sample, prediction)

        total_loss /= self._rollout_steps
        self._loss_terms /= self._rollout_steps
        return total_loss
