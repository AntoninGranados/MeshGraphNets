from compatibility import override
from typing import Callable, Any

import torch
from torch_geometric.data import HeteroData
import random

from network.model import Model
from loss.loss import Loss
from utils import *

def HOOD_rollout_steps_func(storage: dict[str, Any]) -> int:
    n = storage['steps'] // storage['step increase']
    n += 1
    n = min(n, storage['max rollout'])

    return n

def MGN_RP_rollout_steps_func(storage: dict[str, Any]) -> int:
    p = 1 - storage['steps'] / storage['total steps']
    n = 1

    while n < storage['max rollout'] and p < random.random():
        n += 1

    return n

class ScheduledAutoRegressiveLoss(Loss):
    def __init__(self, loss_fn: Loss, rollout_steps_func: Callable[[dict[str, Any]], int], func_storage: dict[str, Any] = {}):
        self._loss_fn = loss_fn
        self._rollout_steps_func = rollout_steps_func
        self._loss_terms = torch.zeros_like(loss_fn.get_loss_terms())
        self._func_storage = func_storage
        self._func_storage['steps'] = 0

    def get_loss_terms(self) -> torch.Tensor:
        return self._loss_terms

    @override
    def __call__(self, model: Model, sample: HeteroData) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=sample[NODE].world_pos.device)
        current_sample = sample.clone()

        self._loss_terms = torch.zeros_like(self._loss_terms)

        rollout_steps = self._rollout_steps_func(self._func_storage)
        for step in range(rollout_steps):
            prediction = model.forward_pass(current_sample, accumulate_stats=step==0)
            loss = self._loss_fn.compute(current_sample, prediction)
            self._loss_terms += self._loss_fn.get_loss_terms()
            total_loss += loss

            # Update current_sample for next step
            current_sample = model.integrate_pos(current_sample, prediction)

        total_loss /= rollout_steps
        self._loss_terms /= rollout_steps

        self._func_storage['steps'] += 1
        return total_loss

class AutoRegressiveLoss(ScheduledAutoRegressiveLoss):
    def __init__(self, loss_fn: Loss, rollout_steps: int):
        assert rollout_steps > 0, "The number of rollout steps need to be strictly positive"
        super().__init__(loss_fn, lambda _ : rollout_steps)
