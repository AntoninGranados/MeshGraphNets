from compatibility import override
from enum import IntEnum

import torch
from torch_geometric.data import HeteroData

from loss.loss import Loss
from utils import *

class SelfSupervisedLoss(Loss):
    class LossTerms(IntEnum):
        Total = 0
        Inertia = 1
        Gravity = 2
        Bending = 3
        Stretch = 4
        Count = 5

    def __init__(self):
        self.g = 9.81
        self._loss_terms = torch.empty((SelfSupervisedLoss.LossTerms.Count,))

    @override
    def get_loss_terms(self) -> torch.Tensor:
        return self._loss_terms

    def __inertia(self, pred_pos, pos, vel, m, dt, mask) -> torch.Tensor:
        x_hat = pos + vel
        x_diff = pred_pos - x_hat
        L_inertia = m * torch.sum(torch.square(x_diff), dim=-1) / (2 * torch.square(dt))
        L_inertia = torch.sum(L_inertia[mask])
        return L_inertia
    
    def __gravity(self, pred_pos, m, mask) -> torch.Tensor:
        L_gravity = m * self.g * pred_pos[:, 2]
        L_gravity = torch.sum(L_gravity[mask])
        return L_gravity

    def __bending(self, theta_pred, theta_0, bending_coeff) -> torch.Tensor:
        theta_diff = theta_pred - theta_0

        L_bending_edge = 0.5 * bending_coeff * torch.square(theta_diff)
        L_bending = torch.sum(L_bending_edge)
        return L_bending

    def __stretching(self, G, lame_mu, lame_lambda, area, thickness) -> torch.Tensor:

        sq_trace_G = torch.square(G.diagonal(dim1=-1, dim2=-2).sum(-1))
        trace_sq_G = (G * G).sum(dim=(1, 2))

        energy_density = lame_mu * trace_sq_G + 0.5 * lame_lambda * sq_trace_G

        L_stretch = torch.sum(area * thickness * energy_density)
        return L_stretch

    @override
    def compute(self, sample: HeteroData, prediction: HeteroData) -> torch.Tensor:
        m = sample[NODE].v_mass.squeeze(-1)
        
        pos = sample[NODE].world_pos
        prev_pos = sample[NODE].prev_world_pos
        vel = pos - prev_pos
        normal_mask = (sample[NODE].type == NodeType.NORMAL).squeeze(-1)

        pred_acc = prediction[NODE].features
        pred_pos = torch.where(
            normal_mask.unsqueeze(-1),
            pos + vel + pred_acc,
            sample[NODE].next_world_pos
        )

        L_inertia = self.__inertia(
            pred_pos,
            pos,
            vel,
            m,
            sample.time_step,
            normal_mask
        )
        self._loss_terms[SelfSupervisedLoss.LossTerms.Inertia] = torch.detach(L_inertia)

        L_gravity = self.__gravity(
            pred_pos,
            m,
            normal_mask
        )
        self._loss_terms[SelfSupervisedLoss.LossTerms.Gravity] = torch.detach(L_gravity)

        L_bending = self.__bending(
            compute_dihedral_angle(sample, pred_pos),
            sample[MESH].theta_0,
            sample.bending_coeff
        )
        self._loss_terms[SelfSupervisedLoss.LossTerms.Bending] = torch.detach(L_bending)

        L_stretch = self.__stretching(
            compute_green_strain(sample, pred_pos),
            sample.lame_mu,
            sample.lame_lambda,
            sample[MESH].face_area.squeeze(-1),
            sample.thickness
        )
        self._loss_terms[SelfSupervisedLoss.LossTerms.Stretch] = torch.detach(L_stretch)

        L_static = L_gravity + L_bending + L_stretch
        L_total = L_inertia + L_static
        self._loss_terms[SelfSupervisedLoss.LossTerms.Total] = torch.detach(L_total)
        return L_total
