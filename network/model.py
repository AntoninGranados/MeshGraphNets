import torch
from torch import nn
from torch_geometric.data import HeteroData

from network.normalizer import Normalizer
from network.core import Encoder, GraphNetBlock, Decoder
from utils import *

class Model(nn.Module):
    def __init__(self,
        node_input_size: int,
        mesh_input_size: int,
        output_size: int,
        graph_net_blocks_count: int = 15
    ) -> None:
        super().__init__()

        self.node_normalizer = Normalizer(size=node_input_size)
        self.edge_normalizer = torch.nn.ModuleDict()
        self.edge_normalizer['mesh'] = Normalizer(size=mesh_input_size)
        self.output_normalizer = Normalizer(size=3)

        self.encoder = Encoder(node_input_size, mesh_input_size)
        self.graph_net_blocks = torch.nn.ModuleList([GraphNetBlock() for _ in range(graph_net_blocks_count)])
        self.decoder = Decoder(output_size=output_size)

        self.loss_fn = torch.nn.MSELoss()

    def normalize_graph(self, sample: HeteroData, is_training: bool) -> HeteroData:
        sample[NODE].features = self.node_normalizer(sample[NODE].features, is_training)
        sample[MESH].features = self.edge_normalizer['mesh'](sample[MESH].features, is_training)

        return sample
    
    def compute_features(self, sample: HeteroData) -> HeteroData:
        # Node features
        velocities = sample[NODE].world_pos - sample[NODE].prev_world_pos
        types_onehot = torch.nn.functional.one_hot(sample[NODE].type, int(sample[NODE].type.max().item())+1).float()
        sample[NODE].features = torch.cat([velocities, types_onehot], dim=-1)

        # Mesh edges features
        mesh_edges = sample[MESH].edge_index

        rel_world_pos = sample[NODE].world_pos[mesh_edges[0]] - sample[NODE].world_pos[mesh_edges[1]]
        rel_mesh_pos  = sample[NODE].mesh_pos[mesh_edges[0]] - sample[NODE].mesh_pos[mesh_edges[1]]
        mesh_edge_attr = torch.cat([
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True),
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
        ], dim=-1)
        sample[MESH].features = mesh_edge_attr

        return sample

    def forward_pass(self, sample: HeteroData, is_training: bool) -> HeteroData:
        sample = self.compute_features(sample)
        normalized_sample = self.normalize_graph(sample, is_training)
        latent_sample = self.encoder(normalized_sample)

        for _, graph_net_block in enumerate(self.graph_net_blocks):
            latent_sample = graph_net_block(latent_sample)

        output_sample = self.decoder(latent_sample)
        return output_sample

    def integrate_pos(self, sample: HeteroData, prediction: HeteroData) -> HeteroData:
        '''Second order integration'''
        
        acceleration = self.output_normalizer.inverse(prediction[NODE].features)

        curr_pos = sample[NODE].world_pos
        prev_pos = sample[NODE].prev_world_pos
        pred_pos = 2*curr_pos + acceleration - prev_pos

        loss_mask = sample[NODE].type == NodeType.NORMAL
        loss_mask = loss_mask.squeeze(-1)

        sample[NODE].prev_world_pos[loss_mask] = sample[NODE].world_pos[loss_mask]
        sample[NODE].world_pos[loss_mask] = pred_pos[loss_mask]

        return sample

    def loss(self, sample: HeteroData, is_training: bool = True) -> torch.Tensor:
        prediction = self.forward_pass(sample, is_training)
        predicted_acc = prediction[NODE].features

        curr_pos = sample[NODE].world_pos
        prev_pos = sample[NODE].prev_world_pos
        targ_pos = sample[NODE].next_world_pos
        target_acc = targ_pos - 2*curr_pos + prev_pos
        target_acc_norm = self.output_normalizer(target_acc, is_training)

        loss_mask = sample[NODE].type == NodeType.NORMAL
        loss_mask = loss_mask.squeeze(-1)

        loss = self.loss_fn(predicted_acc[loss_mask], target_acc_norm[loss_mask])

        return loss

    def __call__(self, sample: HeteroData) -> HeteroData:
        prediction = self.forward_pass(sample, is_training=False)
        return self.integrate_pos(sample, prediction)
