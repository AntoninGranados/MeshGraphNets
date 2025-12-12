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
        graph_net_blocks_count: int = 15,
    ) -> None:
        super().__init__()

        self.node_normalizer = Normalizer(size=node_input_size)
        self.edge_normalizer = torch.nn.ModuleDict()
        self.edge_normalizer['mesh'] = Normalizer(size=mesh_input_size)
        self.output_normalizer = Normalizer(size=3)

        self.encoder = Encoder(node_input_size, mesh_input_size)
        self.graph_net_blocks = torch.nn.ModuleList([GraphNetBlock() for _ in range(graph_net_blocks_count)])
        self.decoder = Decoder(output_size=output_size)

    def normalize_graph(self, sample: HeteroData, accumulate_stats: bool = True) -> HeteroData:
        sample[NODE].features = self.node_normalizer(sample[NODE].features, self.training and accumulate_stats)
        sample[MESH].features = self.edge_normalizer['mesh'](sample[MESH].features, self.training and accumulate_stats)

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

    def forward_pass(self, sample: HeteroData, accumulate_stats: bool = True) -> HeteroData:
        sample = self.compute_features(sample)
        normalized_sample = self.normalize_graph(sample, self.training and accumulate_stats)
        latent_sample = self.encoder(normalized_sample)

        for _, graph_net_block in enumerate(self.graph_net_blocks):
            latent_sample = graph_net_block(latent_sample)

        output_sample = self.decoder(latent_sample)
        output_sample[NODE].features = self.output_normalizer.inverse(output_sample[NODE].features)
        if self.training and accumulate_stats:   # Accumulate stats only on real data
            self.output_normalizer.accumulate(sample[NODE].next_world_pos - 2*sample[NODE].world_pos + sample[NODE].prev_world_pos)

        return output_sample

    def integrate_pos(self, sample: HeteroData, prediction: HeteroData) -> HeteroData:
        """Second order integration"""
        
        pred_acc = prediction[NODE].features

        pos = sample[NODE].world_pos
        prev_pos = sample[NODE].prev_world_pos
        vel = pos - prev_pos
        pred_pos = pos + vel + pred_acc

        mask = sample[NODE].type == NodeType.NORMAL
        mask = mask.squeeze(-1)

        sample[NODE].prev_world_pos[mask] = sample[NODE].world_pos[mask]
        sample[NODE].world_pos[mask] = pred_pos[mask]

        return sample

    def __call__(self, sample: HeteroData, accumulate_stats: bool = True) -> HeteroData:
        prediction = self.forward_pass(sample, accumulate_stats)
        return self.integrate_pos(sample, prediction)
