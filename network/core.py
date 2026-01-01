import torch
from torch import nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from utils import *
from compatibility import inspect_signature, get_flat_param_names, collect_param_data, get_message_passing_attr

class MLP(nn.Module):
    def __init__(self, widths, layer_norm: bool = True):
        super().__init__()

        layers = []

        n_in = widths[0]
        for w in widths[1:-1]:
            layers.append(nn.Linear(n_in, w))
            layers.append(nn.ReLU())
            n_in = w

        layers.append(nn.Linear(n_in, widths[-1]))

        if layer_norm:
            layers.append(nn.LayerNorm(widths[-1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def make_mlp(input_size: int, output_size: int, latent_size: int = 128, num_hidden_layers: int = 1, layer_norm: bool = True) -> nn.Module:
    widths = [input_size] + [latent_size] * num_hidden_layers + [output_size]
    return MLP(widths, layer_norm)

class Encoder(nn.Module):
    def __init__(self,
                 node_input_size: int,
                 mesh_edge_input_size: int,
                 latent_size: int = 128
    ) -> None:
        super().__init__()
        self.node_encoder = make_mlp(node_input_size, latent_size, latent_size)

        self.edge_encoders = nn.ModuleDict()
        self.edge_encoders['mesh'] = make_mlp(mesh_edge_input_size, latent_size, latent_size)

    def __call__(self, sample: HeteroData) -> HeteroData:
        sample[NODE].features = self.node_encoder(sample[NODE].features)

        sample[MESH].features = self.edge_encoders['mesh'](sample[MESH].features)

        return sample

class Decoder(nn.Module):
    def __init__(self,
                 output_size: int,
                 latent_size: int = 128
    ) -> None:
        super().__init__()
        self.node_decoder = make_mlp(latent_size, output_size, latent_size, layer_norm=False)

    def __call__(self, sample: HeteroData) -> HeteroData:
        sample[NODE].features = self.node_decoder(sample[NODE].features)
        return sample

class GraphNetBlock(MessagePassing):
    def __init__(self, latent_size: int = 128):
        super().__init__(aggr='add')
        self.node_mlp = make_mlp(2 * latent_size, latent_size)  # [features, aggregated] -> latent
        self.edge_mlp = make_mlp(3 * latent_size, latent_size)  # [features_i, features_j, edge_features] -> latent

        inspect_signature(self.inspector, self.message_mesh)

        self._user_args = get_flat_param_names(
            self.inspector,
            ['message_mesh', 'aggregate', 'update'], exclude=list(self.special_args))

    def forward(self, sample):
        return self.propagate(sample)
    
    def __call__(self, sample: HeteroData) -> HeteroData:
        return self.forward(sample)

    def message_mesh(self, node_features_i, node_features_j, edge_features):
        features = torch.cat([node_features_i, node_features_j, edge_features], dim=-1)
        return self.edge_mlp(features)

    def update_mesh_edge_features(self, sample):
        edge_index = sample[MESH].edge_index
        node_features = sample[NODE].features
        # size = self._check_input(edge_index, None)
        size = get_message_passing_attr(self, 'check_input')(edge_index, None)

        coll_dict = get_message_passing_attr(self, 'collect')(
            set(self._user_args), edge_index, size,
            dict(node_features=node_features)
        )
        coll_dict['edge_features'] = sample[MESH].features

        msg_kwargs = collect_param_data(self.inspector, 'message_mesh', coll_dict)
        return self.message_mesh(**msg_kwargs)

    def aggregate_nodes(self, edge_features, edge_index, user_args, size, **kwargs):
        coll_dict = get_message_passing_attr(self, 'collect')(user_args, edge_index, size, kwargs)
        aggr_kwargs = collect_param_data(self.inspector, 'aggregate', coll_dict)
        return self.aggregate(edge_features, **aggr_kwargs)

    def update(self, aggregated_features_mesh, features):
        in_features = torch.cat([aggregated_features_mesh, features], dim=-1)
        return self.node_mlp(in_features)

    def propagate(self, sample):
        N = sample['node'].features.shape[0]
        mesh_edge_features_updated = self.update_mesh_edge_features(sample)

        aggr_args = get_flat_param_names(
            self.inspector,
            ['aggregate'],
            exclude=list(self.special_args),
        )
        mesh_edge_index = sample[MESH].edge_index
        mesh_size = (N, N)

        cloth_features_from_mesh = self.aggregate_nodes(
            mesh_edge_features_updated, mesh_edge_index, aggr_args, mesh_size
        )

        cloth_features_new = self.update(cloth_features_from_mesh, sample[NODE].features)
        sample[NODE].features += cloth_features_new

        sample[MESH].features += mesh_edge_features_updated

        return sample
