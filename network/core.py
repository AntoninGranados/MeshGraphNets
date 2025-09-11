import torch

from graph import NodeType, EdgeSet, MultiGraph

def make_linear(input_size: int, output_size: int) -> torch.nn.Linear:
    linear = torch.nn.Linear(input_size, output_size)
    
    # Use the Xavier/Glorot distribution as initial distribution (like Tensorflow)
    torch.nn.init.xavier_uniform_(linear.weight)
    
    return linear

def make_mlp(input_size: int, output_size: int, layer_norm: bool = True) -> torch.nn.Module:
    layers = [
        make_linear(input_size, 128),
        torch.nn.ReLU(),
        make_linear(128, 128),
        torch.nn.ReLU(),
        make_linear(128, output_size),
    ]

    if layer_norm:
        layers.append(torch.nn.LayerNorm(output_size))

    return torch.nn.Sequential(*layers)


class Encoder(torch.nn.Module):
    def __init__(self, latent_size: int = 128):
        super().__init__()
        self.node_encoder = make_mlp(input_size=NodeType.COUNT + 3, output_size=latent_size) # n_i, (x_i^t- x_i^{t-1})
        self.edge_encoders = torch.nn.ModuleDict({
            "mesh" : make_mlp(input_size=2 + 1 + 3 + 1, output_size=latent_size), # u_ij, |u_ij|, x_ij, |x_ij|
            # "world": make_mlp(input_size=3 + 1, output_size=latent_size)          # x_ij, |x_ij|
        })

    def __call__(self, graph: MultiGraph) -> MultiGraph:
        latent_nodes = self.node_encoder(graph.node_features)

        edge_sets = []
        for edge_set in graph.edge_sets:
            latent = self.edge_encoders[edge_set.name](edge_set.edge_features)
            edge_sets.append(edge_set._replace(
                edge_features=latent
            ))

        return MultiGraph(latent_nodes, edge_sets)

class Decoder(torch.nn.Module):
    def __init__(self, output_size: int, latent_size: int = 128):
        super().__init__()
        self.node_decoder = make_mlp(input_size=latent_size, output_size=output_size, layer_norm=False)

    def __call__(self, latent_graph: MultiGraph) -> torch.Tensor:
        decoded_features = self.node_decoder(latent_graph.node_features)
        return decoded_features

class GraphNetBlock(torch.nn.Module):
    def __init__(self, latent_size: int = 128):
        super().__init__()

        self.node_mlp = make_mlp(input_size=3*latent_size, output_size=latent_size) # v_i, sum e_ij^M, sum e_ij^W
        self.edge_mlps = torch.nn.ModuleDict({
            "mesh" : make_mlp(input_size=3*latent_size, output_size=latent_size), # e_ij^M, v_i, v_j
            # "world": make_mlp(input_size=3*latent_size, output_size=latent_size)  # e_ij^W, v_i, v_j
        })

    def __update_edge_features(self, node_features: torch.Tensor, edge_set: EdgeSet) -> torch.Tensor:
        v_i = torch.index_select(node_features, 1, edge_set.senders)
        v_j = torch.index_select(node_features, 1, edge_set.receivers)
        features = torch.concat([ v_i, v_j, edge_set.edge_features], dim=-1)
        return self.edge_mlps[edge_set.name](features)
    
    def __update_node_features(self, node_features: torch.Tensor, edge_sets: list[EdgeSet]) -> torch.Tensor:
        features = [node_features]

        for edge_set in edge_sets:
            sum_eij = torch.zeros_like(node_features).index_add_(1, edge_set.receivers, edge_set.edge_features)
            features.append(sum_eij)
        
        # if there are no world edge
        if len(edge_sets) == 1:
            features.append(torch.zeros_like(node_features))

        return self.node_mlp(torch.concat(features, dim=-1))

    def __call__(self, graph: MultiGraph) -> MultiGraph:
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            new_features = self.__update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(edge_features=new_features))

        new_node_features = self.__update_node_features(graph.node_features, new_edge_sets)

        # residual connections
        new_node_features += graph.node_features
        new_edge_sets = [
            new._replace(edge_features=new.edge_features + old.edge_features) for new, old in zip(new_edge_sets, graph.edge_sets)
        ]

        return MultiGraph(new_node_features, new_edge_sets)
