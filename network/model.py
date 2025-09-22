import torch

from typing import Any

from graph import Mesh, NodeType, EdgeSet, MultiGraph, generate_graph
from network.normalizer import Normalizer
from network.core import Encoder, GraphNetBlock, Decoder

class Model(torch.nn.Module):
    def __init__(self,
                 device: torch.device,
                 node_input_size: int = 3 + NodeType.COUNT,     # n_i, (x_i^t- x_i^{t-1})
                 mesh_input_size: int = 7,                      # u_ij, |u_ij|, x_ij, |x_ij|
                 world_input_size: int = 0,                     #! 4 : x_ij, |x_ij|
                 output_size: int = 3,                          # a_i
                 graph_net_blocks_count: int = 15
    ) -> None:
        super().__init__()

        self.node_normalizer = Normalizer(device, size=node_input_size)
        self.edge_normalizer = torch.nn.ModuleDict({
            "mesh": Normalizer(device, size=mesh_input_size),
            # "world": Normalizer(device, size=world_input_size)
        })
        self.output_normalizer = Normalizer(device, size=3)

        self.encoder = Encoder(node_input_size, mesh_input_size, world_input_size)
        self.graph_net_blocks = torch.nn.ModuleList([GraphNetBlock() for _ in range(graph_net_blocks_count)])
        self.decoder = Decoder(output_size=output_size)

        self.loss_fn = torch.nn.MSELoss()

    def __normalize_graph(self, graph: MultiGraph, is_training: bool) -> MultiGraph:
        graph = graph._replace(
            node_features = self.node_normalizer(graph.node_features, is_training)
        )

        normalized_edge_sets = [
            edge_set._replace(
                edge_features = self.edge_normalizer[edge_set.name](edge_set.edge_features, is_training)
            ) for edge_set in graph.edge_sets
        ]

        graph = graph._replace(
            edge_sets = normalized_edge_sets
        )

        return graph
    
    def __forward_pass(self, input: Mesh, meta: dict, is_training: bool) -> torch.Tensor:
        graph = generate_graph(input, meta)
        graph = self.__normalize_graph(graph, is_training)
        latent_graph = self.encoder(graph)
        for graph_net_block in self.graph_net_blocks:
            latent_graph = graph_net_block(latent_graph)
        return self.decoder(latent_graph)

    def __integrate_pos(self, mesh: Mesh, prediction: torch.Tensor, meta: dict[str, Any]) -> Mesh:
        """Second order integration"""
        acceleration = self.output_normalizer.inverse(prediction)

        curr_pos = mesh["world_pos"]
        prev_pos = mesh["prev|world_pos"]
        pred_pos = 2*curr_pos + acceleration - prev_pos

        loss_mask = mesh["node_type"] == NodeType.NORMAL
        loss_mask = loss_mask.squeeze(-1)

        pred_mesh = {k: v.clone() for k,v in mesh.items()}
        pred_mesh["prev|world_pos"][loss_mask] = mesh["world_pos"][loss_mask]
        pred_mesh["world_pos"][loss_mask] = pred_pos[loss_mask]
        # pred_mesh["target|world_pos"] = torch.Tensor(0)  # remove the target from the prediction (would not be on the correct device)
        return pred_mesh

    def loss(self, mesh: Mesh, meta: dict[str, Any], is_training: bool = True) -> torch.Tensor:
        prediction = self.__forward_pass(mesh, meta, is_training)

        curr_pos = mesh["world_pos"]
        prev_pos = mesh["prev|world_pos"]
        targ_pos = mesh["target|world_pos"]
        targ_acc = targ_pos - 2*curr_pos + prev_pos
        target = self.output_normalizer(targ_acc, is_training)

        loss_mask = mesh["node_type"] == NodeType.NORMAL
        loss_mask = loss_mask.squeeze(-1)

        return self.loss_fn(prediction[loss_mask], target[loss_mask])

    def __call__(self, mesh: Mesh, meta: dict[str, Any]) -> Mesh:
        prediction = self.__forward_pass(mesh, meta, is_training=False)
        return self.__integrate_pos(mesh, prediction, meta)
