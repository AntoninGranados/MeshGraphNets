import torch

from copy import deepcopy
from typing import Any

from graph import Mesh, NodeType, EdgeSet, Graph, cells_to_edges
from normalizer import Normalizer
from core import Encoder, GraphNetBlock, Decoder

class Model(torch.nn.Module):
    def __init__(self, device: torch.device, graph_net_blocks_count: int = 15):
        super().__init__()

        self.node_normalizer = Normalizer(device, size=3+NodeType.COUNT)    # n_i, (x_i^t- x_i^{t-1})
        self.edge_normalizer = torch.nn.ModuleDict({
            "mesh": Normalizer(device, size=7), # u_ij, |u_ij|, x_ij, |x_ij|
            # "world": Normalizer(device, size=0) # TODO
        })
        self.output_normalizer = Normalizer(device, size=3)

        self.encoder = Encoder()
        self.graph_net_blocks = torch.nn.ModuleList([GraphNetBlock() for _ in range(graph_net_blocks_count)])
        self.decoder = Decoder(output_size=3)   # nodes acceleration

        self.loss_fn = torch.nn.MSELoss()

    def __generate_graph(self, mesh: Mesh, is_training: bool) -> Graph:
        # compute node features
        velocities = torch.subtract(mesh["world_pos"], mesh["prev|world_pos"])
        types = torch.nn.functional.one_hot(mesh["node_type"].squeeze(-1).type(torch.long), NodeType.COUNT)
        node_features = torch.concat([
            velocities,
            types
        ], dim=-1)
        node_features = self.node_normalizer(node_features, is_training)
        
        # compute mesh edge sets
        edges = cells_to_edges(mesh)
        senders, receivers = edges[0,:], edges[1,:]

        rel_world_pos = (torch.index_select(mesh["world_pos"], 1, senders) -
                          torch.index_select(mesh["world_pos"], 1, receivers))
        rel_mesh_pos = (torch.index_select(mesh["mesh_pos"], 1, senders) -
                            torch.index_select(mesh["mesh_pos"], 1, receivers))
        mesh_edge_features = torch.concat([
            rel_world_pos,
            torch.norm(rel_world_pos, dim=-1, keepdim=True),
            rel_mesh_pos,
            torch.norm(rel_mesh_pos, dim=-1, keepdim=True),
        ], dim=-1)
        mesh_edge_features = self.edge_normalizer["mesh"](mesh_edge_features, is_training)

        mesh_edge_set = EdgeSet(
            name="mesh",
            edge_features=mesh_edge_features,
            senders=senders,
            receivers=receivers
        )

        return Graph(
            node_features=node_features,
            edge_sets=[
                mesh_edge_set
            ]
        )
    
    def __forward_pass(self, input: Mesh, is_training: bool) -> torch.Tensor:
        graph = self.__generate_graph(input, is_training)
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
        prediction = self.__forward_pass(mesh, is_training)

        curr_pos = mesh["world_pos"]
        prev_pos = mesh["prev|world_pos"]
        targ_pos = mesh["target|world_pos"]
        targ_acc = targ_pos - 2*curr_pos + prev_pos
        target = self.output_normalizer(targ_acc, is_training)

        loss_mask = mesh["node_type"] == NodeType.NORMAL
        loss_mask = loss_mask.squeeze(-1)

        return self.loss_fn(prediction[loss_mask], target[loss_mask])

    def __call__(self, mesh: Mesh, meta: dict[str, Any]) -> Mesh:
        prediction = self.__forward_pass(mesh, is_training=False)
        return self.__integrate_pos(mesh, prediction, meta)
