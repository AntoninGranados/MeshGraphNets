from pathlib import Path
import numpy as np
from numpy.lib.npyio import NpzFile
from tqdm import tqdm
import sys

import torch
from torch import nn
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader

from utils import *

def faces_to_edges(faces: np.ndarray) -> torch.Tensor:
    faces = np.sort(faces, axis=-1)

    # Make sure the nodes indices are ordered to correctly identify unique edges
    edges = np.concatenate([
        faces[:, [0,1]],
        faces[:, [1,2]],
        faces[:, [0,2]],
    ], axis=0)

    edges = np.unique(edges, axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]])  # Undirected edges
    return torch.from_numpy(edges.T).long()    # [2, num_edges]

# TODO: save the faces in the `HeteroData`
def heterodata_from_npz(simulation: NpzFile, time_ind: int) -> HeteroData:
    mesh_pos  = simulation['verts']
    world_pos = simulation['nodes'][1:-1]
    prev_world = simulation['nodes'][:-2]
    next_world = simulation['nodes'][2:]

    sample = HeteroData()
    
    # ===== Node data
    sample[NODE].world_pos = torch.from_numpy(world_pos[time_ind]).float()
    sample[NODE].prev_world_pos = torch.from_numpy(prev_world[time_ind]).float()
    sample[NODE].next_world_pos = torch.from_numpy(next_world[time_ind]).float()
    sample[NODE].mesh_pos = torch.from_numpy(mesh_pos).float()

    sample[NODE].type = torch.from_numpy(simulation['node_type'][time_ind]).long()

    # ===== Mesh edges data
    mesh_edges = faces_to_edges(simulation['faces'])
    sample[MESH].edge_index = mesh_edges

    sample[NODE].num_nodes = simulation['nodes'].shape[1]

    return sample


class SimulationDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path | str,
        noise_scale: float = 1e-3,
        noise_gamma: float = 1e-1
    ):
        super().__init__(str(root))

        # weights_only=False because I generated the data (I trust it)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        self.noise_scale = noise_scale
        self.noise_gamma = noise_gamma

    @property
    def raw_file_names(self):
        return sorted([f.name for f in Path(self.raw_dir).glob('*.npz')])

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def __add_noise(self, sample: HeteroData) -> HeteroData:
        if self.noise_scale == 0:
            return sample
        
        noise = torch.normal(torch.zeros_like(sample[NODE].world_pos), self.noise_scale).type(sample[NODE].world_pos.dtype)

        mask: torch.Tensor = sample[NODE].type == NodeType.NORMAL
        mask = torch.stack([mask]*3, dim=-1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        
        sample[NODE].world_pos += noise
        sample[NODE].next_world_pos += (1.0 - self.noise_gamma) * noise

        return sample
    
    def get(self, idx: int) -> HeteroData:
        sample = super().get(idx)
        return self.__add_noise(sample) # type: ignore
        
    def process(self):
        data_list = []
        for file_path in self.raw_paths:
            simulation = np.load(file_path)

            n_frames = simulation['nodes'].shape[0]
            for t in tqdm(range(n_frames-2), file=sys.stdout, desc=f'Processing {Path(file_path).name}'):
                data = heterodata_from_npz(simulation, t)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SimulationLoader(DataLoader):
    def __init__(
        self,
        root: Path | str,
        noise_scale: float = 1e-3,
        noise_gamma: float = 0.1,
        shuffle: bool = False
    ):
        dataset = SimulationDataset(root, noise_scale, noise_gamma)
        super().__init__(dataset, batch_size=1, shuffle=shuffle)
