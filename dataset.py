from tfrecord.torch.dataset import TFRecordDataset
import torch

import numpy as np

import json
from pathlib import Path
from typing import Any, override
from copy import deepcopy

from graph import NodeType, Mesh, interpolate_field
from utils import BATCH

def move_batch_to(batch: Mesh, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}

class Dataset(torch.utils.data.Dataset[Mesh]):
    def __init__(self, data_path: Path|str, stage: str, noise_scale: float = 3e-3, noise_gamma: float = 1e-1):
        super().__init__()

        self.stage = stage

        self.noise_scale = noise_scale
        self.noise_gamma = noise_gamma

        with open(Path(data_path, "meta.json"), "r") as file:
            self.meta = json.loads(file.read())
            
        self.content = list(iter(TFRecordDataset(
            data_path = str(Path(data_path, f"{stage}.tfrecord")),
            index_path = None,
            transform = self.__parse
        )))

    @override
    def __getitem__(self, idx: int) -> Mesh:
        def from_idx(idx: int) -> tuple[int, int]:
            sim = idx // (self.meta["trajectory_length"]-2)
            time = idx % (self.meta["trajectory_length"]-2)+1
            return sim, time
        
        sim, time = from_idx(idx)
        mesh = {k: v[time] for k,v in self.content[sim].items()}

        #! should only be done during training (but the "sphere_dynamic" training dataset is very heavy so I use the validation set for now)
        sim, time = from_idx(idx-1)
        prev = {k: v[time] for k,v in self.content[sim].items()}
        sim, time = from_idx(idx+1)
        targ = {k: v[time] for k,v in self.content[sim].items()}
        mesh = self.__add_targets(mesh, prev, targ)
        if self.stage == "train":
            mesh = self.__add_noise(mesh)
            
        return mesh
    
    # @override # (No default __len__ method)
    def __len__(self) -> int:
        return len(self.content) * (self.meta["trajectory_length"]-2)

    def __parse(self, proto: dict) -> Mesh:
        out = {}
        for key, info in self.meta["features"].items():
            data = torch.tensor(np.frombuffer(proto[key], np.dtype(info["dtype"])))
            data = data.reshape(info["shape"])

            if info["type"] == "static":
                data = data.tile([self.meta["trajectory_length"], 1, 1])
            elif info["type"] == "dynamic_varlen":
                length = torch.tensor(np.frombuffer(proto["length_"+key], np.int32))
                length = length.flatten()
                data = list(torch.split(data, length.tolist()))
            elif info["type"] != "dynamic":
                raise ValueError("Invalid data format")

            out[key] = data

        return out
    
    def __add_targets(self, curr: Mesh, prev: Mesh, targ: Mesh) -> Mesh:
        out = deepcopy(curr)

        prev_mesh = {k: v.unsqueeze(0) for k,v in prev.items()}
        curr_mesh = {k: v.unsqueeze(0) for k,v in curr.items()}
        targ_mesh = {k: v.unsqueeze(0) for k,v in targ.items()}

        out["prev|world_pos"] = interpolate_field(curr_mesh, prev_mesh, prev_mesh["world_pos"])[0]
        out["target|world_pos"] = interpolate_field(curr_mesh, targ_mesh, targ_mesh["world_pos"])[0]
            
        return out

    def __add_noise(self, data: Mesh) -> Mesh:
        noise = torch.normal(torch.zeros_like(data["world_pos"]), self.noise_scale).type(torch.float32)

        mask = (data["node_type"] == NodeType.NORMAL)
        mask = mask.tile([3])
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        
        out = deepcopy(data)
        out["world_pos"] += noise
        out["target|world_pos"] += (1.0 - self.noise_gamma) * noise
        return out
