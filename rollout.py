import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse

import torch

from network.model import Model
from dataset import SimulationLoader
from utils import *

def rollout(args: argparse.Namespace, model: Model, device: torch.device) -> None:
    model.eval()
    if str(args.rollout) == 'last':
        ckp = sorted(list(args.checkpoints.glob('*.pt')))[-1]
    else:
        ckp = args.rollout

    state = torch.load(ckp)
    model.load_state_dict(state['model_state_dict'])
    print(f'[INFO] Rollout on `{ckp}` for {args.roll_length} time steps')

    fig = plt.figure()

    loader = SimulationLoader(args.dataset, noise_scale=0.0, shuffle=False)
    data = loader.dataset[args.roll_data]

    for i in tqdm(range(args.roll_length), file=sys.stdout):
        fig.clf()
        ax = fig.add_subplot(projection='3d')

        pred = model(data.to(device))
        data = pred.detach().cpu()
        
        ax.plot_trisurf(data[NODE].world_pos[:,0], data[NODE].world_pos[:,1], data[NODE].world_pos[:,2], triangles=data.face_index)
        # center = np.mean(data[NODE].world_pos.detach().numpy(), axis=0)
        # ax.set_xlim([center[0]-10, center[0]+10])
        # ax.set_ylim([center[1]-10, center[1]+10])
        # ax.set_zlim([center[2]-10, center[2]+10])
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([-0.5, 2.5])
        ax.set_zlim([-2, 2])
        ax.set_aspect('equal')
        ax.set_axis_off()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)