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

    state = torch.load(ckp, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print(f'[INFO] Rollout on `{ckp}` for {args.roll_length} time steps')

    loader = SimulationLoader(args.dataset, noise_scale=0.0, shuffle=False)
    data = loader.dataset[args.roll_data]

    plt.style.use('dark_background')
    fig = plt.figure()
    fig.subplots_adjust(left=0.05, bottom=0, right=0.95, top=0.80)
    plt.rcParams["font.family"] = "monospace"

    for i in tqdm(range(args.roll_length), file=sys.stdout):
        fig.clf()
        ax = fig.add_subplot(projection='3d')

        with torch.no_grad():
            pred = model(data.to(device))
        data = pred.detach().cpu()
        
        ax.plot_trisurf(
            data[NODE].world_pos[:,0], data[NODE].world_pos[:,1], data[NODE].world_pos[:,2],
            triangles=data.face_index,
            color=(1.,1.,1.,1.), # edgecolor=(0.,0.,0.,0.3), linewidth=0.5
        )
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([-0.5, 2.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_aspect('equal')
        ax.set_axis_off()

        ax.set_title(f"Frame {i+1}/{args.roll_length}", color="gray", fontsize=12)
        ax.patch.set_edgecolor("w")
        ax.patch.set_facecolor("k")
        ax.patch.set_alpha(0.2)
        ax.patch.set_linewidth(1)

        fig.suptitle(f"Rollout after self-supervised training", fontsize=16, y=0.95)
        fig.patch.set_facecolor('k')
        fig.patch.set_alpha(0.2)

        plt.tight_layout()
        # plt.draw()
        # plt.pause(0.01)

        # plt.savefig(f"img/frame_{i:03}.png", dpi=400)

        
