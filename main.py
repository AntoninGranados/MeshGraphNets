from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import argparse

import torch

from network.model import Model
from dataset import heterodata_from_npz, SimulationLoader
from utils import *

parser = argparse.ArgumentParser(
    prog='MeshGraphNet',
    description='This script train or run a MeshGraphNet model'
)
parser.add_argument('-r', '--rollout', type=Path, help='the path for the checkpoint (\"last\"/path) to use when doing a rollout')
parser.add_argument('-rs', '--roll-data', type=int, help='the dataset index for the rollout (only used with --run)', default=0)
parser.add_argument('-rl', '--roll-length', type=int, help='the number of epochs of the rollout (only used with --run)', default=100)
parser.add_argument('--hyper', type=Path, help='the path to the hyperparameter file (JSON)', default=Path('hyperparam.json'))
parser.add_argument('--checkpoints', type=Path, help='the path to the checkpoints directory', default=Path('checkpoints', 'flag'))
parser.add_argument('--save-format', type=Path, help='the format of the checkpoint files (ex: mgn_[e].pt)', default="mgn_[e].pt")
parser.add_argument('--dataset', type=Path, help='the path to the dataset directory', default=Path('datasets', 'flag'))
args = parser.parse_args()

device = get_device()
hyper = json.load(open(args.hyper, 'r'))

model = Model(
    node_input_size=5,
    mesh_input_size=8,
    output_size=3,
    graph_net_blocks_count=hyper['network']['graph-net-blocks']
)
model.to(device)

# Rollout using the `args.rollout` checkpoint)
if args.rollout is not None:
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
    simulation = np.load(Path('datasets', 'flag', 'raw', 'flag-1.npz'))
    faces = simulation['faces'] # FIXME: should use the faces from the data (not implemented for now, see `dataset.py`)
    # data = heterodata_from_npz(simulation, 0)
    data = list(iter(loader))[args.roll_data]

    for i in tqdm(range(args.roll_length), file=sys.stdout):
        fig.clf()
        ax = fig.add_subplot(projection='3d')
        
        pred = model(data.to(device))   # type: ignore (device should be int|str ?)
        data = pred.detach().cpu()
        ax.plot_trisurf(data[NODE].world_pos[:,0], data[NODE].world_pos[:,1], data[NODE].world_pos[:,2], triangles=faces)
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([-0.5, 2.5])
        ax.set_zlim([-2, 2])
        ax.set_axis_off()
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

# Training
else:
    model.train()

    loader = SimulationLoader(args.dataset, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda s: lr_lambda(s, hyper)
    )

    starting_epoch = 0
    checkpoints = sorted(list(args.checkpoints.glob('*.pt')))
    if len(checkpoints) > 0:
        print(f"[WARN] checkpoints already exist, do you want to continue ? If yes, the latest checkpoint will be loaded [y/N] ", end='')
        res = input()
        if res.lower() != 'y':
            raise FileExistsError(f'There are already checkpoints saved in `{args.checkpoints}`')
        
        ckp = torch.load(checkpoints[-1], map_location=device)

        starting_epoch = int(ckp.get('epoch', 0))

        model.load_state_dict(ckp['model_state_dict']) 
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        # FIXME: might be better (and lighter when saving) to simply create the scheduler using the `last_epoch` field
        scheduler.load_state_dict(ckp['scheduler_state_dict'])
        print(f"[INFO] Loaded checkpoint '{checkpoints[-1].name}'")

    if starting_epoch == 0:
        for batch in tqdm(loader, desc='Warmup', file=sys.stdout):
            batch = batch.to(device)
            with torch.no_grad():
                _ = model.loss(batch)

    epochs = int(hyper['training']['steps'] / len(loader))
    for e in range(starting_epoch+1, epochs):
        loop = tqdm(loader, desc=f'Epoch {e:>3}/{epochs-1}', file=sys.stdout)
        loss_sum = 0
        for it, batch in enumerate(loop):
            optimizer.zero_grad()
            
            batch.to(device)
            loss = model.loss(batch)
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            loop.set_postfix({'Loss': f'{loss_sum/(it+1): .3f}'})

        if e % 10 == 0:
            save_epoch(Path(args.checkpoints, args.save_format), e, model, optimizer, scheduler)

    # Always save the last epoch
    save_epoch(Path(args.checkpoints, args.save_format), epochs, model, optimizer, scheduler)
