from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import argparse

import torch

from network.model import Model
from dataset import SimulationLoader
from utils import *
from loss.self_supervised_loss import SelfSupervisedLoss
from rollout import rollout

parser = argparse.ArgumentParser(
    prog='MeshGraphNet',
    description='This script train or run a MeshGraphNet model'
)
parser.add_argument('-roll', '--rollout', type=Path, help='the path for the checkpoint (\"last\"/<path>) to use when doing a rollout')
parser.add_argument('-rd', '--roll-data', type=int, help='the dataset index for the rollout (only used with --run)', default=0)
parser.add_argument('-rl', '--roll-length', type=int, help='the number of epochs of the rollout (only used with --run)', default=100)
parser.add_argument('-ckp', '--checkpoints', type=Path, help='the path to the checkpoints directory', required=True)
parser.add_argument('-ds', '--dataset', type=Path, help='the path to the dataset directory', required=True)
parser.add_argument('--save-format', type=Path, help='the format of the checkpoint files (ex: mgn_[e].pt)', default="mgn_[e].pt")
parser.add_argument('-hp', '--hyperparam', type=Path, help='the path to the hyperparameter file (JSON)', default=Path('hyperparam.json'))
args = parser.parse_args()

device = get_device()
hyper = json.load(open(args.hyperparam, 'r'))

model = Model(
    node_input_size=5,
    mesh_input_size=8,
    output_size=3,
    graph_net_blocks_count=hyper['network']['graph-net-blocks'],
)
model.to(device)

if args.rollout is not None:
    rollout(args, model, device)

else:
    model.train()

    loader = SimulationLoader(args.dataset, shuffle=True)
    # loader = SimulationLoader(args.dataset, shuffle=True, noise_scale=0.0)
    # loader = list(iter(loader))[:10]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda s: lr_lambda(s, hyper)
    )

    loss_fn = SelfSupervisedLoss()

    starting_epoch = 0
    checkpoints = sorted(list(args.checkpoints.glob('*.pt')))
    if len(checkpoints) > 0:
        print(f"[WARN] checkpoints already exist, do you want to continue ? If yes, the last checkpoint will be loaded [y/N] ", end='')
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
                _ = model(batch)

    #! Make sure it does not already exist !!
    # output = open(Path(args.checkpoints, "loss_log.txt"), 'w')
    # output.write("total_loss, L_inertia, L_gravity, L_bending, L_stretch\n")

    epochs = int(hyper['training']['steps'] / len(loader))
    for e in range(starting_epoch+1, epochs):
        loop = tqdm(loader, desc=f'Epoch {e:>3}/{epochs-1}', file=sys.stdout)
        loss_sum = 0

        for it, batch in enumerate(loop):
            optimizer.zero_grad()
            
            batch.to(device)
            loss = loss_fn(batch, model.forward_pass(batch))
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            loop.set_postfix({'Loss': f'{loss_sum/(it+1): .3f}'})
        
        # output.write(f'{loss_sum / (it+1)}, {", ".join(str(term / (it+1)) for term in loss_terms_sum)}\n')
        # output.flush()

        if e % 10 == 0:
            save_epoch(Path(args.checkpoints, args.save_format), e, model, optimizer, scheduler)

    # Always save the last epoch
    save_epoch(Path(args.checkpoints, args.save_format), epochs, model, optimizer, scheduler)
