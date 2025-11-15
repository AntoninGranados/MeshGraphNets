from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys

import torch
from torch_geometric.loader import DataLoader

from network.model import Model
from dataset import SimulationDataset
from utils import *

device = torch.device("mps")

checkpoints_dir = Path("checkpoints", "flag")
dataset_dir = Path("datasets", "flag")

dataset = SimulationDataset(root=dataset_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

sim_1 = np.load(Path("datasets", "flag", "raw", "flag-1.npz"))
faces_1 = sim_1["faces"]

model = Model(
    node_input_size=5,
    mesh_input_size=8,
    output_size=3,
)
model.to(device)

"""
model.eval()
ckp = sorted(list(checkpoints_dir.glob("*.pt")))[-1]
print(f"Loading `{ckp}`")
state = torch.load(ckp)
model.load_state_dict(state["model_state_dict"])

fig = plt.figure()

data = dataset.get(400)
for i in tqdm(range(200), file=sys.stdout):
    fig.clf()
    ax = fig.add_subplot(projection="3d")
    
    pred = model(data.to(device), {})
    data = pred.detach().cpu()
    ax.plot_trisurf(data[NODE].world_pos[:,0], data[NODE].world_pos[:,1], data[NODE].world_pos[:,2], triangles=faces_1)
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([-0.5, 2.5])
    ax.set_zlim([-2, 2])
    ax.set_axis_off()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

exit(0)
"""


def lr_lambda(step):
    decay = 0.1 ** (step / 5_000_000)
    min_lr = 1e-6/1e-4
    return max(decay, min_lr)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda = lambda s: lr_lambda(s)
)

from_checkpoint = False
checkpoints = sorted(list(checkpoints_dir.glob("*.pt")))
starting_epoch = 0
if len(checkpoints) > 0:
    print(f"[WARN] checkpoints already exist, do you want to continue ? If yes, the latest checkpoint will be loaded [y/N] ", end="")
    res = input()
    if res.lower() != "y":
        raise FileExistsError(f"There are already checkpoints saved in `{checkpoints_dir}`")
    
    from_checkpoint = True

    ckp = torch.load(checkpoints[-1], map_location=device)

    starting_epoch = int(ckp.get("epoch", 0))

    model.load_state_dict(ckp["model_state_dict"]) 
    optimizer.load_state_dict(ckp["optimizer_state_dict"])
    scheduler.load_state_dict(ckp["scheduler_state_dict"]) 
    print(f"[INFO] Loaded checkpoint '{checkpoints[-1].name}' (epoch={starting_epoch})")

model.train()

if not from_checkpoint:
    for batch in tqdm(loader, desc="Warmup", file=sys.stdout):
        batch = batch.to(device)
        with torch.no_grad():
            _ = model.loss(batch, {})

epochs = 401
for e in range(starting_epoch+1, epochs):
    loop = tqdm(loader, desc=f"Epoch {e:>3}/{epochs-1}", file=sys.stdout)
    loss_sum = 0
    for it, batch in enumerate(loop):
        optimizer.zero_grad()
        
        batch.to(device)
        loss = model.loss(batch, {})
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()
        loop.set_postfix({"Loss": f"{loss_sum/(it+1): .3f}"})

    if e % 10 == 0:
        torch.save({
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, Path(checkpoints_dir, f"mgn_{e:03}.pt"))
