from torch.utils.data import DataLoader
import torch

import numpy as np

import json
from pathlib import Path
from tqdm import tqdm

import sys
from typing import Any

from dataset import Dataset, move_batch_to
from display.trajectory import display_trajectory, display_prediction_target, display_trajectory_list
from network.model import Model

def get_device() -> torch.device:
    from sys import platform
    import os

    # MacOs
    if platform == "darwin":
        return torch.device("mps")
    # Telecom's GPUs
    else:
        device_index = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device_index is None:
            raise KeyError("You should set the [CUDA_VISIBLE_DEVICES] env var to select a device")
        # return torch.device(type="cuda", index=int(device_index))
        return torch.device(type="cuda", index=0)   # only one visible device


def lr_lambda(step, hyper):
    decay = hyper["training"]["decay-rate"] ** (step / hyper["training"]["decay-steps"])
    min_lr = hyper["training"]["min-lr"]/hyper["training"]["start-lr"]
    return max(decay, min_lr)


def init_model(device: torch.device, hyper: dict[str, Any]) -> tuple[Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, int, list[float], list[float]]:
    print("Initializing model...")

    model = Model(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["training"]["start-lr"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda step: lr_lambda(step, hyper)
    )

    return model, optimizer, scheduler, -1, [], []

def init_from_checkpoint(device: torch.device, hyper: dict[str, Any], checkpoint: int|None = None) -> tuple[Model, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, int, list[float], list[float]]:
    if checkpoint is None:
        checkpoint_path = sorted(list(Path(hyper["training"]["checkpoint-dir"]).glob("*.pt")))[-1]
    else:
        checkpoint_path = Path(hyper["training"]["checkpoint-dir"], f"mgn_{checkpoint:0>4}.pt")
    print(f"Loading model from {checkpoint_path}")
    last_checkpoint = torch.load(checkpoint_path, map_location=device)

    model = Model(device)
    model.load_state_dict(last_checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["training"]["start-lr"])
    optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda step: lr_lambda(step, hyper)
    )
    scheduler.load_state_dict(last_checkpoint['scheduler_state_dict'])

    train_loss_file = Path(hyper["training"]["checkpoint-dir"], "loss.txt")
    train_loss = list(np.loadtxt(train_loss_file))
    # train_loss = []

    valid_loss_file = Path(hyper["training"]["checkpoint-dir"], "validation_loss.txt")
    valid_loss = list(np.loadtxt(valid_loss_file))
    # valid_loss = []

    return model, optimizer, scheduler, last_checkpoint['epoch'], train_loss, valid_loss


def train(device: torch.device, hyper: dict[str, Any]) -> None:
    print("Loading training datasets   (length: 1000) ...")
    train_ds = Dataset(Path("dataset", "sphere_dynamic"), stage="train", estimated_size=1000)
    print("Loading validation datasets (length:  100) ...")
    valid_ds = Dataset(Path("dataset", "sphere_dynamic"), stage="valid", estimated_size=100)
    
    train_loader = DataLoader(train_ds, batch_size = 1, shuffle = True, num_workers = 8, prefetch_factor = 10)
    valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle = True, num_workers = 8, prefetch_factor = 10)
    
    # model, optimizer, scheduler, start_epoch, train_loss, valid_loss = init_model(device, hyper)
    model, optimizer, scheduler, start_epoch, train_loss, valid_loss = init_from_checkpoint(device, hyper)
    model.train()

    # Warmup
    if start_epoch == -1:
        iter_train_loader = iter(train_loader)
        for _ in tqdm(range(1000), desc="Warmup", file=sys.stdout):
            batch = move_batch_to(next(iter_train_loader), device)
            with torch.no_grad():
                _ = model.loss(batch, train_ds.meta)

    epoch_len = min(int(len(train_loader)), hyper["training"]["max-epoch-len"])     # limit epoch length (could be around 50_000 with large dataset)
    epochs = int(hyper["training"]["steps"])//epoch_len
    for e in range(start_epoch+1, epochs):

        # update the model
        model.train()
        loss_sum, it = 0, 0
        loop = tqdm(range(epoch_len), desc=f"Train {e:>4}/{epochs}", ncols=100, file=sys.stdout)
        iter_train_loader = iter(train_loader)
        for _ in loop:
            optimizer.zero_grad()

            batch = move_batch_to(next(iter_train_loader), device)
            loss = model.loss(batch, train_ds.meta)
            loss.backward()
            optimizer.step()

            scheduler.step()

            loss_sum += loss.item()
            it += 1
            loop.set_postfix({"loss": f"{loss_sum/(it+1):.6f}"})
    
        train_loss.append(loss_sum/it)

        # compute validation loss
        model.eval()
        loss_sum, it = 0, 0
        loop = tqdm(range(epoch_len), desc=f"Valid {e:>4}/{epochs}", ncols=100, file=sys.stdout)
        iter_valid_loader = iter(valid_loader)
        for _ in loop:
            batch = move_batch_to(next(iter_valid_loader), device)
            with torch.no_grad():
                loss = model.loss(batch, valid_ds.meta, is_training=False)
            loss_sum += loss.item()
            it += 1
            loop.set_postfix({"loss": f"{loss_sum/it:.6f}"})

        valid_loss.append(loss_sum/it)

        # save the model
        torch.save({
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, Path(hyper["training"]["checkpoint-dir"], f"mgn_{e:0>4}.pt"))
        np.savetxt(Path(hyper["training"]["checkpoint-dir"], "loss.txt"), train_loss)
        np.savetxt(Path(hyper["training"]["checkpoint-dir"], "validation_loss.txt"), valid_loss)
        
        print()

def rollout(device: torch.device, hyper: dict[str, Any], checkpoints: list[int|None] = [None], test_set: str = "valid", test_idx: int = 0) -> None:
    print(f"Loading {test_set} datasets ...")
    ds = Dataset(Path("dataset", "sphere_dynamic"), stage=test_set)
    
    models = []
    for checkpoint in checkpoints:
        model, _, _, _, _, _ = init_from_checkpoint(device, hyper, checkpoint)
        model.eval()
        models.append(model)

    # rollout
    # mesh = ds[test_idx*(ds.meta["trajectory_length"]-2)]
    # pred_meshs = [
    #     {k: v.unsqueeze(0) for k,v in mesh.items()} for _ in range(len(models))
    # ]
    # targ_mesh = {k: v.unsqueeze(0) for k,v in mesh.items()}

    mesh1 = ds[test_idx*(ds.meta["trajectory_length"]-2)+5]
    pred_meshs = []
    pred_meshs.append({k: v.unsqueeze(0) for k,v in mesh1.items()})

    
    max_frame = 300
    prev_meshs = [
        {k: v.clone() for k,v in pred_mesh.items()}
        for pred_mesh in pred_meshs
    ]
    loss = [0.0]
    for i in tqdm(range(1, max_frame), file=sys.stdout):
        for m in range(len(models)):
            current_mesh = move_batch_to(prev_meshs[m], device)
            with torch.no_grad():
                pred_mesh_i = models[m](current_mesh, ds.meta)

            mesh_i_cpu = {k: v.detach().cpu() for k,v in pred_mesh_i.items()}
            pred_meshs[m] = {k: torch.concat([v, mesh_i_cpu[k]], dim=0) for k,v in pred_meshs[m].items()}

            prev_meshs[m] = mesh_i_cpu

        # targ_mesh_i = {k: v.unsqueeze(0) for k,v in ds[test_idx*399+i].items()}
        # targ_mesh = {k: torch.concat([v, targ_mesh_i[k]], dim=0) for k,v in targ_mesh.items()}
        # loss.append(torch.sqrt(torch.nn.MSELoss()(targ_mesh_i["world_pos"], prev_meshs[0]["world_pos"])).item())

    display_trajectory_list(pred_meshs, ["Prediction"], ds.meta, title="Sphere Dynamic", save=True, save_path=Path("img"))
    # display_trajectory_list([*pred_meshs, targ_mesh], [*[f"Pred {ck}" for ck in checkpoints], "Target"], ds.meta, "Flag Minimal", save=False, save_path=Path("img"))
    # display_prediction_target(pred_meshs[0], targ_mesh, ds.meta, title="Flag Minimal", save=True, save_path=Path("img"))

if __name__ == "__main__":
    print(f"PyTorch version : {torch.__version__}")

    device = get_device()
    print(f"Device: {device.type}:{device.index}")

    with open(Path(".", "hyperparam.json"), "r") as file:
        hyper = json.loads(file.read())

    # train(device, hyper)
    rollout(device, hyper, [None], test_set="valid", test_idx=0)
