import torch
from pathlib import Path
from enum import IntEnum

class NodeType(IntEnum):
    NORMAL = 0
    HANDLES = 1

# Define HeteroData component names
NODE = 'node'
MESH = (NODE, 'mesh', NODE)

def get_device() -> torch.device:
    if torch.cuda.is_available():   # Nvidia GPU
        return torch.device('cuda')
    elif torch.mps.is_available():  # MacOS (Metal Performance Shaders)
        return torch.device('mps')
    return torch.device('cpu')

def save_epoch(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    parsed_path = str(path).replace('[e]', f'{epoch:03}')
    print(f"Saving checkpoint to `{parsed_path}`")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, parsed_path)

def lr_lambda(step, hyper):
    """
    Exponential decay learning rate
    """
    decay_steps = hyper['training']['steps'] * hyper['training']['decay-steps']
    decay = pow(hyper['training']['decay-rate'], step / decay_steps)
    min_lr = hyper['training']['min-lr'] / hyper['training']['start-lr']
    return max(decay, min_lr)
