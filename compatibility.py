try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch

try:
    from torch.optim.lr_scheduler import LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def inspect_signature(inspector, fn) -> None:
    if hasattr(inspector, "inspect_signature"):
        inspector.inspect_signature(fn)
    elif hasattr(inspector, "inspect"):
        inspector.inspect(fn)
    else:
        inspector._inspect_signature(fn)


def get_flat_param_names(inspector, fn_names, exclude=None):
    """
    Compute the names manually for older pytorch versions
    """
    if hasattr(inspector, 'get_flat_param_names'):
        return inspector.get_flat_param_names(fn_names, exclude=exclude)
    
    else:
        param = set()
        for fn_name in fn_names:
            for p in inspector.params[fn_name].keys():
                if exclude is not None and p not in exclude:
                    param.add(p)

        return list(param)


def collect_param_data(inspector, fn_name, kwargs):
    if hasattr(inspector, 'collect_param_data'):
        return inspector.collect_param_data(fn_name, kwargs)
    if hasattr(inspector, 'params') and fn_name in inspector.params:
        out = {}
        for key in inspector.params[fn_name].keys():
            if key in kwargs:
                out[key] = kwargs[key]
        return out

def stable_argsort(tensor, dim: int=-1):
    """
    Ignore stability for older pytorch versions
    """
    try:
        return torch.argsort(tensor, dim=dim, stable=True)
    except TypeError:
        return torch.argsort(tensor, dim=dim)


def load_weight_only(f):
    """
    The default value was changed in newer pytorch version, and they added the param 'weights_only'
    """
    try:
        return torch.load(f, weights_only=False)
    except TypeError:
        return torch.load(f)
    

def get_message_passing_attr(block, attr: str):
    if hasattr(block, f'_{attr}'):
        return getattr(block, f'_{attr}')
    else:
        return getattr(block, f'__{attr}__')
