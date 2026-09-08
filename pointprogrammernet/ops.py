import random

import numpy as np
import torch


def apply_pointwise(module: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    """Apply a module independently to the last dimension of [B,N,C]."""
    if values.ndim != 3:
        raise ValueError("pointwise input must be [B,N,C]")
    batch, count, channels = values.shape
    return module(values.reshape(-1, channels)).reshape(batch, count, -1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
