"""Seeding of every random number generator."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator a training run draws from."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
