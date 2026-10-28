"""Activation functions."""

import torch
import torch.nn as nn


class Sine(nn.Module):
    """sin(x), elementwise."""

    def forward(self, x):
        """Apply sin elementwise."""
        return torch.sin(x)
