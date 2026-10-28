"""Network for the complex state vector ψ(t)."""

import torch
import torch.nn as nn


class ComplexStateNet(nn.Module):
    """ψ(t) = real(t) + i imag(t), with separate real-valued networks for each part.

    Both networks map times of shape (N, 1) to (N, D); the output is complex (N, D).
    """

    def __init__(self, real: nn.Module, imag: nn.Module):
        super().__init__()
        self.real = real
        self.imag = imag

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Map times (N, 1) to the complex state (N, D)."""
        return self.real(time) + 1j * self.imag(time)
