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


class InitialConditionStateNet(nn.Module):
    """ψ(t) = ψ0 + (t - t0) net(t), so that ψ(t0) = ψ0 holds exactly (hard constraint).

    ``psi0`` (D,) and ``t0`` are buffers: they are saved and restored with the state
    dict, and are not trained.
    """

    def __init__(self, net: nn.Module, psi0: torch.Tensor, t0: float = 0.0):
        super().__init__()
        self.net = net
        self.register_buffer("psi0", torch.as_tensor(psi0, dtype=torch.complex64))
        self.register_buffer("t0", torch.tensor(float(t0)))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Map times (N, 1) to the complex state (N, D)."""
        return self.psi0 + (time - self.t0) * self.net(time)
