from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Positivity = Literal["softplus", "abs"]


class Couplings(nn.Module):
    """Learnable non-negative coupling constants g = f(θ), θ ~ U(0, 1) initially.

    ``positivity`` chooses f: softplus (smooth, the default) or abs (|θ|, whose
    gradient flips sign at θ = 0).
    """

    def __init__(self, n_couplings: int, positivity: Positivity = "softplus"):
        super().__init__()
        if positivity not in ("softplus", "abs"):
            raise ValueError(
                f"positivity must be 'softplus' or 'abs', got {positivity!r}"
            )
        self.positivity = positivity
        self.raw = nn.Parameter(torch.rand(n_couplings))

    def forward(self) -> torch.Tensor:
        if self.positivity == "softplus":
            return F.softplus(self.raw)
        return self.raw.abs()
