"""Learnable, non-negative coupling constants."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Positivity = Literal["softplus", "abs"]


class Couplings(nn.Module):
    """Learnable non-negative coupling constants g = f(θ), θ ~ U(0, 1) initially.

    ``positivity`` chooses f: softplus (smooth, the default) or abs (|θ|, whose
    gradient flips sign at θ = 0). ``initial`` and ``fixed`` hold one entry per
    coupling, None where they do not apply: an ``initial`` value g sets θ = f⁻¹(g)
    instead of the random draw; a ``fixed`` value is returned as is and never learned.
    θ is drawn for every coupling in any case, so the random stream does not depend on
    them. With ``draw="softplus"`` the random start is g = softplus(θ) (0.69 to 1.31)
    whatever ``positivity`` is, so abs and softplus runs with the same seed start from
    the same couplings; the default, ``draw="positivity"``, starts at g = f(θ).
    """

    def __init__(
        self,
        n_couplings: int,
        positivity: Positivity = "softplus",
        initial: Sequence[float | None] | None = None,
        fixed: Sequence[float | None] | None = None,
        draw: Literal["positivity", "softplus"] = "positivity",
    ):
        super().__init__()
        if positivity not in ("softplus", "abs"):
            raise ValueError(
                f"positivity must be 'softplus' or 'abs', got {positivity!r}"
            )
        initial = list(initial) if initial is not None else [None] * n_couplings
        fixed = list(fixed) if fixed is not None else [None] * n_couplings
        if len(initial) != n_couplings or len(fixed) != n_couplings:
            raise ValueError(
                f"initial and fixed need {n_couplings} entries, "
                f"got {len(initial)} and {len(fixed)}"
            )
        self.positivity = positivity
        if draw not in ("positivity", "softplus"):
            raise ValueError(f"draw must be 'positivity' or 'softplus', got {draw!r}")
        self.raw = nn.Parameter(torch.rand(n_couplings))
        with torch.no_grad():
            if draw == "softplus" and positivity == "abs":
                self.raw.copy_(F.softplus(self.raw))
            for i, value in enumerate(initial):
                if value is not None:
                    self.raw[i] = _inverse(value, positivity)
        # Rebuilt from the config on loading, so not part of the state dict.
        self.register_buffer(
            "fixed_mask", torch.tensor([v is not None for v in fixed]), persistent=False
        )
        self.register_buffer(
            "fixed_values",
            torch.tensor([0.0 if v is None else float(v) for v in fixed]),
            persistent=False,
        )

    def forward(self) -> torch.Tensor:
        """Return the non-negative couplings, shape (n_couplings,)."""
        if self.positivity == "softplus":
            learned = F.softplus(self.raw)
        else:
            learned = self.raw.abs()
        return torch.where(self.fixed_mask, self.fixed_values, learned)


def _inverse(value, positivity):
    """θ with f(θ) = value."""
    if value < 0 or (positivity == "softplus" and value == 0):
        raise ValueError(
            f"initial coupling {value} is out of the range of {positivity}"
        )
    if positivity == "abs":
        return value
    return value + math.log(-math.expm1(-value))  # log(e^g - 1), stable for large g
