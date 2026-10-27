from collections.abc import Sequence
from itertools import pairwise

import torch.nn as nn

from pinn_jc.models.activations import Sine


class MLP(nn.Module):
    """Fully connected network: hidden layers with activation and dropout, linear output."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_units: Sequence[int],
        activation: nn.Module | Sequence[nn.Module] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        widths = [in_features, *hidden_units]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in pairwise(widths)
        )
        self.output_layer = nn.Linear(hidden_units[-1], out_features)

        if activation is None:
            activation = Sine()
        if isinstance(activation, nn.Module):
            activation = [activation] * len(hidden_units)
        if len(activation) != len(hidden_units):
            raise ValueError(
                f"got {len(activation)} activations for {len(hidden_units)} hidden layers"
            )
        self.activations = nn.ModuleList(activation)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in hidden_units)

    def forward(self, x):
        for layer, activation, dropout in zip(
            self.hidden_layers, self.activations, self.dropouts, strict=True
        ):
            x = dropout(activation(layer(x)))
        return self.output_layer(x)
