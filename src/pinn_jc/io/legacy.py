"""Read results written before the current model classes existed.

Two legacy formats exist:

- result pickles holding whole ``Neural_Net`` objects (one network for the real part,
  carrying the learnable couplings as ``.param``, one for the imaginary part), saved
  under the import paths of the time;
- runs converted by ``scripts/migrate_pickles.py``: ``{"model_real": state_dict,
  "model_imag": state_dict}``.

Both are converted to the current ``ComplexStateNet`` and ``Couplings`` modules.
"""

import pickle
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from pinn_jc.models import MLP, ComplexStateNet, Couplings, Sine
from pinn_jc.models.couplings import Positivity


class LegacyNeuralNet(nn.Module):
    """The pre-Phase-4 network, kept verbatim so that legacy pickles can be loaded."""

    def __init__(
        self,
        units,
        activation,
        input=1,
        output=1,
        create_parameter=False,
        n_paramater=1,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.units = units
        self.output = output
        self.create_parameter = create_parameter
        self.n_paramater = n_paramater
        self.dropout_prob = dropout_prob
        self.hidden_layers = nn.ModuleList([nn.Linear(input, units[0])])
        self.hidden_layers.extend(
            [nn.Linear(units[i], units[i + 1]) for i in range(len(self.units) - 1)]
        )
        self.output_layer = nn.Linear(units[-1], self.output)
        if not isinstance(activation, list):
            self.activation = [activation] * len(self.units)
        else:
            self.activation = activation
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_prob) for _ in range(len(self.units))]
        )
        if self.create_parameter:
            self.param = nn.Parameter(torch.rand(self.n_paramater), requires_grad=True)

    def forward(self, x):
        for layer, activation, dropout in zip(
            self.hidden_layers, self.activation, self.dropouts, strict=True
        ):
            x = activation(layer(x))
            x = dropout(x)
        return self.output_layer(x)


LEGACY_CLASSES = {
    # thesis-era pickles
    ("src.model.neural_network", "Neural_Net"): (
        "pinn_jc.io.legacy",
        "LegacyNeuralNet",
    ),
    ("utils", "SIN"): ("pinn_jc.models.activations", "Sine"),
    # pickles written by the notebooks between Phases 1 and 4
    ("pinn_jc.models.mlp", "Neural_Net"): ("pinn_jc.io.legacy", "LegacyNeuralNet"),
    ("pinn_jc.models.activations", "SIN"): ("pinn_jc.models.activations", "Sine"),
}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module, name = LEGACY_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


def load_legacy_pickle(path):
    """Only use on trusted files: unpickling can execute arbitrary code."""
    with Path(path).open("rb") as f:
        return LegacyUnpickler(f).load()


def _mlp_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> MLP:
    hidden = sorted(
        (
            k
            for k in state_dict
            if k.startswith("hidden_layers.") and k.endswith("weight")
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    mlp = MLP(
        in_features=state_dict[hidden[0]].shape[1],
        out_features=state_dict["output_layer.weight"].shape[0],
        hidden_units=[state_dict[k].shape[0] for k in hidden],
        activation=Sine(),
    )
    mlp.load_state_dict({k: v for k, v in state_dict.items() if k != "param"})
    return mlp.eval()


def models_from_legacy_state_dicts(
    state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    positivity: Positivity = "softplus",
) -> dict:
    """``{"model_real": sd, "model_imag": sd}`` -> ``{"state": ..., "couplings": ...}``.

    ``positivity`` is not recorded in legacy files; it only matters when the couplings
    module is evaluated, not for the logged ``learned_param`` history.
    """
    with (
        torch.random.fork_rng()
    ):  # building the modules must not consume the caller's RNG
        state = ComplexStateNet(
            _mlp_from_state_dict(state_dicts["model_real"]),
            _mlp_from_state_dict(state_dicts["model_imag"]),
        )
        couplings = None
        if "param" in state_dicts["model_real"]:
            raw = state_dicts["model_real"]["param"]
            couplings = Couplings(raw.numel(), positivity)
            couplings.load_state_dict({"raw": raw})
    return {"state": state, "couplings": couplings}


def as_models(models: Mapping) -> dict:
    """Accept current ``{"state", "couplings"}`` models or legacy ``models_dict``s."""
    if "state" in models:
        return dict(models)
    return models_from_legacy_state_dicts(
        {name: models[name].state_dict() for name in ("model_real", "model_imag")}
    )
