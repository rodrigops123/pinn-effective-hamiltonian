"""Global defaults shared across the package. Replaced by config dataclasses in Phase 5."""

from pinn_jc.models.activations import SIN

SEED = 42

model_train_params = {
    "units": [100, 100, 100],
    "activation": SIN(),
    "learning_rate": 0.01,
}

DEVICE = "cpu"
