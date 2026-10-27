"""Characterization tests: pin the current numerical behaviour of the pipeline.

They do not claim the behaviour is *correct* (the physics and regression tests do
that). They exist so that structural refactors can prove they changed nothing.
Intentional changes must regenerate the affected golden files with UPDATE_GOLDEN=1
in the same commit, and say why in the commit message.
"""

import numpy as np
import pytest
import torch

from conftest import seed_everything
from pinn_jc.data.simulate import simulate
from pinn_jc.physics import Hamiltonian
from pinn_jc.training.config import (
    DataConfig,
    ExperimentConfig,
    PhysicsConfig,
    TrainingConfig,
)
from pinn_jc.training.trainer import train, train_with_parameter

SIMULATION_CASES = {
    "interaction_fock": dict(
        params={"wc": 1.0, "wa": 1.0, "g1": 0.7},
        picture="interaction",
        init_state="fock",
        dims={"atom": 2, "field": 2},
    ),
    "atom_coherent_detuned": dict(
        params={"wc": 1.3, "wa": 1.0, "g1": 0.7},
        picture="atom",
        init_state="coherent",
        dims={"atom": 2, "field": 4},
    ),
    # Resonant with g2 = 0: simulated in the interaction picture instead.
    "full_fock_superposition_resonant": dict(
        params={"wc": 1.0, "wa": 1.0, "g1": 0.7},
        picture="full",
        init_state="fock_superposition",
        dims={"atom": 2, "field": 3},
    ),
    "rabi2_squeeze": dict(
        params={"wc": 1.0, "wa": 1.2, "g1": 0.7, "g2": 0.4},
        picture="rabi2",
        init_state="squeeze",
        dims={"atom": 2, "field": 4},
    ),
}


@pytest.mark.parametrize("frame", ["own", "interaction"])
@pytest.mark.parametrize("case", sorted(SIMULATION_CASES))
def test_simulate(golden, case, frame):
    data = simulate(
        tfinal=2 * np.pi,
        n_time_steps=50,
        frame=None if frame == "own" else frame,
        **SIMULATION_CASES[case],
    )
    golden(
        f"simulate__{case}__{frame}",
        {
            "state": data.states.numpy(),
            "expect": data.expectations.numpy(),
            "operators": data.observables.numpy(),
            "time": data.time.detach().numpy(),
        },
    )


HAMILTONIAN_CASES = {
    "interaction": [0.7],
    "atom": [0.7],
    "full": [0.7],
    "rabi2": [0.7, 0.4],
    "rabi2_ip": [0.7, 0.4],
}


@pytest.mark.parametrize("picture", sorted(HAMILTONIAN_CASES))
def test_hamiltonian_to_torch(golden, picture):
    time = torch.linspace(0, 2, 7).reshape(-1, 1)
    hamiltonian = Hamiltonian(picture, {"atom": 2, "field": 3}).to_torch(
        params={"wc": 1.3, "wa": 1.0},
        couplings=torch.tensor(HAMILTONIAN_CASES[picture]),
        time=time,
    )
    golden(f"hamiltonian__{picture}", {"hamiltonian": hamiltonian.numpy()})


TRAINING_CASES = {
    "atom_fock": dict(
        params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
        picture="atom",
        init_state="fock",
        dims={"atom": 2, "field": 2},
        n_parameter=1,
    ),
    "atom_coherent": dict(
        params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
        picture="atom",
        init_state="coherent",
        dims={"atom": 2, "field": 4},
        n_parameter=1,
    ),
    "rabi2_fock_usc": dict(
        params={"wc": 1.0, "wa": 1.0, "g1": 1.0, "g2": 1.0},
        picture="rabi2",
        init_state="fock",
        dims={"atom": 2, "field": 4},
        n_parameter=2,
    ),
}


def _history_arrays(loss_dict):
    return {key: np.asarray(values) for key, values in loss_dict.items()}


def _predictions(models, n_time_steps=40, tfinal=2 * np.pi):
    time = torch.linspace(0, tfinal, n_time_steps).reshape(-1, 1)
    with torch.no_grad():
        return {
            "pred_real": models["state"].real(time).numpy(),
            "pred_imag": models["state"].imag(time).numpy(),
        }


@pytest.mark.parametrize("case", sorted(TRAINING_CASES))
def test_train_with_parameter(golden, case):
    seed_everything(0)
    n_time_steps = 40
    models, loss_dict = train_with_parameter(
        epochs=15,
        tfinal=2 * np.pi,
        n_time_steps=n_time_steps,
        n_points_loss=n_time_steps,
        is_scaled=False,
        **TRAINING_CASES[case],
    )
    golden(
        f"train_with_parameter__{case}",
        {**_history_arrays(loss_dict), **_predictions(models)},
    )


def test_train_with_fixed_couplings_and_validation_window(golden):
    """The pre-Phase-5 fixed-Hamiltonian train(): last 20% of the grid held out,
    expectation data at the first instant only."""
    config = ExperimentConfig(
        physics=PhysicsConfig(
            picture="atom",
            params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
            init_state="fock",
            dims={"atom": 2, "field": 2},
            learn_couplings=False,
        ),
        data=DataConfig(tfinal=2 * np.pi, n_time_steps=40, validation_fraction=0.2),
        training=TrainingConfig(epochs=15, seed=0, progress_bar=False),
    )
    result = train(config)
    history = result.history.to_legacy_dict()
    golden(
        "train__atom_fock", {**_history_arrays(history), **_predictions(result.models)}
    )
