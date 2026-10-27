"""Characterization tests: pin the current numerical behaviour of the pipeline.

They do not claim the behaviour is *correct* (see test_physics_consistency.py and
test_known_bugs.py for that). They exist so that the structural refactor phases can
prove they changed nothing. Intentional changes (e.g. the Phase 3 bug fixes) must
regenerate the affected golden files with UPDATE_GOLDEN=1 in the same commit.
"""

import numpy as np
import pytest
import torch

from conftest import seed_everything
from src.data_simulation.jaynes_cummings_data import data_jc
from src.model.loss_functions import hamiltonian_with_params
from src.model.train_and_eval import train, train_with_parameter

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
    # Resonant with g2 = 0: data_jc silently simulates in the interaction picture.
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


@pytest.mark.parametrize("state_frame", ["lab", "interaction"])
@pytest.mark.parametrize("case", sorted(SIMULATION_CASES))
def test_data_jc(golden, case, state_frame):
    y, expect, hamiltonian, operators, time = data_jc(
        tfinal=2 * np.pi, n_time_steps=50, state_frame=state_frame, **SIMULATION_CASES[case]
    )
    golden(
        f"data_jc__{case}__{state_frame}",
        {
            "state": y.numpy(),
            "expect": expect.numpy(),
            "hamiltonian": hamiltonian.numpy(),
            "operators": operators.numpy(),
            "time": time.detach().numpy(),
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
def test_hamiltonian_with_params(golden, picture):
    time = torch.linspace(0, 2, 7).reshape(-1, 1)
    hamiltonian = hamiltonian_with_params(
        picture=picture,
        params={"wc": 1.3, "wa": 1.0},
        coupling_strength=torch.tensor(HAMILTONIAN_CASES[picture]),
        dims={"atom": 2, "field": 3},
        tempo=time,
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


def _predictions(models_dict, n_time_steps=40, tfinal=2 * np.pi):
    time = torch.linspace(0, tfinal, n_time_steps).reshape(-1, 1)
    with torch.no_grad():
        return {
            "pred_real": models_dict["model_real"](time).numpy(),
            "pred_imag": models_dict["model_imag"](time).numpy(),
        }


@pytest.mark.parametrize("case", sorted(TRAINING_CASES))
def test_train_with_parameter(golden, case):
    seed_everything(0)
    n_time_steps = 40
    models_dict, loss_dict = train_with_parameter(
        epochs=15,
        tfinal=2 * np.pi,
        n_time_steps=n_time_steps,
        n_points_loss=n_time_steps,
        is_scaled=False,
        **TRAINING_CASES[case],
    )
    golden(
        f"train_with_parameter__{case}",
        {**_history_arrays(loss_dict), **_predictions(models_dict)},
    )


def test_train(golden):
    seed_everything(0)
    models_dict, loss_dict = train(
        epochs=15,
        params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
        tfinal=2 * np.pi,
        n_time_steps=40,
        init_state="fock",
        picture="atom",
        dims={"atom": 2, "field": 2},
    )
    golden("train__atom_fock", {**_history_arrays(loss_dict), **_predictions(models_dict)})
