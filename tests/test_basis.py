"""Excitation-number truncation of the basis."""

import numpy as np
import pytest
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.evaluation import evaluate
from pinn_jc.physics import Hamiltonian
from pinn_jc.physics.basis import basis_indices
from pinn_jc.training.config import PhysicsConfig
from pinn_jc.training.losses import loss_ode
from pinn_jc.training.trainer import build_models

DIMS = {"atom": 2, "field": 4}
PARAMS = {"wc": 1.0, "wa": 1.0, "g1": 1.0}


def test_excitation_truncation_drops_only_the_top_excited_level():
    np.testing.assert_array_equal(basis_indices(DIMS, "field"), np.arange(8))
    # |g,0..3>, |e,0..2>; |e,3> (index 7) has 4 excitations.
    np.testing.assert_array_equal(basis_indices(DIMS, "excitation"), np.arange(7))


def test_top_level_is_stationary_in_the_full_basis():
    """The artefact the excitation truncation removes: H |e, n_max> = 0."""
    hamiltonian = Hamiltonian("atom", DIMS).to_torch(PARAMS)

    assert hamiltonian[:, -1].abs().max() == 0


def test_truncated_data_satisfy_the_truncated_ode():
    data = simulate(
        PARAMS,
        2 * np.pi,
        2000,
        "coherent",
        "atom",
        DIMS,
        init_state_options={"alpha": 1.0},
        truncation="excitation",
    )
    full = simulate(
        PARAMS,
        2 * np.pi,
        2000,
        "coherent",
        "atom",
        DIMS,
        init_state_options={"alpha": 1.0},
    )
    hamiltonian = Hamiltonian("atom", DIMS, truncation="excitation").to_torch(PARAMS)
    states = data.states.numpy().astype(np.complex128)
    time = data.time.detach().numpy().ravel()
    dpsi_dt = np.gradient(states, time, axis=0, edge_order=2)
    residual = dpsi_dt + 1j * states @ hamiltonian.numpy().T

    assert data.states.shape == (2000, 7)
    assert data.observables.shape == (2, 7, 7)
    assert np.abs(residual).mean(axis=0).sum() < 1e-3
    torch.testing.assert_close(data.expectations, full.expectations)


def test_excitation_truncation_rejects_counter_rotating_terms():
    with pytest.raises(ValueError, match="excitation-conserving"):
        Hamiltonian("rabi2_ip", DIMS, truncation="excitation")


def test_excitation_truncation_rejects_populated_dropped_states():
    """Counter-rotating terms take |e,0> -> |g,1> -> |e,2>, a dropped state."""
    params = {"wc": 1.0, "wa": 1.0, "g1": 1.0, "g2": 1.0}
    with pytest.raises(ValueError, match="drops populated states"):
        simulate(
            params,
            2.0,
            20,
            "fock",
            "rabi2",
            {"atom": 2, "field": 3},
            truncation="excitation",
        )


def test_evaluation_in_the_truncated_basis():
    physics = PhysicsConfig(
        "atom", PARAMS, "coherent", DIMS, {"alpha": 1.0}, truncation="excitation"
    )
    state, _ = build_models(physics.hilbert_dim)

    evaluation = evaluate({"state": state}, physics, 1.0, 5)

    assert physics.hilbert_dim == 7
    assert evaluation.predicted_state.shape == (5, 7)
    assert len(evaluation.basis_labels) == 7
    assert evaluation.basis_labels[-1] == r"$|e,2\rangle$"


def test_ode_mean_reduction_averages_over_components():
    hamiltonian = torch.randn(5, 5, dtype=torch.complex64)
    psi = torch.randn(30, 5, dtype=torch.complex64)
    dpsi = torch.randn(30, 5, dtype=torch.complex64)

    total = loss_ode(hamiltonian, psi, dpsi)
    mean = loss_ode(hamiltonian, psi, dpsi, reduction="mean")

    torch.testing.assert_close(mean, total / 5)
