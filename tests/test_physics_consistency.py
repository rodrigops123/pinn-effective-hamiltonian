"""Physics invariants: the exact solution must satisfy the equation the PINN enforces.

The exact state is built in closed form with ``torch.matrix_exp`` so that the real
``loss_ode`` (autograd time derivative) is exercised, not a finite-difference proxy.
Tests marked ``xfail(strict=True)`` document known bugs; they will start failing
(XPASS) as soon as the bug is fixed, which forces the marker to be removed.
"""

import numpy as np
import pytest
import qutip
import torch

from pinn_jc.data.simulate import (
    choose_init_state,
    chooses_hamiltonian,
    data_jc,
)
from pinn_jc.training.losses import hamiltonian_with_params, loss_ode

DIMS = {"atom": 2, "field": 3}
ODE_TOLERANCE = 1e-4  # float32 autograd residual of the exact solution is ~1e-6


def _time_grid(n_time_steps=200, tfinal=2 * np.pi):
    return torch.linspace(0, tfinal, n_time_steps).reshape(-1, 1).requires_grad_(True)


def _to_torch(qobj):
    return torch.tensor(qobj.full(), dtype=torch.complex64)


def _exact_state(hamiltonian, psi0, time, frame_hamiltonian=None):
    """psi(t) = exp(-iHt) psi0, optionally rotated by exp(+i H0 t). Rows are times."""
    propagator = torch.matrix_exp(-1j * hamiltonian[None] * time[:, :, None])
    psi = propagator @ psi0
    if frame_hamiltonian is not None:
        rotation = torch.matrix_exp(1j * frame_hamiltonian[None] * time[:, :, None])
        psi = (rotation @ psi[..., None]).squeeze(-1)
    return psi


def _free_hamiltonian(params, dims):
    """H0 = wc a†a + wa σ+σ- in the qutip.tensor(atom, field) basis."""
    return params["wc"] * qutip.tensor(
        qutip.qeye(dims["atom"]), qutip.num(dims["field"])
    ) + params["wa"] * qutip.tensor(qutip.num(dims["atom"]), qutip.qeye(dims["field"]))


TIME_INDEPENDENT_CASES = {
    "interaction": ({"wc": 1.0, "wa": 1.0, "g1": 0.7}, [0.7]),
    "atom": ({"wc": 1.3, "wa": 1.0, "g1": 0.7}, [0.7]),
    "full": ({"wc": 1.3, "wa": 1.0, "g1": 0.7}, [0.7]),
    "rabi2": ({"wc": 1.3, "wa": 1.0, "g1": 0.7, "g2": 0.4}, [0.7, 0.4]),
}


@pytest.mark.parametrize("picture", sorted(TIME_INDEPENDENT_CASES))
def test_loss_ode_vanishes_on_exact_solution(picture):
    params, couplings = TIME_INDEPENDENT_CASES[picture]
    time = _time_grid()
    psi0 = _to_torch(choose_init_state("fock", DIMS)).ravel()
    psi = _exact_state(
        _to_torch(chooses_hamiltonian(picture, params, DIMS)), psi0, time
    )

    hamiltonian = hamiltonian_with_params(
        picture, params, torch.tensor(couplings), DIMS, time
    )

    assert loss_ode(hamiltonian, psi, time).item() < ODE_TOLERANCE


def test_simulation_and_training_hamiltonians_agree():
    params, couplings = TIME_INDEPENDENT_CASES["rabi2"]
    simulated = _to_torch(chooses_hamiltonian("rabi2", params, DIMS))
    trained = hamiltonian_with_params(
        "rabi2", params, torch.tensor(couplings), DIMS, None
    )
    torch.testing.assert_close(simulated, trained)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Bug A: loss_ode computes psi @ H (= H^T psi), wrong for the complex rabi2_ip "
    "Hamiltonian. Fix in Phase 3.",
)
def test_loss_ode_vanishes_on_exact_interaction_picture_solution():
    params = {"wc": 1.3, "wa": 1.0, "g1": 0.7, "g2": 0.4}
    time = _time_grid()
    psi0 = _to_torch(choose_init_state("fock", DIMS)).ravel()
    psi = _exact_state(
        _to_torch(chooses_hamiltonian("rabi2", params, DIMS)),
        psi0,
        time,
        frame_hamiltonian=_to_torch(_free_hamiltonian(params, DIMS)),
    )

    hamiltonian = hamiltonian_with_params(
        "rabi2_ip", params, torch.tensor([0.7, 0.4]), DIMS, time
    )

    assert loss_ode(hamiltonian, psi, time).item() < ODE_TOLERANCE


def _finite_difference_residual(states, time, hamiltonian):
    """Mean |dpsi/dt + i H psi| summed over components, with column-vector convention."""
    states = states.numpy().astype(np.complex128)
    time = time.detach().numpy().ravel()
    hamiltonian = hamiltonian.numpy()
    dpsi_dt = np.gradient(states, time, axis=0, edge_order=2)
    if hamiltonian.ndim == 2:
        h_psi = states @ hamiltonian.T
    else:
        h_psi = np.einsum("nij,nj->ni", hamiltonian, states)
    return np.abs(dpsi_dt + 1j * h_psi).mean(axis=0).sum()


# Mirrors what train_with_parameter does: data in the "interaction" frame, ODE with
# picture "rabi2_ip" when picture == "rabi2", otherwise the requested picture.
FRAME_CASES = {
    "rabi2_detuned_usc": (
        "rabi2",
        "rabi2_ip",
        {"wc": 1.0, "wa": 1.2, "g1": 1.0, "g2": 1.0},
        [1.0, 1.0],
    ),
    "atom_resonant": pytest.param(
        "atom",
        "atom",
        {"wc": 1.0, "wa": 1.0, "g1": 1.0},
        [1.0],
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="Bug B: the 'atom' picture is already a rotating frame, yet the data is "
            "rotated again by exp(+iH0 t). Fix in Phase 3.",
        ),
    ),
    "rabi2_resonant_jc": pytest.param(
        "rabi2",
        "rabi2_ip",
        {"wc": 1.0, "wa": 1.0, "g1": 1.0, "g2": 0.0},
        [1.0, 0.0],
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason="Bug B: data_jc silently simulates in the interaction picture, then "
            "rotates again by exp(+iH0 t). Fix in Phase 3.",
        ),
    ),
}


@pytest.mark.parametrize(
    "picture, picture_train, params, couplings",
    list(FRAME_CASES.values()),
    ids=list(FRAME_CASES),
)
def test_training_data_satisfies_training_hamiltonian(
    picture, picture_train, params, couplings
):
    dims = {"atom": 2, "field": 4}
    states, _, _, _, time = data_jc(
        params, 2.0, 4001, "fock", picture, dims, state_frame="interaction"
    )
    hamiltonian = hamiltonian_with_params(
        picture_train, params, torch.tensor(couplings), dims, time.detach()
    )

    assert _finite_difference_residual(states, time, hamiltonian) < 1e-2
