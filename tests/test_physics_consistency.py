"""Physics invariants: the exact solution must satisfy the equation the PINN enforces.

The exact state is built in closed form with ``torch.matrix_exp`` so that the real
``loss_ode`` (autograd time derivative) is exercised, not a finite-difference proxy.
"""

import numpy as np
import pytest
import qutip
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.physics import Hamiltonian, initial_state
from pinn_jc.training.losses import loss_ode

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
    psi0 = _to_torch(initial_state("fock", DIMS)).ravel()
    hamiltonian = Hamiltonian(picture, DIMS)
    psi = _exact_state(_to_torch(hamiltonian.to_qobj(params)), psi0, time)

    hamiltonian = hamiltonian.to_torch(params, torch.tensor(couplings), time)

    assert loss_ode(hamiltonian, psi, time).item() < ODE_TOLERANCE


def test_loss_ode_vanishes_on_exact_interaction_picture_solution():
    """Regression for bug A: loss_ode applied H^T, wrong for the complex rabi2_ip."""
    params = {"wc": 1.3, "wa": 1.0, "g1": 0.7, "g2": 0.4}
    time = _time_grid()
    psi0 = _to_torch(initial_state("fock", DIMS)).ravel()
    psi = _exact_state(
        _to_torch(Hamiltonian("rabi2", DIMS).to_qobj(params)),
        psi0,
        time,
        frame_hamiltonian=_to_torch(_free_hamiltonian(params, DIMS)),
    )

    hamiltonian = Hamiltonian("rabi2_ip", DIMS).to_torch(
        params, torch.tensor([0.7, 0.4]), time
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


# (picture, frame requested from simulate, picture whose ODE the states must satisfy)
# The first group mirrors the training data of each picture; the rest check frame conversions
# against the Hamiltonian of the target frame. Regression for bug B (data rotated
# into a frame that did not match the ODE).
FRAME_CASES = {
    "atom_resonant": ("atom", None, "atom", {"wc": 1.0, "wa": 1.0, "g1": 1.0}),
    "atom_detuned": ("atom", None, "atom", {"wc": 1.3, "wa": 1.0, "g1": 1.0}),
    "rabi2_ip_resonant_jc": (
        "rabi2_ip",
        None,
        "rabi2_ip",
        {"wc": 1.0, "wa": 1.0, "g1": 1.0, "g2": 0.0},
    ),
    "rabi2_ip_detuned_usc": (
        "rabi2_ip",
        None,
        "rabi2_ip",
        {"wc": 1.0, "wa": 1.2, "g1": 1.0, "g2": 1.0},
    ),
    "full_resonant_in_lab": ("full", None, "full", {"wc": 1.0, "wa": 1.0, "g1": 0.5}),
    "full_detuned_in_rotating": (
        "full",
        "rotating",
        "atom",
        {"wc": 1.3, "wa": 1.0, "g1": 0.5},
    ),
    "full_resonant_in_interaction": (
        "full",
        "interaction",
        "interaction",
        {"wc": 1.0, "wa": 1.0, "g1": 0.5},
    ),
    "rabi2_lab_in_interaction": (
        "rabi2",
        "interaction",
        "rabi2_ip",
        {"wc": 1.0, "wa": 1.2, "g1": 0.5, "g2": 0.3},
    ),
}


@pytest.mark.parametrize(
    "picture, frame, ode_picture, params",
    list(FRAME_CASES.values()),
    ids=list(FRAME_CASES),
)
def test_simulated_states_satisfy_the_ode_of_their_frame(
    picture, frame, ode_picture, params
):
    dims = {"atom": 2, "field": 4}
    data = simulate(params, 2.0, 4001, "fock", picture, dims, frame=frame)
    hamiltonian = Hamiltonian(ode_picture, dims).to_torch(
        params, time=data.time.detach()
    )

    assert _finite_difference_residual(data.states, data.time, hamiltonian) < 1e-2


@pytest.mark.parametrize("picture", ["atom", "rabi2_ip"])
def test_loss_ode_vanishes_on_exact_solution_in_scaled_time(picture):
    """The network sees τ = t / T; loss_ode then enforces dψ/dτ = -i T H(t) ψ."""
    params = {"wc": 1.3, "wa": 1.0, "g1": 0.7, "g2": 0.4}
    total_time = 2 * np.pi
    tau = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    time = tau * total_time
    psi0 = _to_torch(initial_state("fock", DIMS)).ravel()
    lab = picture == "rabi2_ip"
    psi = _exact_state(
        _to_torch(Hamiltonian("rabi2" if lab else picture, DIMS).to_qobj(params)),
        psi0,
        time,
        frame_hamiltonian=_to_torch(_free_hamiltonian(params, DIMS)) if lab else None,
    )
    couplings = torch.tensor([0.7, 0.4] if lab else [0.7])

    hamiltonian = Hamiltonian(picture, DIMS).to_torch(params, couplings, time.detach())

    assert loss_ode(hamiltonian, psi, tau, time_scale=total_time).item() < ODE_TOLERANCE
