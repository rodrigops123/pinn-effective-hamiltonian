import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch

from utils import msa
from config import global_variables


a = torch.tensor(global_variables.a.full(), dtype=torch.complex64)
sm = torch.tensor(global_variables.sm.full(), dtype=torch.complex64)
a_dag = a.T.conj()
sm_dag = sm.T.conj()


def hamiltonian_with_params(
    picture: str, params: list, coupling_strength: float
) -> torch.Tensor:
    """
    Returns the Hamiltonian based on the specified picture and parameters.
    """
    if picture == "interaction":
        hamiltonian = coupling_strength * (a_dag @ sm + a @ sm_dag)

    elif picture == "atom":
        hamiltonian = abs(params[0] - params[1]) / 2 * a_dag @ a + coupling_strength * (
            a_dag @ sm + a @ sm_dag
        )

    return hamiltonian


def loss_ic(nn_state, sim_state):
    loss = msa(nn_state[0], sim_state[0])
    return loss


def loss_norm(nn_state):
    squared_sum = (abs(nn_state) ** 2).sum(-1)
    loss = msa(squared_sum, 1)
    return loss


def loss_data(nn_state, operators, sim_expect, n_points=None):
    """
    Computes the loss using n_points randomly selected from the training set.
    If n_points is None or 1, uses only the first instant (as in the original loss_data).
    operators: list or tensor of shape (n_observables, D, D)
    sim_expect: shape (n_time_steps, n_observables)
    """
    if n_points is None or n_points == 1:
        # Use only the first instant for all observables
        expected_values_nn = []
        for op in operators:
            ev = (torch.conj(nn_state[0]) @ (op @ nn_state[0])).real
            expected_values_nn.append(ev)
        expected_values_nn = torch.stack(expected_values_nn)
        loss = msa(expected_values_nn, sim_expect[0])
        return loss

    torch.manual_seed(global_variables.SEED)

    n_samples = nn_state.shape[0]
    indices = torch.randperm(n_samples)[:n_points]

    selected_nn = nn_state[indices]  # (n_points, D)
    selected_sim = sim_expect[indices]  # (n_points, n_observables)

    expected_values_nn = []
    for op in operators:
        # For each observable, compute <psi|O|psi> for all selected states
        ev = torch.einsum("ni,ij,nj->n", selected_nn.conj(), op, selected_nn).real
        expected_values_nn.append(ev)
    expected_values_nn = torch.stack(
        expected_values_nn, dim=1
    )  # (n_points, n_observables)

    loss = msa(expected_values_nn, selected_sim)
    return loss


def loss_ode(H_, nn_state, tempo):
    nn_real = nn_state.real
    nn_imag = nn_state.imag
    n_time_steps = tempo.shape[0]
    state = nn_real + 1j * nn_imag
    H_psi = 1j * (state @ H_)

    # Calculando o gradiente de drho_dt separando a parte real e imagina
    loss = 0
    for i in range(global_variables.JC_DIM):
        drho_dt_real = torch.autograd.grad(
            outputs=nn_real[:, i],
            inputs=tempo,
            grad_outputs=torch.ones_like(nn_real[:, i]),
            retain_graph=True,
            create_graph=True,
        )[0]

        drho_dt_imag = torch.autograd.grad(
            outputs=nn_imag[:, i],
            inputs=tempo,
            grad_outputs=torch.ones_like(nn_imag[:, i]),
            retain_graph=True,
            create_graph=True,
        )[0]

        drho_dt = drho_dt_real + 1j * drho_dt_imag

        loss += torch.mean( abs( drho_dt + H_psi[:, i].reshape((n_time_steps, 1)) ) ) ** 2

    return loss
