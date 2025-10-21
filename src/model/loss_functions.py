import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import qutip

from utils import mae
from config import global_variables


def hamiltonian_with_params(
    picture: str, params: dict, coupling_strength: float, dims: dict
) -> torch.Tensor:
    """
    Returns the Hamiltonian based on the specified picture and parameters.
    """

    a       = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm      = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))
    sz      = qutip.tensor(qutip.sigmaz(), qutip.qeye(dims["field"]))
    sy      = qutip.tensor(qutip.sigmay(), qutip.qeye(dims["field"]))
    sx      = qutip.tensor(qutip.sigmax(), qutip.qeye(dims["field"]))

    a_dag   = a.dag()
    sm_dag  = sm.dag()

    a       = torch.tensor(a.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    sm      = torch.tensor(sm.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    a_dag   = torch.tensor(
        a_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE
    )
    sm_dag  = torch.tensor(
        sm_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE
    )
    sz      = torch.tensor(sz.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    sy      = torch.tensor(sy.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    sx      = torch.tensor(sx.full(), dtype=torch.complex64, device=global_variables.DEVICE) 

    if picture == "interaction":
        hamiltonian = coupling_strength[0] * (a_dag @ sm + a @ sm_dag)

    elif picture == "atom":
        hamiltonian = 0.5 * abs(
            params["wc"] - params["wa"]
        ) / 2 * a_dag @ a + coupling_strength[0] * (a_dag @ sm + a @ sm_dag)

    elif picture == "full":
        hamiltonian =  params["wc"] * a_dag * a  + params["wa"] * sm_dag * sm + coupling_strength[0] * (a_dag @ sm + a @ sm_dag)

    elif picture == "rabi":

        hamiltonian =  params["wc"] * a_dag * a  + params["wa"] * sm_dag * sm + \
                        coupling_strength[0] * a_dag @ sm + \
                        coupling_strength[1] * a     @ sm_dag + \
                        coupling_strength[2] * a     @ sm       + \
                        coupling_strength[3] * a_dag @ sm_dag
    elif picture == "rabi2":

        hamiltonian =  params["wc"] * a_dag * a  + params["wa"] * sm_dag * sm + \
                        coupling_strength[0] * (a_dag @ sm + a @ sm_dag )+ \
                        coupling_strength[1] * (a @ sm + a_dag @ sm_dag)
    elif picture == "geral":   
        hamiltonian =   params["wc"] * a_dag * a  + params["wa"] * sm_dag * sm + \
                        coupling_strength[0] * a_dag @ sm + \
                        coupling_strength[1] * a     @ sm_dag + \
                        coupling_strength[2] * a     @  sm       + \
                        coupling_strength[3] * a_dag @ sm_dag + \
                        coupling_strength[4] * a_dag @ sz + \
                        coupling_strength[5] * a_dag @ sy + \
                        coupling_strength[6] * a_dag @ sx + \
                        coupling_strength[7] * a     @ sz + \
                        coupling_strength[8] * a     @ sy + \
                        coupling_strength[9] * a     @ sx 
    else:
        raise ValueError(
            "Invalid picture. Choose from 'interaction', 'atom', or 'full'."
        )

    return hamiltonian


def loss_ic(nn_state, sim_state):
    loss = mae(nn_state[0], sim_state[0])
    return loss


def loss_norm(nn_state):
    squared_sum = (abs(nn_state) ** 2).sum(-1)
    loss = mae(squared_sum, 1)
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
        loss = mae(expected_values_nn, sim_expect[0])
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

    loss = mae(expected_values_nn, selected_sim)
    return loss


def loss_ode(H_, nn_state, tempo):
    nn_real = nn_state.real
    nn_imag = nn_state.imag
    n_time_steps = tempo.shape[0]
    state = nn_real + 1j * nn_imag
    H_psi = 1j * (state @ H_)

    # Calculando o gradiente de dpsi_dt separando a parte real e imagina
    loss = 0
    for i in range(state.shape[-1]):
        dpsi_dt_real = torch.autograd.grad(
            outputs=nn_real[:, i],
            inputs=tempo,
            grad_outputs=torch.ones_like(nn_real[:, i]),
            retain_graph=True,
            create_graph=True,
        )[0]

        dpsi_dt_imag = torch.autograd.grad(
            outputs=nn_imag[:, i],
            inputs=tempo,
            grad_outputs=torch.ones_like(nn_imag[:, i]),
            retain_graph=True,
            create_graph=True,
        )[0]

        dpsi_dt = dpsi_dt_real + 1j * dpsi_dt_imag

        loss += torch.mean(abs(dpsi_dt + H_psi[:, i].reshape((n_time_steps, 1))))

    return loss
