import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import qutip

from utils import mae, mse
from config import global_variables

def hamiltonian_with_params(
    picture: str,
    params: dict,
    coupling_strength: torch.Tensor,
    dims: dict,
    tempo#: torch.Tensor | None = None,  # needed for *_ip pictures
) -> torch.Tensor:
    """
    Returns Hamiltonian in torch.

    New picture:
      - "rabi2_ip": interaction picture w.r.t. H0 = wc a†a + wa σ+σ-
                   H_I(t) = g1[ a†σ- e^{+iΔt} + aσ+ e^{-iΔt} ]
                          + g2[ aσ- e^{-iΣt} + a†σ+ e^{+iΣt} ]
                   with Δ = wc - wa, Σ = wc + wa.
                   Returns shape (N,D,D).

    For time-independent pictures, returns shape (D,D).
    """
    a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))

    a_dag = a.dag()
    sm_dag = sm.dag()

    a = torch.tensor(a.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    sm = torch.tensor(sm.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    a_dag = torch.tensor(a_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE)
    sm_dag = torch.tensor(sm_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE)

    # common operators
    JC_up = a_dag @ sm         # a† σ-
    JC_down = a @ sm_dag       # a  σ+
    CR_down = a @ sm           # a  σ-
    CR_up = a_dag @ sm_dag     # a† σ+

    if picture == "interaction":
        # time-independent JC interaction picture (resonant)
        g1 = coupling_strength if coupling_strength.ndim == 0 else coupling_strength[0]
        return g1 * (JC_up + JC_down)

    elif picture == "atom":
        # rotating @ ωa for both subsystems: Δ a†a + g1(JC)
        g1 = coupling_strength if coupling_strength.ndim == 0 else coupling_strength[0]
        delta = (params["wc"] - params["wa"])
        return delta * (a_dag @ a) + g1 * (JC_up + JC_down)

    elif picture == "full":
        # CONSISTENT: use wa * (σ+σ-) == wa * sm†sm
        g1 = coupling_strength if coupling_strength.ndim == 0 else coupling_strength[0]
        return (
            params["wc"] * (a_dag @ a)
            + params["wa"] * (sm_dag @ sm)
            + g1 * (JC_up + JC_down)
        )

    elif picture == "rabi2":
        g1 = coupling_strength[0]
        g2 = coupling_strength[1]
        return (
            params["wc"] * (a_dag @ a)
            + params["wa"] * (sm_dag @ sm)
            + g1 * (JC_up + JC_down)
            + g2 * (CR_down + CR_up)
        )

    elif picture == "rabi2_ip":
        if tempo is None:
            raise ValueError("tempo must be provided for picture='rabi2_ip'")

        # tempo: (N,1) float
        t = tempo.to(dtype=torch.float32, device=global_variables.DEVICE).view(-1, 1, 1)
        wc = torch.tensor(float(params["wc"]), device=global_variables.DEVICE)
        wa = torch.tensor(float(params["wa"]), device=global_variables.DEVICE)

        delta = (wc - wa)
        sigma = (wc + wa)

        g1 = coupling_strength[0]
        g2 = coupling_strength[1]

        # phases: (N,1,1) complex
        phase_p_delta = torch.exp(1j * delta * t).to(torch.complex64)
        phase_m_delta = torch.exp(-1j * delta * t).to(torch.complex64)
        phase_p_sigma = torch.exp(1j * sigma * t).to(torch.complex64)
        phase_m_sigma = torch.exp(-1j * sigma * t).to(torch.complex64)

        # build H_I(t): (N,D,D)
        H_t = (
            g1 * (phase_p_delta * JC_up + phase_m_delta * JC_down)
            + g2 * (phase_m_sigma * CR_down + phase_p_sigma * CR_up)
        )
        return H_t

    else:
        raise ValueError(
            "Invalid picture. Choose from 'interaction', 'atom', 'full', 'rabi2', 'rabi2_ip'."
        )


def loss_ode(H_, nn_state, tempo, time_scale: float = 1.0):
    assert tempo.requires_grad, "tempo must have requires_grad=True"

    state = nn_state  # (N, D) complex
    N, D = state.shape

    # Compute i * time_scale * (psi H)  (row-vector convention)
    if H_.dim() == 2:
        H_psi = 1j * float(time_scale) * (state @ H_)  # (N, D)
    elif H_.dim() == 3:
        # H_: (N,D,D) -> batch matmul
        H_psi = 1j * float(time_scale) * torch.bmm(state.unsqueeze(1), H_).squeeze(1)
    else:
        raise ValueError("H_ must have shape (D,D) or (N,D,D)")

    eye = torch.eye(D, device=state.device, dtype=state.dtype)  # (D, D) complex

    grad_out_real = eye[:, None, :].expand(D, N, D)             # (D, N, D)
    grad_out_imag = (1j * eye)[:, None, :].expand(D, N, D)      # (D, N, D)
    grad_out = torch.cat([grad_out_real, grad_out_imag], dim=0) # (2D, N, D)

    grads = torch.autograd.grad(
        outputs=state,
        inputs=tempo,
        grad_outputs=grad_out,
        is_grads_batched=True,
        retain_graph=True,
        create_graph=True,
    )[0]  # (2D, N, 1)

    grads = grads.squeeze(-1)          # (2D, N)
    du_dt = grads[:D].permute(1, 0)    # (N, D)
    dv_dt = grads[D:].permute(1, 0)    # (N, D)
    dpsi_dt = du_dt + 1j * dv_dt       # (N, D)

    residual = dpsi_dt + H_psi
    return residual.abs().mean(dim=0).sum()


def loss_ic(nn_state, sim_state):
    loss = mae(nn_state[0], sim_state[0])
    # loss = torch.mean(torch.log(torch.cosh(nn_state[0] - sim_state[0])))
    # loss = mse(nn_state[0], sim_state[0])
    return loss


def loss_norm(nn_state):
    squared_sum = (abs(nn_state) ** 2).sum(-1)
    loss = mae(squared_sum, 1)
    # loss = torch.mean(torch.log(torch.cosh(squared_sum - 1)))
    # loss = mse(squared_sum, 1)
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
        # loss = mae(expected_values_nn, sim_expect[0])
        loss = torch.mean(torch.log(torch.cosh(expected_values_nn - sim_expect[0])))
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
    # loss = torch.mean(torch.log(torch.cosh(expected_values_nn - selected_sim)))
    # loss = mse(expected_values_nn, selected_sim)
    return loss







########################## IMPLEMENTACAO ANTIGA ##########################


# def hamiltonian_with_params(
#     picture: str, params: dict, coupling_strength: list, dims: dict
# ) -> torch.Tensor:
#     """
#     Returns the Hamiltonian based on the specified picture and parameters.
#     """

#     a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
#     sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))
#     sz = qutip.tensor(qutip.sigmaz(), qutip.qeye(dims["field"]))
#     sx = qutip.tensor(qutip.sigmax(), qutip.qeye(dims["field"]))
    
#     a_dag = a.dag()
#     sm_dag = sm.dag()

#     a = torch.tensor(a.full(), dtype=torch.complex64, device=global_variables.DEVICE)
#     sm = torch.tensor(sm.full(), dtype=torch.complex64, device=global_variables.DEVICE)
#     a_dag = torch.tensor(
#         a_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE
#     )
#     sm_dag = torch.tensor(
#         sm_dag.full(), dtype=torch.complex64, device=global_variables.DEVICE
#     )
#     sz = torch.tensor(sz.full(), dtype=torch.complex64, device=global_variables.DEVICE)
#     sx = torch.tensor(sx.full(), dtype=torch.complex64, device=global_variables.DEVICE)

#     if picture == "interaction":
#         hamiltonian = coupling_strength * (a_dag @ sm + a @ sm_dag)

#     elif picture == "atom":
#         hamiltonian = 0.5 * abs(
#             params["wc"] - params["wa"]
#         ) / 2 * a_dag @ a + coupling_strength * (a_dag @ sm + a @ sm_dag)

#     elif picture == "full":
#         hamiltonian = (
#             params["wc"] * a_dag @ a
#             + 0.5 * params["wa"] * sz
#             + coupling_strength * (a_dag @ sm + a @ sm_dag)
#         )

#     elif picture == "rabi":
#         hamiltonian = (
#             params["wc"] * a_dag @ a
#             + params["wa"] * sm_dag @ sm
#             + coupling_strength[0] * a_dag @ sm
#             + coupling_strength[1] * a @ sm_dag
#             + coupling_strength[2] * a @ sm
#             + coupling_strength[3] * a_dag @ sm_dag
#         )

#     elif picture == "rabi2":
#         hamiltonian = (
#             params["wc"] * a_dag @ a
#             + params["wa"] * sm_dag @ sm
#             + coupling_strength[0] * (a_dag @ sm + a @ sm_dag)
#             + coupling_strength[1] * (a @ sm + a_dag @ sm_dag)
#         )

#     elif picture == "geral":
#         hamiltonian = (
#             params["wc"] * a_dag @ a
#             + params["wa"] * sm_dag @ sm
#             + coupling_strength[0] * (a_dag * sm + a @ sm_dag)  # jaynes-cummings
#             + coupling_strength[1] * (a * sm + a_dag @ sm_dag)  # rabi
#             + coupling_strength[2] * (a + a_dag) ** 2 @ sx  # two-photon
#             + coupling_strength[3] * sx  # classic field
#         )

#     else:
#         raise ValueError(
#             "Invalid picture. Choose from 'interaction', 'atom', 'full', 'rabi', 'rabi2', or 'geral'."
#         )

#     return hamiltonian


# def loss_ic(nn_state, sim_state):
#     loss = mae(nn_state[0], sim_state[0])
#     # loss = torch.mean(torch.log(torch.cosh(nn_state[0] - sim_state[0])))
#     # loss = mse(nn_state[0], sim_state[0])
#     return loss


# def loss_norm(nn_state):
#     squared_sum = (abs(nn_state) ** 2).sum(-1)
#     loss = mae(squared_sum, 1)
#     # loss = torch.mean(torch.log(torch.cosh(squared_sum - 1)))
#     # loss = mse(squared_sum, 1)
#     return loss


# def loss_data(nn_state, operators, sim_expect, n_points=None):
#     """
#     Computes the loss using n_points randomly selected from the training set.
#     If n_points is None or 1, uses only the first instant (as in the original loss_data).
#     operators: list or tensor of shape (n_observables, D, D)
#     sim_expect: shape (n_time_steps, n_observables)
#     """
#     if n_points is None or n_points == 1:
#         # Use only the first instant for all observables
#         expected_values_nn = []
#         for op in operators:
#             ev = (torch.conj(nn_state[0]) @ (op @ nn_state[0])).real
#             expected_values_nn.append(ev)
#         expected_values_nn = torch.stack(expected_values_nn)
#         # loss = mae(expected_values_nn, sim_expect[0])
#         loss = torch.mean(torch.log(torch.cosh(expected_values_nn - sim_expect[0])))
#         return loss

#     torch.manual_seed(global_variables.SEED)

#     n_samples = nn_state.shape[0]
#     indices = torch.randperm(n_samples)[:n_points]

#     selected_nn = nn_state[indices]  # (n_points, D)
#     selected_sim = sim_expect[indices]  # (n_points, n_observables)

#     expected_values_nn = []
#     for op in operators:
#         # For each observable, compute <psi|O|psi> for all selected states
#         ev = torch.einsum("ni,ij,nj->n", selected_nn.conj(), op, selected_nn).real
#         expected_values_nn.append(ev)
#     expected_values_nn = torch.stack(
#         expected_values_nn, dim=1
#     )  # (n_points, n_observables)

#     loss = mae(expected_values_nn, selected_sim)
#     # loss = torch.mean(torch.log(torch.cosh(expected_values_nn - selected_sim)))
#     # loss = mse(expected_values_nn, selected_sim)
#     return loss


# # def loss_ode(H_, nn_state, tempo):
    
# #     # print(H_.shape)
# #     # print(nn_state.shape)
# #     # print(tempo.shape)
    
# #     # print(type(H_))
# #     # print(type(nn_state))
# #     # print(type(tempo))
    
# #     nn_real = nn_state.real
# #     nn_imag = nn_state.imag
# #     n_time_steps = tempo.shape[0]
# #     state = nn_real + 1j * nn_imag
# #     H_psi = 1j * (state @ H_)

# #     # Calculando o gradiente de dpsi_dt separando a parte real e imagina
# #     loss = 0
# #     for i in range(state.shape[-1]):
# #         dpsi_dt_real = torch.autograd.grad(
# #             outputs=nn_real[:, i],
# #             inputs=tempo,
# #             grad_outputs=torch.ones_like(nn_real[:, i]),
# #             retain_graph=True,
# #             create_graph=True,
# #         )[0]

# #         dpsi_dt_imag = torch.autograd.grad(
# #             outputs=nn_imag[:, i],
# #             inputs=tempo,
# #             grad_outputs=torch.ones_like(nn_imag[:, i]),
# #             retain_graph=True,
# #             create_graph=True,
# #         )[0]

# #         dpsi_dt = dpsi_dt_real + 1j * dpsi_dt_imag

# #         loss += torch.mean(abs(dpsi_dt + H_psi[:, i].reshape((n_time_steps, 1))))

# #     return loss


# def loss_ode(H_, nn_state, tempo):
#     assert tempo.requires_grad, "tempo must have requires_grad=True"

#     state = nn_state  # (N, D) complex
#     N, D = state.shape

#     H_psi = 1j * (state @ H_)  # (N, D)

#     eye = torch.eye(D, device=state.device, dtype=state.dtype)  # (D, D) complex

#     # v = e_i  -> d/dt Re(psi_i)
#     grad_out_real = eye[:, None, :].expand(D, N, D)             # (D, N, D)

#     # v = 1j e_i -> d/dt Im(psi_i)
#     grad_out_imag = (1j * eye)[:, None, :].expand(D, N, D)      # (D, N, D)

#     grad_out = torch.cat([grad_out_real, grad_out_imag], dim=0) # (2D, N, D)

#     grads = torch.autograd.grad(
#         outputs=state,
#         inputs=tempo,
#         grad_outputs=grad_out,
#         is_grads_batched=True,
#         retain_graph=True,
#         create_graph=True,
#     )[0]  # (2D, N, 1), real dtype (same as tempo)

#     grads = grads.squeeze(-1)  # (2D, N)

#     du_dt = grads[:D].permute(1, 0)  # (N, D)
#     dv_dt = grads[D:].permute(1, 0)  # (N, D)

#     dpsi_dt = du_dt + 1j * dv_dt     # (N, D)

#     residual = dpsi_dt + H_psi
#     loss = residual.abs().mean(dim=0).sum()
#     return loss
