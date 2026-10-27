import torch

from pinn_jc import settings


def mae(y_pred, y_true):
    return torch.mean(abs(y_pred - y_true))


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

    grad_out_real = eye[:, None, :].expand(D, N, D)  # (D, N, D)
    grad_out_imag = (1j * eye)[:, None, :].expand(D, N, D)  # (D, N, D)
    grad_out = torch.cat([grad_out_real, grad_out_imag], dim=0)  # (2D, N, D)

    grads = torch.autograd.grad(
        outputs=state,
        inputs=tempo,
        grad_outputs=grad_out,
        is_grads_batched=True,
        retain_graph=True,
        create_graph=True,
    )[0]  # (2D, N, 1)

    grads = grads.squeeze(-1)  # (2D, N)
    du_dt = grads[:D].permute(1, 0)  # (N, D)
    dv_dt = grads[D:].permute(1, 0)  # (N, D)
    dpsi_dt = du_dt + 1j * dv_dt  # (N, D)

    residual = dpsi_dt + H_psi
    return residual.abs().mean(dim=0).sum()


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
        loss = torch.mean(torch.log(torch.cosh(expected_values_nn - sim_expect[0])))
        return loss

    torch.manual_seed(settings.SEED)

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
