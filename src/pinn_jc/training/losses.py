import torch


def mae(y_pred, y_true):
    return torch.mean(abs(y_pred - y_true))


def loss_ode(H_, nn_state, tempo, time_scale: float = 1.0):
    """Residual of the Schrödinger equation dψ/dτ = -i T H ψ, with τ = t / T.

    nn_state: (N, D) complex, row n is ψ(τ_n); tempo: (N, 1) network input τ;
    H_: (D, D) or (N, D, D); time_scale: T (1 when the network sees physical time).
    """
    assert tempo.requires_grad, "tempo must have requires_grad=True"

    state = nn_state
    N, D = state.shape

    if H_.dim() == 2:
        h_psi = torch.einsum("ij,nj->ni", H_, state)
    elif H_.dim() == 3:
        h_psi = torch.einsum("nij,nj->ni", H_, state)
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

    residual = dpsi_dt + 1j * float(time_scale) * h_psi
    return residual.abs().mean(dim=0).sum()


def loss_ic(nn_state, sim_state):
    loss = mae(nn_state[0], sim_state[0])
    return loss


def loss_norm(nn_state):
    squared_sum = (abs(nn_state) ** 2).sum(-1)
    loss = mae(squared_sum, 1)
    return loss


def loss_data(nn_state, operators, sim_expect, indices=None):
    """MAE between predicted and simulated expectation values.

    operators: (n_observables, D, D); sim_expect: (N, n_observables);
    indices: time indices to compare at (all of them when None).
    """
    if indices is not None:
        nn_state = nn_state[indices]
        sim_expect = sim_expect[indices]
    predicted = torch.einsum("ni,kij,nj->nk", nn_state.conj(), operators, nn_state).real
    return mae(predicted, sim_expect)
