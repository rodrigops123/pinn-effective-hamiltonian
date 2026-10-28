"""PINN loss terms.

Conventions: a predicted trajectory ``nn_state`` has shape (N, D) and dtype complex,
row n being ψ(t_n) in the |atom> ⊗ |n> basis; operators act as H ψ (column vectors).
"""

import torch


def mae(y_pred, y_true):
    """Mean absolute error (complex inputs: mean modulus of the difference)."""
    return torch.mean(abs(y_pred - y_true))


def loss_ode(H_, nn_state, tempo, time_scale: float = 1.0):
    """Residual of the Schrödinger equation dψ/dτ = -i T H ψ, with τ = t / T.

    Parameters
    ----------
    H_ : torch.Tensor
        (D, D) or, for time-dependent pictures, (N, D, D) complex Hamiltonian.
    nn_state : torch.Tensor
        (N, D) complex network output, row n being ψ(τ_n).
    tempo : torch.Tensor
        (N, 1) network input τ, with ``requires_grad``.
    time_scale : float
        T; 1 when the network sees physical time.

    Returns
    -------
    torch.Tensor
        Scalar: per-component mean over time of |dψ/dτ + i T H ψ|, summed over D.
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
    """MAE between the predicted and the exact state at the first time point."""
    loss = mae(nn_state[0], sim_state[0])
    return loss


def loss_norm(nn_state):
    """MAE of the squared norm ||ψ(t)||² from 1, over all time points."""
    squared_sum = (abs(nn_state) ** 2).sum(-1)
    loss = mae(squared_sum, 1)
    return loss


def loss_data(nn_state, operators, sim_expect, indices=None):
    """MAE between predicted and simulated expectation values.

    Parameters
    ----------
    nn_state : torch.Tensor
        (N, D) complex predicted states.
    operators : torch.Tensor
        (K, D, D) observables.
    sim_expect : torch.Tensor
        (N, K) exact expectation values.
    indices : torch.Tensor, optional
        Time indices to compare at; all of them when None.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if indices is not None:
        nn_state = nn_state[indices]
        sim_expect = sim_expect[indices]
    predicted = torch.einsum("ni,kij,nj->nk", nn_state.conj(), operators, nn_state).real
    return mae(predicted, sim_expect)
