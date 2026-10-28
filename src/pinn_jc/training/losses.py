"""PINN loss terms.

Conventions: a predicted trajectory ``nn_state`` has shape (N, D) and dtype complex,
row n being ψ(t_n) in the |atom> ⊗ |n> basis; operators act as H ψ (column vectors).
"""

import torch


def mae(y_pred, y_true):
    """Mean absolute error (complex inputs: mean modulus of the difference)."""
    return torch.mean(abs(y_pred - y_true))


def state_and_time_derivative(state, time):
    """Network output ψ(τ) and its time derivative dψ/dτ, in one forward-mode pass.

    Each row of the output depends only on the same row of the input, so the
    Jacobian-vector product with a tangent of ones gives every dψ_k/dτ at once.

    Parameters
    ----------
    state : callable
        Maps times (N, 1) to complex states (N, D), e.g. a ``ComplexStateNet``.
    time : torch.Tensor
        (N, 1) network input τ.

    Returns
    -------
    tuple of torch.Tensor
        ψ and dψ/dτ, both (N, D) complex; dψ/dτ stays differentiable with respect
        to the network parameters.
    """
    return torch.func.jvp(state, (time,), (torch.ones_like(time),))


def loss_ode(H_, nn_state, dpsi_dt, time_scale: float = 1.0):
    """Residual of the Schrödinger equation dψ/dτ = -i T H ψ, with τ = t / T.

    Parameters
    ----------
    H_ : torch.Tensor
        (D, D) or, for time-dependent pictures, (N, D, D) complex Hamiltonian.
    nn_state : torch.Tensor
        (N, D) complex network output, row n being ψ(τ_n).
    dpsi_dt : torch.Tensor
        (N, D) complex dψ/dτ, from ``state_and_time_derivative``.
    time_scale : float
        T; 1 when the network sees physical time.

    Returns
    -------
    torch.Tensor
        Scalar: per-component mean over time of |dψ/dτ + i T H ψ|, summed over D.
    """
    if H_.dim() == 2:
        h_psi = torch.einsum("ij,nj->ni", H_, nn_state)
    elif H_.dim() == 3:
        h_psi = torch.einsum("nij,nj->ni", H_, nn_state)
    else:
        raise ValueError("H_ must have shape (D,D) or (N,D,D)")

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
