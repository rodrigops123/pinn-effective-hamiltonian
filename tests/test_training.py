"""Regression tests for the training loop (bugs C, D and E of the Phase 0 audit)."""

import numpy as np
import torch

from conftest import seed_everything
from pinn_jc.training.losses import loss_data
from pinn_jc.training.trainer import data_indices, train_with_parameter

TRAINING = dict(
    params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
    tfinal=2 * np.pi,
    n_time_steps=20,
    init_state="fock",
    picture="atom",
    dims={"atom": 2, "field": 2},
    n_parameter=1,
)


def test_training_with_scaled_time_runs_and_stays_finite():
    """Bug C: is_scaled=True referenced an undefined variable."""
    seed_everything(0)
    _, history = train_with_parameter(
        epochs=3, n_points_loss=20, is_scaled=True, **TRAINING
    )

    assert all(np.isfinite(history["total_loss"]))


def test_training_does_not_reset_the_global_rng():
    """Bug D: loss_data reseeded the global RNG on every call, so any two runs ended in
    the same RNG state regardless of their seed."""
    final_draws = []
    for seed in (0, 1):
        seed_everything(seed)
        train_with_parameter(epochs=2, n_points_loss=10, is_scaled=False, **TRAINING)
        final_draws.append(torch.rand(1))

    assert not torch.equal(*final_draws)


def test_data_indices_are_fixed_and_leave_the_global_rng_untouched():
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)

    first, second = data_indices(50, 10), data_indices(50, 10)

    assert torch.equal(torch.rand(1), expected_next)
    assert torch.equal(first, second)
    assert len(first.unique()) == 10


def test_data_indices_conventions():
    assert torch.equal(data_indices(50, None), torch.tensor([0]))
    assert torch.equal(data_indices(50, 1), torch.tensor([0]))
    assert data_indices(50, 50) is None


def test_loss_data_is_the_mae_of_expectation_values():
    """Bug E: the first-instant path used log-cosh while every other path used MAE."""
    state = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.complex64)
    operators = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], dtype=torch.complex64)
    expect = torch.tensor([[0.5], [0.5]], dtype=torch.float64)

    assert loss_data(state, operators, expect, torch.tensor([0])).item() == 0.5
    np.testing.assert_allclose(loss_data(state, operators, expect).item(), 0.32)
