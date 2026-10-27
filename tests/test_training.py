"""Training loop: regression tests for bugs C-E of the Phase 0 audit and the
config-driven features."""

import logging

import numpy as np
import torch

from pinn_jc.training.config import ExperimentConfig
from pinn_jc.training.losses import loss_data
from pinn_jc.training.trainer import data_indices, train


def _config(**overrides):
    sections = {
        "physics": {
            "picture": "atom",
            "params": {"wc": 1.0, "wa": 1.0, "g1": 1.0},
            "init_state": "fock",
        },
        "data": {"tfinal": 2 * np.pi, "n_time_steps": 30, "n_data_points": 30},
        "training": {"epochs": 6, "log_every": 2, "progress_bar": False},
    }
    for section, values in overrides.items():
        sections[section] = {**sections[section], **values}
    return ExperimentConfig.from_dict(sections)


def test_training_with_scaled_time_runs_and_stays_finite():
    """Bug C: is_scaled=True referenced an undefined variable."""
    history = train(_config(data={"scale_time": True}, training={"seed": 0})).history

    assert all(np.isfinite(history.total_loss))


def test_training_does_not_reset_the_global_rng():
    """Bug D: loss_data reseeded the global RNG on every call, so any two runs ended in
    the same RNG state regardless of their seed."""
    final_draws = []
    for seed in (0, 1):
        train(_config(data={"n_data_points": 10}, training={"seed": seed}))
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


def test_validation_window_is_held_out_and_evaluated():
    result = train(_config(data={"validation_fraction": 0.2}))
    history = result.history

    assert history.validation_epochs == [2, 4, 6]
    assert all(0 <= f <= 1 for f in history.validation_fidelity)
    assert len(history.validation_expect_mae) == 3
    assert len(history.total_loss) == 6


def test_no_validation_without_a_held_out_window():
    history = train(_config()).history

    assert history.validation_epochs == []


def test_total_loss_is_the_weighted_sum_of_the_terms():
    weights = {"ic": 2.0, "norm": 0.5, "data": 3.0, "ode": 0.0}
    history = train(_config(training={"loss_weights": weights})).history

    for epoch in range(len(history.total_loss)):
        expected = (
            2.0 * history.loss_ic[epoch]
            + 0.5 * history.loss_norm[epoch]
            + 3.0 * history.loss_data[epoch]
        )
        np.testing.assert_allclose(history.total_loss[epoch], expected, rtol=1e-6)


def test_a_seeded_config_is_reproducible():
    first = train(_config(training={"seed": 11})).history
    second = train(_config(training={"seed": 11})).history

    assert first.total_loss == second.total_loss
    assert first.couplings == second.couplings


def test_fixed_couplings_learn_only_the_state():
    result = train(_config(physics={"learn_couplings": False}))

    assert result.couplings is None
    assert result.history.couplings == []
    assert "learned_param" not in result.history.to_legacy_dict()


def test_training_progress_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="pinn_jc.training.trainer"):
        train(_config(data={"validation_fraction": 0.2}))

    messages = [record.getMessage() for record in caplog.records]
    assert sum("epoch" in m for m in messages) == 3
    assert sum("validation: fidelity" in m for m in messages) == 3


def test_history_exports_arrays_for_saving():
    arrays = train(_config()).history.to_arrays()

    assert arrays["total_loss"].shape == (6,)
    assert arrays["couplings"].shape == (6, 1)
