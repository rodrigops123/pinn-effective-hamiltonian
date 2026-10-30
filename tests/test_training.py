"""Training loop: regression tests and config-driven features."""

import logging

import numpy as np
import pytest
import torch

from pinn_jc.training.config import ExperimentConfig
from pinn_jc.training.losses import loss_data, loss_ode, state_and_time_derivative
from pinn_jc.training.trainer import (
    build_models,
    data_indices,
    noisy_expectations,
    train,
)


def _config(**overrides):
    sections = {
        "physics": {
            "picture": "atom",
            "params": {"wc": 1.0, "wa": 1.0, "g1": 1.0},
            "init_state": "fock",
        },
        "data": {"tfinal": 2 * np.pi, "n_time_steps": 30, "n_data_points": 30},
        "training": {"epochs": 6, "log_every": 2, "progress_bar": False},
        "model": {},
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


def _batched_backward_time_derivative(psi, time):
    """dψ/dτ by 2D batched reverse-mode passes, as loss_ode computed it before."""
    n, d = psi.shape
    eye = torch.eye(d, dtype=psi.dtype)
    grad_outputs = torch.cat(
        [eye[:, None, :].expand(d, n, d), (1j * eye)[:, None, :].expand(d, n, d)]
    )
    grads = torch.autograd.grad(
        psi, time, grad_outputs, is_grads_batched=True, create_graph=True
    )[0].squeeze(-1)
    return (grads[:d] + 1j * grads[d:]).T


@pytest.mark.parametrize("dim", [4, 20])
def test_forward_mode_time_derivative_matches_batched_backward(dim):
    """The one-pass derivative is the one the 2D backward passes gave, and so are the
    parameter gradients of loss_ode."""
    torch.manual_seed(0)
    state, _ = build_models(dim)
    time = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
    hamiltonian = torch.randn(dim, dim, dtype=torch.complex64)
    hamiltonian = hamiltonian + hamiltonian.mH
    parameters = list(state.parameters())

    psi, dpsi_dt = state_and_time_derivative(state, time)
    reference = _batched_backward_time_derivative(state(time), time)
    gradients = torch.autograd.grad(loss_ode(hamiltonian, psi, dpsi_dt), parameters)
    reference_gradients = torch.autograd.grad(
        loss_ode(hamiltonian, state(time), reference), parameters
    )

    torch.testing.assert_close(dpsi_dt, reference)
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


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


def test_hard_initial_condition_holds_exactly_without_loss_ic():
    result = train(
        _config(model={"hard_initial_condition": True}, training={"seed": 0})
    )

    with torch.no_grad():
        psi = result.state(torch.zeros(1, 1))[0]
    expected = torch.zeros(4, dtype=torch.complex64)
    expected[2] = 1  # fock: |e, 0>, the third basis state of the 2 x 2 space
    torch.testing.assert_close(psi, expected)
    assert max(result.history.loss_ic) < 1e-6


@pytest.mark.parametrize("schedule", ["cosine", "exponential"])
def test_learning_rate_schedules_end_at_the_final_rate(schedule, monkeypatch):
    rates = []
    original_step = torch.optim.Adam.step

    def recording_step(self, *args, **kwargs):
        rates.append(self.param_groups[0]["lr"])
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", recording_step)
    train(
        _config(
            training={
                "epochs": 50,
                "lr_schedule": schedule,
                "learning_rate": 1e-2,
                "final_learning_rate": 1e-4,
            }
        )
    )

    assert rates[0] == pytest.approx(1e-2)
    assert rates[-1] == pytest.approx(1e-4, rel=0.2)
    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False))


def test_time_window_grows_to_the_whole_grid(monkeypatch):
    import pinn_jc.training.trainer as trainer

    sizes = []
    original = trainer.state_and_time_derivative

    def recording(state, time):
        sizes.append(len(time))
        return original(state, time)

    monkeypatch.setattr(trainer, "state_and_time_derivative", recording)
    train(_config(training={"epochs": 12, "window_start": 0.2, "window_epochs": 10}))

    assert sizes[0] == 6  # 20% of 30 points
    assert sizes == sorted(sizes)
    assert sizes[-2:] == [30, 30]  # full grid from epoch window_epochs on


def test_fixed_and_initialised_couplings_in_training():
    config = _config(
        physics={
            "picture": "rabi2_ip",
            "params": {"wc": 5.0, "wa": 5.0, "g1": 1.0, "g2": 1.0},
            "dims": {"atom": 2, "field": 3},
            "fixed_couplings": {"g2": 0.25},
            "coupling_init": {"g1": 0.5},
        }
    )

    couplings = np.array(train(config).history.couplings)

    np.testing.assert_allclose(couplings[:, 1], 0.25)
    assert couplings[0, 0] == pytest.approx(0.5, abs=0.02)  # after one Adam step
    assert couplings[-1, 0] != couplings[0, 0]


def test_frozen_couplings_start_learning_after_the_freeze():
    config = _config(
        physics={
            "picture": "rabi2_ip",
            "params": {"wc": 5.0, "wa": 5.0, "g1": 1.0, "g2": 1.0},
            "dims": {"atom": 2, "field": 3},
            "coupling_init": {"g2": 0.05},
        },
        training={"freeze_couplings_epochs": 3, "seed": 0},
    )

    result = train(config)
    couplings = np.array(result.history.couplings)

    np.testing.assert_array_equal(couplings[:4], np.repeat(couplings[:1], 4, axis=0))
    assert couplings[0, 1] == pytest.approx(0.05)
    assert not np.array_equal(couplings[-1], couplings[3])


def test_noisy_expectations_are_reproducible_and_leave_the_global_stream():
    exact = torch.zeros(2000, 2, dtype=torch.float64)
    torch.manual_seed(0)
    expected_draw = torch.rand(1)

    torch.manual_seed(0)
    noisy = noisy_expectations(exact, 0.01, seed=3)
    draw = torch.rand(1)

    assert noisy_expectations(exact, 0.0) is exact
    torch.testing.assert_close(noisy, noisy_expectations(exact, 0.01, seed=3))
    assert not torch.equal(noisy, noisy_expectations(exact, 0.01, seed=4))
    assert noisy.std().item() == pytest.approx(0.01, rel=0.05)
    torch.testing.assert_close(draw, expected_draw)


def test_noise_changes_the_fit_but_not_the_initialisation():
    clean = train(_config(training={"seed": 0})).history
    noisy = train(_config(data={"noise_std": 0.05}, training={"seed": 0})).history

    assert noisy.loss_ic[0] == pytest.approx(clean.loss_ic[0])
    assert noisy.loss_data[0] != pytest.approx(clean.loss_data[0])
