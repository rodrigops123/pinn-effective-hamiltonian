"""Known non-physics bugs, documented as strict xfails until they are fixed in Phase 3."""

import numpy as np
import pytest
import torch

from conftest import seed_everything
from pinn_jc.training.losses import loss_data
from pinn_jc.training.trainer import train_with_parameter


@pytest.mark.xfail(
    strict=True,
    raises=NameError,
    reason="Bug C: is_scaled=True references the undefined name time_train. Fix in Phase 3.",
)
def test_train_with_scaled_time_runs():
    seed_everything(0)
    train_with_parameter(
        epochs=1,
        params={"wc": 1.0, "wa": 1.0, "g1": 1.0},
        tfinal=2 * np.pi,
        n_time_steps=20,
        init_state="fock",
        picture="atom",
        dims={"atom": 2, "field": 2},
        n_points_loss=20,
        n_parameter=1,
        is_scaled=True,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Bug D: loss_data calls torch.manual_seed on every call, resetting the global "
    "RNG. Fix in Phase 3.",
)
def test_loss_data_leaves_global_rng_untouched():
    state = torch.randn(10, 4, dtype=torch.complex64)
    operators = torch.eye(4, dtype=torch.complex64)[None]
    expect = torch.ones(10, 1)

    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    loss_data(state, operators, expect, n_points=5)

    assert torch.equal(torch.rand(1), expected_next)
