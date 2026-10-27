"""Shared fixtures.

Golden files live in ``tests/golden``. Regenerate them only on purpose, e.g. after
an intentional behaviour change:

    UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN") == "1"


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture
def golden():
    """Compare a dict of arrays against ``tests/golden/<name>.npz`` (or rewrite it)."""

    def check(name: str, arrays: dict, rtol: float = 1e-5, atol: float = 1e-6) -> None:
        path = GOLDEN_DIR / f"{name}.npz"
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        if UPDATE_GOLDEN:
            GOLDEN_DIR.mkdir(exist_ok=True)
            np.savez_compressed(path, **arrays)
            return
        assert path.exists(), f"missing golden file {path}; run with UPDATE_GOLDEN=1"
        reference = np.load(path)
        assert set(reference.files) == set(arrays), "golden keys changed"
        for key in reference.files:
            np.testing.assert_allclose(
                arrays[key],
                reference[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"{name}:{key}",
            )

    return check
