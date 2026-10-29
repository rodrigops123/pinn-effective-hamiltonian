"""Run directories: everything needed to reproduce, inspect and reload one training run.

Layout written by ``save_run``::

    <run_dir>/
        config.yaml     resolved ExperimentConfig (seed included)
        model.pt        {"format": 1, "state": state_dict, "couplings": state_dict | None}
        history.npz     TrainingHistory.to_arrays()
        metrics.json    final losses, couplings and validation metrics, elapsed time
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

from pinn_jc.models import ComplexStateNet, Couplings
from pinn_jc.training.config import ExperimentConfig
from pinn_jc.training.history import TrainingHistory, TrainingResult
from pinn_jc.training.trainer import build_models

CHECKPOINT_FORMAT = 1


def git_info(path: Path | str = ".") -> dict:
    """Commit and dirty flag of the repository containing ``path`` (None if not in git)."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def environment_info() -> dict:
    """Python version and versions of the packages that affect results."""
    packages = ("torch", "qutip", "numpy", "scipy", "pinn-effective-hamiltonian")
    return {
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in packages},
    }


def write_yaml(path: Path, data: dict) -> None:
    """Write ``data`` as YAML, keeping key order."""
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def final_metrics(history: TrainingHistory) -> dict:
    """Last value of every loss term, of the couplings and of the validation metrics."""
    metrics = {
        key: getattr(history, key)[-1]
        for key in ("total_loss", "loss_ic", "loss_norm", "loss_data", "loss_ode")
        if getattr(history, key)
    }
    if history.couplings:
        metrics["couplings"] = history.couplings[-1]
    if history.validation_fidelity:
        metrics["validation_fidelity"] = history.validation_fidelity[-1]
        metrics["validation_expect_mae"] = history.validation_expect_mae[-1]
    return metrics


def save_run(
    result: TrainingResult, run_dir: Path | str, elapsed_seconds: float
) -> Path:
    """Write a run directory (see module docstring); returns its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(run_dir / "config.yaml", result.config.to_dict())
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state": result.state.state_dict(),
            "couplings": None
            if result.couplings is None
            else result.couplings.state_dict(),
        },
        run_dir / "model.pt",
    )
    np.savez_compressed(run_dir / "history.npz", **result.history.to_arrays())
    metrics = {
        **final_metrics(result.history),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return run_dir


@dataclass
class LoadedRun:
    """A run read back from disk by ``load_run``."""

    config: ExperimentConfig
    state: ComplexStateNet
    couplings: Couplings | None
    history: dict[str, np.ndarray]
    metrics: dict

    @property
    def models(self) -> dict:
        """``{"state", "couplings"}``, the format the plotting functions accept."""
        return {"state": self.state, "couplings": self.couplings}


def load_run(run_dir: Path | str) -> LoadedRun:
    """Rebuild a saved run: config, trained modules (in eval mode), history, metrics."""
    run_dir = Path(run_dir)
    config = ExperimentConfig.from_dict(
        yaml.safe_load((run_dir / "config.yaml").read_text())
    )
    checkpoint = torch.load(run_dir / "model.pt", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {checkpoint.get('format')!r}")

    with torch.random.fork_rng():  # rebuilding must not consume the caller's RNG
        state, couplings = build_models(
            config.physics.hilbert_dim,
            config.physics.n_couplings,
            config.physics.positivity,
            config.model,
            *config.physics.coupling_overrides(),
            config.physics.coupling_draw,
        )
    state.load_state_dict(checkpoint["state"])
    if couplings is not None:
        couplings.load_state_dict(checkpoint["couplings"])
        couplings.eval()
    with np.load(run_dir / "history.npz") as history:
        arrays = {key: history[key] for key in history.files}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return LoadedRun(config, state.eval(), couplings, arrays, metrics)
