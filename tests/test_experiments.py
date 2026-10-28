import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import pinn_jc.experiments as experiments
from pinn_jc.experiments import (
    DirtyWorkingTreeError,
    Experiment,
    load_experiment,
    run_experiment,
)
from pinn_jc.io.runs import load_run

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_config_is_a_valid_experiment(path):
    experiment = load_experiment(path)

    assert experiment.name == path.stem
    assert Experiment.from_dict(experiment.to_dict()) == experiment


def _tiny_experiment():
    experiment = load_experiment(CONFIGS[0])
    return experiment.with_overrides(epochs=3, seeds=[5, 6])


def test_run_experiment_writes_a_complete_run_directory(tmp_path):
    out = run_experiment(
        _tiny_experiment(), tmp_path, allow_dirty=True, progress_bar=False
    )

    assert {p.name for p in out.iterdir()} == {
        "experiment.yaml",
        "meta.json",
        "summary.json",
        "train.log",
        "seed_5",
        "seed_6",
    }
    for seed in (5, 6):
        files = {p.name for p in (out / f"seed_{seed}").iterdir()}
        assert files == {"config.yaml", "model.pt", "history.npz", "metrics.json"}
        config = yaml.safe_load((out / f"seed_{seed}" / "config.yaml").read_text())
        assert config["training"]["seed"] == seed

    meta = json.loads((out / "meta.json").read_text())
    assert set(meta["git"]) == {"commit", "dirty"}
    assert "torch" in meta["environment"]["packages"]
    assert load_experiment(out / "experiment.yaml") == _tiny_experiment()
    assert "seed 5" in (out / "train.log").read_text()


def test_summary_aggregates_the_final_couplings(tmp_path):
    out = run_experiment(
        _tiny_experiment(), tmp_path, allow_dirty=True, progress_bar=False
    )
    summary = json.loads((out / "summary.json").read_text())

    finals = np.array([summary["seeds"][s]["couplings"] for s in ("5", "6")])
    np.testing.assert_allclose(summary["couplings_mean"], finals.mean(axis=0))
    np.testing.assert_allclose(summary["couplings_std"], finals.std(axis=0))


def test_load_run_restores_the_trained_models_exactly(tmp_path):
    out = run_experiment(
        _tiny_experiment(), tmp_path, allow_dirty=True, progress_bar=False
    )

    first = load_run(out / "seed_5")
    second = load_run(out / "seed_5")
    time = torch.linspace(0, 6, 25).reshape(-1, 1)

    with torch.no_grad():
        assert torch.equal(first.state(time), second.state(time))
    assert first.config.training.seed == 5
    np.testing.assert_allclose(
        first.history["couplings"][-1], first.metrics["couplings"]
    )
    assert first.history["total_loss"].shape == (3,)


def test_seeded_runs_are_reproducible(tmp_path):
    a = run_experiment(
        _tiny_experiment(), tmp_path / "a", allow_dirty=True, progress_bar=False
    )
    b = run_experiment(
        _tiny_experiment(), tmp_path / "b", allow_dirty=True, progress_bar=False
    )

    for seed in (5, 6):
        ha, hb = (
            load_run(a / f"seed_{seed}").history,
            load_run(b / f"seed_{seed}").history,
        )
        np.testing.assert_array_equal(ha["total_loss"], hb["total_loss"])


def test_dirty_working_tree_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments, "git_info", lambda path: {"commit": "abc", "dirty": True}
    )

    with pytest.raises(DirtyWorkingTreeError):
        run_experiment(_tiny_experiment(), tmp_path, progress_bar=False)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "meta, message",
    [
        ({"name": "x", "seeds": []}, "seeds"),
        ({"name": "x", "seeds": [1, 1]}, "seeds"),
        ({"name": "../escape", "seeds": [1]}, "filesystem-safe"),
    ],
)
def test_invalid_experiments_are_rejected(meta, message):
    data = load_experiment(CONFIGS[0]).to_dict()
    data["experiment"] = meta

    with pytest.raises(ValueError, match=message):
        Experiment.from_dict(data)
