import json
from dataclasses import replace
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

CONFIGS = sorted((Path(__file__).parents[1] / "configs").rglob("*.yaml"))


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


def test_parallel_seeds_match_sequential_training(tmp_path):
    """Workers train each seed on one thread; the results are bit-identical.

    The sequential side also trains on one thread: torch results depend on the
    thread count on some platforms (x86), so this compares like with like.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sequential = run_experiment(
            _tiny_experiment(), tmp_path / "a", allow_dirty=True, progress_bar=False
        )
    finally:
        torch.set_num_threads(threads)
    parallel = run_experiment(
        _tiny_experiment(),
        tmp_path / "b",
        allow_dirty=True,
        progress_bar=False,
        workers=2,
    )

    for seed in (5, 6):
        a, b = (
            load_run(sequential / f"seed_{seed}"),
            load_run(parallel / f"seed_{seed}"),
        )
        for key in ("total_loss", "couplings"):
            np.testing.assert_array_equal(a.history[key], b.history[key])
    log = (parallel / "train.log").read_text()
    assert log.index("seed 5") < log.index("seed 6")
    assert "epoch 3/3" in log
    meta = json.loads((parallel / "meta.json").read_text())
    assert (meta["workers"], meta["torch_threads"]) == (2, 1)


def test_workers_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run_experiment(
            _tiny_experiment(),
            tmp_path,
            allow_dirty=True,
            progress_bar=False,
            workers=0,
        )


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


def test_hard_initial_condition_survives_save_and_load(tmp_path):
    experiment = _tiny_experiment()
    model = replace(experiment.config.model, hard_initial_condition=True)
    experiment = replace(experiment, config=replace(experiment.config, model=model))
    out = run_experiment(experiment, tmp_path, allow_dirty=True, progress_bar=False)

    run = load_run(out / "seed_5")

    with torch.no_grad():
        psi0 = run.state(torch.zeros(1, 1))[0]
    torch.testing.assert_close(psi0, run.state.psi0)
    assert run.state.psi0.abs().sum() > 0


def test_fixed_couplings_survive_save_and_load(tmp_path):
    experiment = load_experiment(
        next(p for p in CONFIGS if p.stem == "rwa_rabi_w5")
    ).with_overrides(epochs=2, seeds=[5])
    physics = replace(experiment.config.physics, fixed_couplings={"g2": 0.25})
    experiment = replace(experiment, config=replace(experiment.config, physics=physics))
    out = run_experiment(experiment, tmp_path, allow_dirty=True, progress_bar=False)

    run = load_run(out / "seed_5")

    assert run.couplings()[1].item() == pytest.approx(0.25)
