"""Experiments: a named ExperimentConfig trained once per seed, stored as YAML.

An experiment file is an ExperimentConfig plus an ``experiment`` section::

    experiment:
      name: atom_fock
      description: ...
      seeds: [42, 43, 44]
    physics: {...}
    data: {...}
    model: {...}
    training: {...}     # training.seed is set per run from experiment.seeds

``run_experiment`` writes::

    <output_root>/<name>/<timestamp>_<commit>/
        experiment.yaml   the experiment as run (overrides applied)
        meta.json         git commit and dirty flag, library versions, command, times
        train.log         training log of every seed
        summary.json      final metrics of every seed, mean and std of the couplings
        seed_<s>/         one run directory per seed (see pinn_jc.io.runs)

With ``workers`` > 1 the seeds train in parallel processes, one torch thread each;
``train.log`` still lists the seeds in order.
"""

import io
import json
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from pinn_jc.io.runs import environment_info, git_info, save_run, write_yaml
from pinn_jc.training.config import ExperimentConfig
from pinn_jc.training.trainer import train

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    """A named ``ExperimentConfig`` with its description and seeds."""

    name: str
    description: str
    seeds: tuple[int, ...]
    config: ExperimentConfig

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                f"seeds must be a non-empty list of distinct ints: {self.seeds}"
            )
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"name must be a filesystem-safe identifier, got {self.name!r}"
            )

    def to_dict(self) -> dict:
        """Return the experiment in the YAML layout (``experiment`` section first)."""
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "seeds": list(self.seeds),
            },
            **self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Build and validate an experiment from the YAML layout."""
        data = dict(data)
        meta = data.pop("experiment")
        return cls(
            name=meta["name"],
            description=meta.get("description", ""),
            seeds=meta["seeds"],
            config=ExperimentConfig.from_dict(data),
        )

    def with_overrides(self, epochs: int | None = None, seeds=None) -> "Experiment":
        """Return a copy with a different number of epochs and/or seeds."""
        experiment = self
        if epochs is not None:
            training = replace(self.config.training, epochs=epochs)
            experiment = replace(
                experiment, config=replace(self.config, training=training)
            )
        if seeds is not None:
            experiment = replace(experiment, seeds=tuple(seeds))
        return experiment


def load_experiment(path: Path | str) -> Experiment:
    """Read an experiment YAML file."""
    return Experiment.from_dict(yaml.safe_load(Path(path).read_text()))


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a run would not map to a commit (uncommitted tracked changes)."""


def run_experiment(
    experiment: Experiment,
    output_root: Path | str = "saved_artifacts/runs",
    allow_dirty: bool = False,
    progress_bar: bool = True,
    workers: int = 1,
) -> Path:
    """Train every seed of ``experiment`` and save the results; returns the output dir.

    Refuses to run with uncommitted changes to tracked files unless ``allow_dirty``,
    so that every stored result maps to a commit. With ``workers`` > 1, up to that
    many seeds train at once, each in its own process with one torch thread and no
    progress bar.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    git = git_info(Path(__file__).parent)
    if git["dirty"] and not allow_dirty:
        raise DirtyWorkingTreeError(
            "uncommitted changes to tracked files: commit them, or pass allow_dirty=True "
            "(--allow-dirty) for runs that do not need to be reproducible"
        )

    started = datetime.now(UTC)
    commit = (git["commit"] or "nogit")[:7]
    out_dir = Path(output_root) / experiment.name / f"{started:%Y%m%dT%H%M%SZ}_{commit}"
    out_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(out_dir / "experiment.yaml", experiment.to_dict())

    # The run log is always written at INFO, whatever the caller's logging setup.
    handler = logging.FileHandler(out_dir / "train.log")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    package_logger = logging.getLogger("pinn_jc")
    previous_level = package_logger.level
    if package_logger.getEffectiveLevel() > logging.INFO:
        package_logger.setLevel(logging.INFO)
    package_logger.addHandler(handler)
    configs = {
        seed: replace(
            experiment.config,
            training=replace(
                experiment.config.training,
                seed=seed,
                progress_bar=progress_bar and workers == 1,
            ),
        )
        for seed in experiment.seeds
    }
    summaries = {}
    try:
        if workers == 1:
            for seed, config in configs.items():
                summaries[seed] = _train_seed(experiment.name, config, out_dir)
        else:
            summaries = _train_seeds_in_parallel(
                experiment.name, configs, out_dir, workers, handler
            )
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)
        handler.close()

    meta = {
        "experiment": experiment.name,
        "git": git,
        "environment": environment_info(),
        "command": sys.argv,
        "workers": workers,
        "torch_threads": 1 if workers > 1 else torch.get_num_threads(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summarize(summaries), indent=2) + "\n"
    )
    return out_dir


def _train_seed(name: str, config: ExperimentConfig, out_dir: Path) -> dict:
    """Train one seed, save its run directory and return its final metrics."""
    seed = config.training.seed
    logger.info("%s: seed %d", name, seed)
    start = time.perf_counter()
    result = train(config)
    save_run(result, out_dir / f"seed_{seed}", time.perf_counter() - start)
    return json.loads((out_dir / f"seed_{seed}" / "metrics.json").read_text())


def _init_worker() -> None:
    torch.set_num_threads(1)


def _train_seed_in_worker(name: str, config: ExperimentConfig, out_dir: Path):
    """``_train_seed`` in a worker process; also returns the seed's log text."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    package_logger = logging.getLogger("pinn_jc")
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(handler)
    try:
        metrics = _train_seed(name, config, out_dir)
    finally:
        package_logger.removeHandler(handler)
    return metrics, stream.getvalue()


def _train_seeds_in_parallel(name, configs, out_dir, workers, log_handler) -> dict:
    """Train the seeds in ``workers`` processes; logs are written in seed order."""
    summaries = {}
    pool = ProcessPoolExecutor(
        max_workers=min(workers, len(configs)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )
    try:
        futures = {
            seed: pool.submit(_train_seed_in_worker, name, config, out_dir)
            for seed, config in configs.items()
        }
        for seed, future in futures.items():
            summaries[seed], log_text = future.result()
            log_handler.stream.write(log_text)
            log_handler.flush()
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    pool.shutdown()
    return summaries


def summarize(metrics_by_seed: dict[int, dict]) -> dict:
    """Collect per-seed final metrics, with the mean and std of the couplings."""
    summary = {"seeds": {str(seed): m for seed, m in metrics_by_seed.items()}}
    couplings = [m["couplings"] for m in metrics_by_seed.values() if "couplings" in m]
    if couplings:
        couplings = np.array(couplings)
        summary["couplings_mean"] = couplings.mean(axis=0).tolist()
        summary["couplings_std"] = couplings.std(axis=0, ddof=0).tolist()
    return summary
