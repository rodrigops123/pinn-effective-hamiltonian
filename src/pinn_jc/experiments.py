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
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from pinn_jc.io.runs import environment_info, git_info, save_run, write_yaml
from pinn_jc.training.config import ExperimentConfig
from pinn_jc.training.trainer import train

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
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
        data = dict(data)
        meta = data.pop("experiment")
        return cls(
            name=meta["name"],
            description=meta.get("description", ""),
            seeds=meta["seeds"],
            config=ExperimentConfig.from_dict(data),
        )

    def with_overrides(self, epochs: int | None = None, seeds=None) -> "Experiment":
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
    return Experiment.from_dict(yaml.safe_load(Path(path).read_text()))


class DirtyWorkingTreeError(RuntimeError):
    pass


def run_experiment(
    experiment: Experiment,
    output_root: Path | str = "saved_artifacts/runs",
    allow_dirty: bool = False,
    progress_bar: bool = True,
) -> Path:
    """Train every seed of ``experiment`` and save the results; returns the output dir.

    Refuses to run with uncommitted changes to tracked files unless ``allow_dirty``,
    so that every stored result maps to a commit.
    """
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
    summaries = {}
    try:
        for seed in experiment.seeds:
            logger.info("%s: seed %d", experiment.name, seed)
            training = replace(
                experiment.config.training, seed=seed, progress_bar=progress_bar
            )
            config = replace(experiment.config, training=training)
            start = time.perf_counter()
            result = train(config)
            save_run(result, out_dir / f"seed_{seed}", time.perf_counter() - start)
            summaries[seed] = json.loads(
                (out_dir / f"seed_{seed}" / "metrics.json").read_text()
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
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summarize(summaries), indent=2) + "\n"
    )
    return out_dir


def summarize(metrics_by_seed: dict[int, dict]) -> dict:
    summary = {"seeds": {str(seed): m for seed, m in metrics_by_seed.items()}}
    couplings = [m["couplings"] for m in metrics_by_seed.values() if "couplings" in m]
    if couplings:
        couplings = np.array(couplings)
        summary["couplings_mean"] = couplings.mean(axis=0).tolist()
        summary["couplings_std"] = couplings.std(axis=0, ddof=0).tolist()
    return summary
