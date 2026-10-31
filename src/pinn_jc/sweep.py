"""The RWA sweep: per-seed start selection and the data's resolution of g2.

The sweep (docs/studies/rwa_sweep.md, docs/studies/final_sweep.md) trains every seed
from two g2 starts and keeps, per seed, the run with the lower total loss
(docs/studies/g2_start.md). Total losses are compared only between runs whose
settings differ in nothing but the start, which ``select_starts`` checks.

``resolution_band`` gives the range of g2 the data cannot tell from the truth: the
values whose exact trajectory stays within the PINN's fit floor of the data.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinn_jc.data.simulate import simulate
from pinn_jc.io.runs import load_run
from pinn_jc.training.config import ExperimentConfig

LOSSES = ("loss_ic", "loss_norm", "loss_data", "loss_ode")


@dataclass(frozen=True)
class SeedRun:
    """The outcome of one seed from one start.

    ``total`` is the sum of the unweighted loss terms, each averaged over the last
    epochs, as quoted in the study reports.
    """

    seed: int
    start: str
    couplings: tuple[float, ...]
    loss_data: float
    total: float
    config: ExperimentConfig


def seed_run(run_dir: Path | str, start: str, last: int = 1000) -> SeedRun:
    """Read the outcome of a saved run, its losses averaged over the last ``last`` epochs."""
    run = load_run(run_dir)
    losses = {key: float(np.mean(run.history[key][-last:])) for key in LOSSES}
    return SeedRun(
        seed=run.config.training.seed,
        start=start,
        couplings=tuple(float(g) for g in run.history["couplings"][-1]),
        loss_data=losses["loss_data"],
        total=sum(losses.values()),
        config=run.config,
    )


def _settings(config: ExperimentConfig) -> dict:
    """Return the config without what a start may change (seed, ``coupling_init``)."""
    settings = config.to_dict()
    settings["physics"] = {
        k: v for k, v in settings["physics"].items() if k != "coupling_init"
    }
    settings["training"] = {
        k: v for k, v in settings["training"].items() if k != "seed"
    }
    return settings


def select_starts(starts: Mapping[str, Sequence[SeedRun]]) -> dict[int, SeedRun]:
    """Keep, per seed, the start with the lowest total loss.

    Parameters
    ----------
    starts : mapping of str to sequence of SeedRun
        The runs of each start, keyed by the start's name.

    Returns
    -------
    dict of int to SeedRun
        The selected run of every seed that ran from every start.

    Raises
    ------
    ValueError
        If two starts differ in other settings than the seed and ``coupling_init``,
        since their total losses are then not comparable.
    """
    runs = [run for start_runs in starts.values() for run in start_runs]
    reference = _settings(runs[0].config)
    for run in runs[1:]:
        if _settings(run.config) != reference:
            raise ValueError(
                f"start {run.start!r} (seed {run.seed}) differs from start "
                f"{runs[0].start!r} in more than the seed and coupling_init"
            )
    by_seed: dict[int, list[SeedRun]] = {}
    for run in runs:
        by_seed.setdefault(run.seed, []).append(run)
    return {
        seed: min(candidates, key=lambda run: run.total)
        for seed, candidates in sorted(by_seed.items())
        if len(candidates) == len(starts)
    }


def resolution_band(
    config: ExperimentConfig,
    floor: float,
    grid: Sequence[float] = tuple(np.linspace(0.0, 1.5, 76)),
) -> tuple[float, float]:
    """Range of g2 whose exact data stay within ``floor`` of the training data.

    Every g2 on ``grid`` is simulated with the rest of ``config`` (g1 at its true
    value) and compared with the noiseless data of the true g2 by the data loss
    (MAE of the observables). The band is the contiguous run of grid points around
    the true g2 below ``floor``; the true g2 is always inside it.

    Parameters
    ----------
    config : ExperimentConfig
        The trained experiment; its picture must have a coupling ``g2``.
    floor : float
        The data loss a successful run reaches (the fit floor).
    grid : sequence of float
        The values of g2 to try, in increasing order.

    Returns
    -------
    tuple of float
        The lowest and highest g2 of the band.
    """
    physics, data = config.physics, config.data

    def expectations(g2):
        return simulate(
            {**physics.params, "g2": g2},
            data.tfinal,
            data.n_time_steps,
            physics.init_state,
            physics.picture,
            physics.dims,
            init_state_options=physics.init_state_options,
            truncation=physics.truncation,
        ).expectations.numpy()

    truth = physics.params["g2"]
    reference = expectations(truth)
    grid = np.asarray(sorted({*grid, truth}))
    inside = np.array(
        [np.abs(expectations(g2) - reference).mean() < floor for g2 in grid]
    )
    centre = int(np.flatnonzero(grid == truth)[0])
    low = high = centre
    while low > 0 and inside[low - 1]:
        low -= 1
    while high < len(grid) - 1 and inside[high + 1]:
        high += 1
    return float(grid[low]), float(grid[high])


@dataclass(frozen=True)
class SweepPoint:
    """One point of the sweep figure: one data type at one g/ω.

    ``selected`` holds the learned g2 of each seed's selected start, ``rejected`` that
    of the start it beat; ``band`` is the ``resolution_band`` of the data.
    """

    data: str  # "rabi" or "jc"
    g_over_omega: float
    g2_true: float
    selected: tuple[float, ...]
    rejected: tuple[float, ...]
    band: tuple[float, float]
