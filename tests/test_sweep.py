from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pinn_jc.experiments import load_experiment, run_experiment  # noqa: E402
from pinn_jc.sweep import (  # noqa: E402
    SeedRun,
    SweepPoint,
    resolution_band,
    seed_run,
    select_starts,
)
from pinn_jc.visualization import plots  # noqa: E402

FINAL = Path(__file__).parents[1] / "configs" / "final"


def short(name, seeds=(1, 2)):
    """A final-sweep experiment cut to a few epochs and seeds, with a coarse grid."""
    experiment = load_experiment(FINAL / f"{name}.yaml")
    data = replace(experiment.config.data, n_time_steps=60, n_data_points=60)
    experiment = replace(experiment, config=replace(experiment.config, data=data))
    return experiment.with_overrides(epochs=4, seeds=list(seeds))


@pytest.fixture(scope="module")
def two_starts(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    outputs = {}
    for start, name in (("near 1", "final_rabi_w5"), ("0.05", "final_rabi_w5_init005")):
        out = run_experiment(short(name), root, allow_dirty=True, progress_bar=False)
        outputs[start] = [seed_run(out / f"seed_{s}", start) for s in (1, 2)]
    return outputs


def fake(seed, start, total, config):
    return SeedRun(seed, start, (1.0, 0.5), total / 2, total, config)


def test_seed_run_reads_the_start(two_starts):
    run = two_starts["0.05"][0]
    assert run.seed == 1
    assert run.start == "0.05"
    assert run.couplings[1] == pytest.approx(0.05, abs=0.01)  # 4 epochs from 0.05
    assert run.total > run.loss_data > 0


def test_the_lower_total_loss_wins_per_seed(two_starts):
    config = two_starts["near 1"][0].config
    starts = {
        "a": [fake(1, "a", 2.0, config), fake(2, "a", 1.0, config)],
        "b": [fake(1, "b", 1.0, config), fake(2, "b", 3.0, config)],
    }
    selected = select_starts(starts)
    assert {seed: run.start for seed, run in selected.items()} == {1: "b", 2: "a"}

    selected = select_starts(two_starts)  # the real configs differ only in the start
    assert sorted(selected) == [1, 2]


def test_seeds_missing_a_start_are_dropped(two_starts):
    config = two_starts["near 1"][0].config
    starts = {
        "a": [fake(1, "a", 1.0, config), fake(2, "a", 1.0, config)],
        "b": [fake(1, "b", 2.0, config)],
    }
    assert sorted(select_starts(starts)) == [1]


def test_starts_with_other_settings_are_not_compared(two_starts):
    config = two_starts["near 1"][0].config
    longer = replace(config, training=replace(config.training, epochs=10))
    starts = {"a": [fake(1, "a", 1.0, config)], "b": [fake(1, "b", 2.0, longer)]}
    with pytest.raises(ValueError, match="more than the seed"):
        select_starts(starts)


def test_resolution_band_contains_the_truth():
    rabi = short("final_rabi_w5").config
    jc = short("final_jc_w100").config

    assert resolution_band(rabi, floor=1e-12) == (1.0, 1.0)
    low, high = resolution_band(rabi, floor=2e-3)
    assert low < 1.0 < high < 1.5  # g2 is resolved at g/ω = 0.2
    low, high = resolution_band(jc, floor=2e-3, grid=(0.0, 0.5, 1.0))
    assert (low, high) == (0.0, 1.0)  # and not at all at g/ω = 0.01


def test_sweep_figure_renders():
    points = [
        SweepPoint("rabi", 0.2, 1.0, (0.99, 1.01), (0.01,), (0.9, 1.1)),
        SweepPoint("rabi", 0.01, 1.0, (0.0, 0.0), (0.4,), (0.0, 1.5)),
        SweepPoint("jc", 0.2, 0.0, (0.0, 0.01), (), (0.0, 0.1)),
    ]
    fig = plots.plot_g2_sweep(points)
    rabi, jc = fig.axes
    assert rabi.get_xscale() == "log"
    assert len(rabi.collections) == 1  # the band
    plt.close(fig)
