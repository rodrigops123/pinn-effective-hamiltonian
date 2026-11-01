"""Figure of the final sweep: learned g2 against g/ω, Rabi and JC data.

For every ``<prefix>_{rabi,jc}_w<ω><suffix>`` experiment with a
``<prefix>_{rabi,jc}_w<ω>_init005<suffix>`` partner (the second g2 start,
docs/studies/g2_start.md), each seed keeps the start with the lower
total loss. The shaded band is the range of g2 whose exact data stay within the fit
floor of the training data (``pinn_jc.sweep.resolution_band``). The selected
couplings and losses of every seed go to ``<prefix>_sweep.json`` next to the
figure. ``--suffix`` selects variants such as the extra noise draws
(``final_noise_rabi_w10_nseed1``); outputs are named ``<prefix><suffix>_...``.

Usage:
    uv run python scripts/make_sweep_figure.py
    uv run python scripts/make_sweep_figure.py --prefix final --floor 2e-3 --no-usetex
    uv run python scripts/make_sweep_figure.py --prefix final_noise --suffix _nseed1

Each experiment directory stands for its latest output (as in summarize_runs.py).
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pinn_jc.physics import get_picture  # noqa: E402
from pinn_jc.sweep import (  # noqa: E402
    SweepPoint,
    resolution_band,
    seed_run,
    select_starts,
)
from pinn_jc.visualization import plots  # noqa: E402

SECOND_START = "_init005"


def latest_output(experiment_dir: Path) -> Path:
    """Return the latest output directory of an experiment."""
    outputs = sorted(p for p in experiment_dir.iterdir() if (p / "meta.json").exists())
    if not outputs:
        raise SystemExit(f"no runs found in {experiment_dir}")
    return outputs[-1]


def check_clean(output: Path) -> str:
    """Return the commit of an output directory, refusing outputs of a dirty tree."""
    git = json.loads((output / "meta.json").read_text())["git"]
    if git["dirty"]:
        raise SystemExit(f"{output} comes from a dirty tree")
    return git["commit"]


def sweep_experiments(
    runs: Path, prefix: str, suffix: str = ""
) -> list[tuple[str, Path, Path]]:
    """Return (data, first start, second start) for every experiment pair."""
    pattern = re.compile(rf"({re.escape(prefix)}_(rabi|jc)_w[\d.]+){re.escape(suffix)}")
    pairs = []
    for first in sorted(runs.iterdir()):
        match = pattern.fullmatch(first.name)
        if match is None:
            continue
        second = runs / (match.group(1) + SECOND_START + suffix)
        if second.is_dir():
            pairs.append((match.group(2), first, second))
    if not pairs:
        raise SystemExit(
            f"no {prefix}_*_w*{suffix} experiments with a {SECOND_START} partner"
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("saved_artifacts/runs"))
    parser.add_argument("--prefix", default="final")
    parser.add_argument("--suffix", default="", help="e.g. _nseed1")
    parser.add_argument(
        "--floor", type=float, default=2e-3, help="fit floor of the data loss"
    )
    parser.add_argument("--last", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("saved_artifacts/figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument(
        "--no-usetex", action="store_true", help="render text without LaTeX"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    plots.use_style(usetex=not args.no_usetex)
    points, table = [], {}
    for data, first, second in sweep_experiments(args.runs, args.prefix, args.suffix):
        starts = {}
        commits = set()
        for name, experiment in (("g2 near 1", first), ("g2 = 0.05", second)):
            output = latest_output(experiment)
            commits.add(check_clean(output))
            starts[name] = [
                seed_run(seed_dir, name, args.last)
                for seed_dir in sorted(output.glob("seed_*"))
            ]
        selected = select_starts(starts)
        config = next(iter(selected.values())).config
        params = config.physics.params
        g2 = get_picture(config.physics.picture).couplings.index("g2")
        rejected = [
            run.couplings[g2]
            for runs in starts.values()
            for run in runs
            if run.seed in selected and run is not selected[run.seed]
        ]
        band = resolution_band(config, args.floor)
        point = SweepPoint(
            data=data,
            g_over_omega=params["g1"] / params["wc"],
            g2_true=params["g2"],
            selected=tuple(run.couplings[g2] for run in selected.values()),
            rejected=tuple(rejected),
            band=band,
        )
        points.append(point)
        table[first.name] = {
            "commits": sorted(commits),
            "g_over_omega": point.g_over_omega,
            "g2_true": point.g2_true,
            "band": band,
            "seeds": {
                seed: {
                    "start": run.start,
                    "couplings": run.couplings,
                    "loss_data": run.loss_data,
                    "total": run.total,
                }
                for seed, run in selected.items()
            },
        }
        print(
            f"{first.name}: g/ω {point.g_over_omega:.3g}  band {band[0]:.2f}–"
            f"{band[1]:.2f}  g2 " + " ".join(f"{g:.3f}" for g in point.selected),
            flush=True,
        )

    fig = plots.plot_g2_sweep(points)
    saved = plots.save_figure(
        fig, args.out / f"{args.prefix}{args.suffix}_g2_sweep", args.formats
    )
    plt.close(fig)
    (args.out / f"{args.prefix}{args.suffix}_sweep.json").write_text(
        json.dumps(table, indent=1)
    )
    print("\n".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
