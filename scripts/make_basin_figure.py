"""Figure of the g2 start map: learned g2 against its initial value, per ω.

Reads every ``<prefix>_w<ω>_g2init<...>`` experiment (configs/g2_basins/), takes the
couplings at the last epoch and the losses averaged over the last ``--last`` epochs
of each seed, and writes ``<prefix>.{pdf,png}`` and ``<prefix>.json`` (per ω and
start: every seed's couplings, data loss and total loss).

Usage:
    uv run python scripts/make_basin_figure.py
    uv run python scripts/make_basin_figure.py --runs <...>/runs --no-usetex

Each experiment directory stands for its latest output (as in summarize_runs.py).
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pinn_jc.physics import get_picture  # noqa: E402
from pinn_jc.sweep import seed_run  # noqa: E402
from pinn_jc.visualization import plots  # noqa: E402


def latest_output(experiment_dir: Path) -> Path:
    """Return the latest output directory of an experiment, refusing a dirty tree."""
    outputs = sorted(p for p in experiment_dir.iterdir() if (p / "meta.json").exists())
    if not outputs:
        raise SystemExit(f"no runs found in {experiment_dir}")
    if json.loads((outputs[-1] / "meta.json").read_text())["git"]["dirty"]:
        raise SystemExit(f"{outputs[-1]} comes from a dirty tree")
    return outputs[-1]


def basin_experiments(runs: Path, prefix: str) -> list[Path]:
    """Return the ``<prefix>_w<ω>_g2init<...>`` experiment directories."""
    pattern = re.compile(rf"{re.escape(prefix)}_w[\d.]+_g2init\d+")
    found = [p for p in sorted(runs.iterdir()) if pattern.fullmatch(p.name)]
    if not found:
        raise SystemExit(f"no {prefix}_w*_g2init* experiments in {runs}")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("saved_artifacts/runs"))
    parser.add_argument("--prefix", default="g2basin_rabi")
    parser.add_argument("--last", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("saved_artifacts/figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument(
        "--no-usetex", action="store_true", help="render text without LaTeX"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    plots.use_style(usetex=not args.no_usetex)
    panels, table, g2_true = defaultdict(dict), defaultdict(dict), set()
    for experiment in basin_experiments(args.runs, args.prefix):
        output = latest_output(experiment)
        runs = [seed_run(d, experiment.name, args.last) for d in output.glob("seed_*")]
        config = runs[0].config
        physics = config.physics
        g2 = get_picture(physics.picture).couplings.index("g2")
        init = float(physics.coupling_init["g2"])
        g_over_omega = physics.params["g1"] / physics.params["wc"]
        title = (
            rf"$\omega = {physics.params['wc']:g}$ ($g/\omega = {g_over_omega:.3g}$)"
        )
        g2_true.add(physics.params["g2"])
        panels[title][init] = [run.couplings[g2] for run in runs]
        table[f"w{physics.params['wc']:g}"][f"{init:g}"] = {
            "commit": json.loads((output / "meta.json").read_text())["git"]["commit"],
            "seeds": {
                run.seed: {
                    "couplings": run.couplings,
                    "loss_data": run.loss_data,
                    "total": run.total,
                }
                for run in sorted(runs, key=lambda r: r.seed)
            },
        }
        learned = " ".join(f"{run.couplings[g2]:.2f}" for run in runs)
        print(f"{experiment.name}: g2 {learned}", flush=True)
    if len(g2_true) != 1:
        raise SystemExit(f"experiments differ in the true g2: {sorted(g2_true)}")

    order = sorted(panels, key=lambda t: float(re.search(r"= ([\d.]+)", t)[1]))
    fig = plots.plot_g2_basins({t: panels[t] for t in order}, g2_true.pop())
    saved = plots.save_figure(fig, args.out / args.prefix, args.formats)
    plt.close(fig)
    (args.out / f"{args.prefix}.json").write_text(json.dumps(table, indent=1))
    print("\n".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
