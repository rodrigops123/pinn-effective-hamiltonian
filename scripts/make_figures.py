"""Render the figures of saved runs into a figures/ folder next to each run.

Usage:
    uv run python scripts/make_figures.py saved_artifacts/runs/atom_fock/<timestamp>_<commit>
    uv run python scripts/make_figures.py <...>/seed_42 --formats png --no-usetex

Accepts an experiment output directory (every seed_* inside it) or a single run.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pinn_jc.evaluation import evaluate_run  # noqa: E402
from pinn_jc.io.runs import load_run  # noqa: E402
from pinn_jc.visualization import plots  # noqa: E402


def run_directories(path: Path) -> list[Path]:
    if (path / "model.pt").exists():
        return [path]
    runs = sorted(p for p in path.glob("seed_*") if (p / "model.pt").exists())
    if not runs:
        raise SystemExit(f"no runs found in {path}")
    return runs


def make_figures(run_dir: Path, formats, n_time_steps=None) -> list[Path]:
    run = load_run(run_dir)
    evaluation = evaluate_run(run, n_time_steps)
    figures = {
        "expectations": plots.plot_expectations(evaluation),
        "fidelity": plots.plot_fidelity(evaluation),
        "state_components": plots.plot_state_components(evaluation),
        "losses": plots.plot_losses(run.history),
    }
    if run.couplings is not None:
        figures["couplings"] = plots.plot_couplings(
            run.history, run.config.physics.picture, run.config.physics.params
        )
    saved = []
    for name, fig in figures.items():
        saved += plots.save_figure(fig, run_dir / "figures" / name, formats)
        plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", type=Path, help="experiment output directory or seed run"
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument(
        "--n-time-steps", type=int, help="evaluation grid (default: run's)"
    )
    parser.add_argument(
        "--no-usetex", action="store_true", help="render text without LaTeX"
    )
    args = parser.parse_args()

    plots.use_style(usetex=not args.no_usetex)
    for run_dir in run_directories(args.path):
        saved = make_figures(run_dir, args.formats, args.n_time_steps)
        print(f"{run_dir}: {len(saved)} files in {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
