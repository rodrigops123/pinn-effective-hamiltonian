"""Render the figures of saved runs into a figures/ folder next to each run.

Usage:
    uv run python scripts/make_figures.py saved_artifacts/runs/atom_fock/<timestamp>_<commit>
    uv run python scripts/make_figures.py <...>/seed_42 --formats png --no-usetex

Accepts an experiment output directory (every seed_* inside it) or a single run.
For an experiment with learned couplings, it also writes
``figures/couplings_all_seeds`` next to the seeds: every seed's trajectory, with the
seed of lowest final total loss in bold.
``--state-components`` adds the heat map of every basis component (a diagnostic).
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


def make_figures(
    run_dir: Path, formats, n_time_steps=None, state_components=False
) -> list[Path]:
    run = load_run(run_dir)
    evaluation = evaluate_run(run, n_time_steps)
    figures = {
        "expectations": plots.plot_expectations(evaluation),
        "fidelity": plots.plot_fidelity(evaluation),
        "populations": plots.plot_populations(evaluation),
        "losses": plots.plot_losses(run.history),
    }
    if state_components:
        figures["state_components"] = plots.plot_state_components(evaluation)
    if run.couplings is not None:
        figures["couplings"] = plots.plot_couplings(
            run.history, run.config.physics.picture, run.config.physics.params
        )
    saved = []
    for name, fig in figures.items():
        saved += plots.save_figure(fig, run_dir / "figures" / name, formats)
        plt.close(fig)
    return saved


def make_experiment_figures(run_dirs: list[Path], out_dir: Path, formats) -> list:
    runs = {int(d.name.removeprefix("seed_")): load_run(d) for d in run_dirs}
    first = next(iter(runs.values()))
    if first.couplings is None:
        return []
    selected = min(runs, key=lambda seed: runs[seed].history["total_loss"][-1])
    fig = plots.plot_coupling_trajectories(
        {seed: run.history for seed, run in runs.items()},
        first.config.physics.picture,
        first.config.physics.params,
        selected=selected,
    )
    saved = plots.save_figure(fig, out_dir / "figures" / "couplings_all_seeds", formats)
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
        "--state-components",
        action="store_true",
        help="also render the heat map of every basis component",
    )
    parser.add_argument(
        "--no-usetex", action="store_true", help="render text without LaTeX"
    )
    args = parser.parse_args()

    plots.use_style(usetex=not args.no_usetex)
    run_dirs = run_directories(args.path)
    for run_dir in run_dirs:
        saved = make_figures(
            run_dir, args.formats, args.n_time_steps, args.state_components
        )
        print(f"{run_dir}: {len(saved)} files in {run_dir / 'figures'}")
    if len(run_dirs) > 1:
        saved = make_experiment_figures(run_dirs, args.path, args.formats)
        if saved:
            print(f"{args.path}: {len(saved)} files in {args.path / 'figures'}")


if __name__ == "__main__":
    main()
