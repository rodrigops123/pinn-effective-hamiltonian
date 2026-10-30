"""Summarise saved runs per seed: final couplings, losses, fidelity, ⟨O⟩ MAE.

The numbers quoted in the study reports (docs/studies/): couplings at the last epoch,
losses averaged over the last ``--last`` epochs, and the fidelity (mean and minimum)
and expectation-value MAE of the prediction against the exact solution on the
training grid. ``--trajectory`` adds the couplings at the given epochs.

Usage:
    uv run python scripts/summarize_runs.py saved_artifacts/runs/rwa_rabi_w5
    uv run python scripts/summarize_runs.py saved_artifacts/runs/rwa_* --trajectory 5000 20000
    uv run python scripts/summarize_runs.py <...>/<timestamp>_<commit> --json summary.json

An experiment directory (saved_artifacts/runs/<name>) stands for its latest output;
an output directory (<timestamp>_<commit>) for every seed in it.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from pinn_jc.evaluation import evaluate_run
from pinn_jc.io.runs import load_run

LOSSES = ("loss_ic", "loss_norm", "loss_data", "loss_ode")


def output_directory(path: Path) -> Path:
    """Return the output directory of ``path``: itself, or its latest output."""
    if (path / "meta.json").exists():
        return path
    outputs = sorted(p for p in path.iterdir() if (p / "meta.json").exists())
    if not outputs:
        raise SystemExit(f"no runs found in {path}")
    return outputs[-1]


def summarize_seed(seed_dir: Path, last: int, trajectory) -> dict:
    """Return the summary of one seed's run."""
    run = load_run(seed_dir)
    history = run.history
    evaluation = evaluate_run(run)
    losses = {k: float(np.mean(history[k][-last:])) for k in LOSSES}
    summary = {
        "seed": seed_dir.name.removeprefix("seed_"),
        **losses,
        "total": sum(losses.values()),
        "fidelity_mean": float(np.mean(evaluation.fidelity)),
        "fidelity_min": float(np.min(evaluation.fidelity)),
        "expectation_mae": float(
            np.abs(
                evaluation.predicted_expectations - evaluation.exact_expectations
            ).mean()
        ),
    }
    couplings = np.asarray(history["couplings"])
    if couplings.size:
        summary["couplings"] = couplings[-1].tolist()
        summary["trajectory"] = {
            epoch: couplings[epoch - 1].tolist()
            for epoch in trajectory
            if epoch <= len(couplings)
        }
    return summary


def _format(summary: dict) -> str:
    parts = [f"seed {summary['seed']:>4}"]
    if "couplings" in summary:
        parts.append("g " + " ".join(f"{g:.3f}" for g in summary["couplings"]))
        for epoch, values in summary["trajectory"].items():
            parts.append(f"@{epoch // 1000}k " + " ".join(f"{g:.3f}" for g in values))
    parts += [
        f"data {summary['loss_data']:.1e}",
        f"ode {summary['loss_ode']:.1e}",
        f"total {summary['total']:.1e}",
        f"fid {summary['fidelity_mean']:.4f} ({summary['fidelity_min']:.3f})",
        f"<O> MAE {summary['expectation_mae']:.1e}",
    ]
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--last", type=int, default=1000)
    parser.add_argument("--trajectory", type=int, nargs="*", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    results = {}
    for path in args.paths:
        output = output_directory(path)
        meta = json.loads((output / "meta.json").read_text())
        git = meta["git"]
        print(f"== {meta['experiment']}  ({output.name}, dirty={git['dirty']})")
        seeds = [
            summarize_seed(seed_dir, args.last, args.trajectory)
            for seed_dir in sorted(output.glob("seed_*"))
        ]
        for summary in seeds:
            print("  " + _format(summary), flush=True)
        results[meta["experiment"]] = {
            "output": str(output),
            "commit": git["commit"],
            "seeds": seeds,
        }
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
