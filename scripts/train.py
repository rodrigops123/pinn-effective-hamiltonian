"""Train an experiment from its YAML config, once per seed.

Usage:
    uv run python scripts/train.py configs/atom_fock.yaml
    uv run python scripts/train.py configs/atom_fock.yaml --seeds 42 43 --epochs 100 --allow-dirty
    uv run python scripts/train.py configs/atom_coherent.yaml --workers 5

Results go to saved_artifacts/runs/<name>/<timestamp>_<commit>/ (see pinn_jc.experiments).
"""

import argparse
import logging
import sys

from pinn_jc.experiments import DirtyWorkingTreeError, load_experiment, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="experiment YAML file")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="override the config's seeds"
    )
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--output", default="saved_artifacts/runs", help="output root")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run despite uncommitted changes (the result will not map to a commit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="seeds trained in parallel, one thread each (5-6 on an M3 Pro)",
    )
    parser.add_argument("--no-progress", action="store_true", help="hide progress bars")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s"
    )
    experiment = load_experiment(args.config).with_overrides(args.epochs, args.seeds)
    try:
        out_dir = run_experiment(
            experiment,
            args.output,
            allow_dirty=args.allow_dirty,
            progress_bar=not args.no_progress,
            workers=args.workers,
        )
    except DirtyWorkingTreeError as error:
        sys.exit(f"error: {error}")
    print(out_dir)


if __name__ == "__main__":
    main()
