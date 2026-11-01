"""Train many experiment configs at once, keeping a fixed number of cores busy.

Usage:
    uv run python scripts/run_configs.py configs/final
    uv run python scripts/run_configs.py configs/final/final_rabi_w5.yaml configs/final/final_jc_w5.yaml --cores 10
    uv run python scripts/run_configs.py configs/final --epochs 200 --seeds 1 2 --output saved_artifacts/smoke

Each config runs as ``scripts/train.py <config> --workers W`` on W cores, W being its
number of seeds (or ``--workers``, at most the seeds). Configs start longest first,
by epochs times rounds of seeds, whenever enough of the ``--cores`` are free. A config
is skipped when ``<output>/<name>/`` already holds a finished run (one with
``summary.json``) of the current commit, so an interrupted batch resumes with the same
command; unfinished run directories are left as they are. The output of each config
goes to ``saved_artifacts/run_logs/<UTC timestamp>/<name>.log``. Must be run from the
repository root.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pinn_jc.experiments import load_experiment


@dataclass(frozen=True)
class Job:
    """One config to train: its file, experiment name, cores and estimated cost."""

    path: Path
    name: str
    cores: int
    cost: int


def collect_configs(paths) -> list[Path]:
    """Expand directories into their YAML files (recursively, sorted); drop repeats."""
    configs = []
    for path in map(Path, paths):
        found = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
        configs.extend(p for p in found if p not in configs)
    return configs


def plan_jobs(configs, cores: int, workers=None, epochs=None, seeds=None) -> list[Job]:
    """One job per config, most expensive first (ties keep the given order)."""
    jobs = []
    for path in configs:
        experiment = load_experiment(path).with_overrides(epochs, seeds)
        n_seeds = len(experiment.seeds)
        job_cores = min(workers or n_seeds, n_seeds, cores)
        rounds = math.ceil(n_seeds / job_cores)
        cost = experiment.config.training.epochs * rounds
        jobs.append(Job(Path(path), experiment.name, job_cores, cost))
    return sorted(jobs, key=lambda job: -job.cost)


def finished_runs(output: Path, name: str, commit: str) -> list[Path]:
    """Finished run directories of experiment ``name`` made at ``commit``."""
    return sorted(
        p.parent for p in Path(output, name).glob(f"*_{commit[:7]}/summary.json")
    )


def run_jobs(jobs, cores: int, command, log_dir: Path, poll: float = 5.0) -> list:
    """Run ``command(job)`` for every job within ``cores``; returns the failed jobs.

    A job starts as soon as enough cores are free, in the order given (so a smaller
    job may start ahead of a larger one that does not fit yet). On Ctrl-C every running
    job is stopped.
    """
    if jobs:
        log_dir.mkdir(parents=True, exist_ok=True)
    queue, running, failed, free = list(jobs), {}, [], cores
    try:
        while queue or running:
            for job in list(queue):
                if job.cores > free:
                    continue
                log = open(log_dir / f"{job.name}.log", "w")
                process = subprocess.Popen(
                    command(job),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[process] = (job, time.monotonic(), log)
                queue.remove(job)
                free -= job.cores
                _report("START", f"{job.name} on {job.cores} cores")
            time.sleep(poll)
            for process, (job, start, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                del running[process]
                free += job.cores
                minutes = (time.monotonic() - start) / 60
                if process.returncode == 0:
                    _report("END", f"{job.name} after {minutes:.0f} min")
                else:
                    failed.append(job)
                    _report(
                        "FAILED",
                        f"{job.name} (exit {process.returncode}), see {log.name}",
                    )
    except KeyboardInterrupt:
        for process in running:
            os.killpg(process.pid, signal.SIGTERM)
        raise
    return failed


def _report(event: str, message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {event} {message}", flush=True)


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="config files or directories")
    parser.add_argument(
        "--cores", type=int, default=os.cpu_count(), help="cores to fill (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, help="seeds trained at once per config (default: all)"
    )
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--seeds", type=int, nargs="+", help="override the seeds")
    parser.add_argument("--output", default="saved_artifacts/runs", help="output root")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run despite uncommitted changes (the results will not map to a commit)",
    )
    args = parser.parse_args()

    train_script = Path("scripts/train.py")
    if not train_script.is_file():
        sys.exit("error: run from the repository root")
    if _git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        sys.exit("error: uncommitted changes to tracked files: commit them first")
    commit = _git("rev-parse", "HEAD")

    jobs = plan_jobs(
        collect_configs(args.paths), args.cores, args.workers, args.epochs, args.seeds
    )
    pending = []
    for job in jobs:
        done = finished_runs(Path(args.output), job.name, commit)
        if done:
            _report("SKIP", f"{job.name}: finished in {done[-1]}")
        else:
            pending.append(job)

    overrides = ["--output", args.output]
    if args.epochs is not None:
        overrides += ["--epochs", str(args.epochs)]
    if args.seeds is not None:
        overrides += ["--seeds", *map(str, args.seeds)]
    if args.allow_dirty:
        overrides.append("--allow-dirty")

    def command(job: Job) -> list[str]:
        return [
            sys.executable,
            str(train_script),
            str(job.path),
            "--workers",
            str(job.cores),
            "--no-progress",
            "--log-level",
            "WARNING",
            *overrides,
        ]

    # One BLAS/OpenMP thread per process: the cores are shared out by --workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    log_dir = Path("saved_artifacts/run_logs", f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    _report(
        "BATCH",
        f"{len(pending)} configs to train, {len(jobs) - len(pending)} skipped, "
        f"{args.cores} cores, commit {commit[:7]}, logs in {log_dir}",
    )
    failed = run_jobs(pending, args.cores, command, log_dir)
    _report(
        "ALL DONE",
        f"{len(pending) - len(failed)} finished, {len(failed)} failed, "
        f"{len(jobs) - len(pending)} skipped",
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
