import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from pinn_jc.experiments import load_experiment

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "run_configs", ROOT / "scripts" / "run_configs.py"
)
run_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_configs)


def _write_config(directory, name, epochs, seeds):
    experiment = load_experiment(ROOT / "configs" / "atom_fock.yaml")
    data = experiment.with_overrides(epochs=epochs, seeds=seeds).to_dict()
    data["experiment"]["name"] = name
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_collect_configs_expands_directories_and_drops_repeats(tmp_path):
    (tmp_path / "sub").mkdir()
    b = _write_config(tmp_path, "b", 10, [1])
    a = _write_config(tmp_path / "sub", "a", 10, [1])

    assert run_configs.collect_configs([tmp_path, b]) == [b, a]


def test_plan_jobs_caps_cores_and_puts_the_most_expensive_first(tmp_path):
    short = _write_config(tmp_path, "short", 100, [1, 2, 3, 4])
    long = _write_config(tmp_path, "long", 300, [1, 2])
    rounds = _write_config(tmp_path, "rounds", 100, list(range(10)))

    jobs = run_configs.plan_jobs([short, long, rounds], cores=5)

    assert [(job.name, job.cores, job.cost) for job in jobs] == [
        ("long", 2, 300),
        ("rounds", 5, 200),
        ("short", 4, 100),
    ]
    workers = run_configs.plan_jobs([short], cores=8, workers=3, epochs=7)
    assert (workers[0].cores, workers[0].cost) == (3, 14)


def test_finished_runs_needs_the_commit_and_a_summary(tmp_path):
    for run, summary in [
        ("20260101T000000Z_abcdef1", True),
        ("20260102T000000Z_abcdef1", False),
        ("20260103T000000Z_1234567", True),
    ]:
        (tmp_path / "exp" / run).mkdir(parents=True)
        if summary:
            (tmp_path / "exp" / run / "summary.json").write_text("{}")

    assert run_configs.finished_runs(tmp_path, "exp", "abcdef1" + "0" * 33) == [
        tmp_path / "exp" / "20260101T000000Z_abcdef1"
    ]
    assert run_configs.finished_runs(tmp_path, "missing", "abcdef1") == []


def test_run_jobs_respects_the_core_budget_and_reports_failures(tmp_path):
    Job = run_configs.Job
    jobs = [Job(Path(f"{n}.yaml"), n, 2, 1) for n in ("ok1", "bad", "ok2")]
    marker = tmp_path / "running"
    marker.mkdir()

    def command(job):
        # Each job records itself as running, checks at most two others run, then exits.
        code = (
            "import os, sys, time; from pathlib import Path\n"
            f"d = Path({str(marker)!r}); (d / {job.name!r}).touch()\n"
            "time.sleep(0.3); n = len(list(d.iterdir()))\n"
            f"(d / {job.name!r}).unlink(); print('running', n)\n"
            f"sys.exit(3 if {job.name!r} == 'bad' else 0)"
        )
        return [sys.executable, "-c", code]

    failed = run_configs.run_jobs(
        jobs, cores=4, command=command, log_dir=tmp_path / "logs", poll=0.05
    )

    assert failed == [jobs[1]]
    counts = [
        int((tmp_path / "logs" / f"{job.name}.log").read_text().split()[-1])
        for job in jobs
    ]
    assert max(counts) <= 2


def test_main_refuses_to_run_outside_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_configs.py", "configs"])

    with pytest.raises(SystemExit, match="repository root"):
        run_configs.main()
