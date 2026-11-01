import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "make_sweep_figure", ROOT / "scripts" / "make_sweep_figure.py"
)
make_sweep_figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_sweep_figure)

NAMES = [
    "final_noise_rabi_w10",
    "final_noise_rabi_w10_init005",
    "final_noise_jc_w10",
    "final_noise_jc_w10_init005",
    "final_noise_rabi_w10_nseed1",
    "final_noise_rabi_w10_init005_nseed1",
    "final_noise_jc_w15_nseed1",  # no partner: skipped
    "final_rabi_w10",
    "final_rabi_w10_init005",
]


@pytest.fixture
def runs(tmp_path):
    for name in NAMES:
        (tmp_path / name).mkdir()
    return tmp_path


def _names(pairs):
    return [(data, first.name, second.name) for data, first, second in pairs]


def test_pairs_each_experiment_with_its_second_start(runs):
    assert _names(make_sweep_figure.sweep_experiments(runs, "final_noise")) == [
        ("jc", "final_noise_jc_w10", "final_noise_jc_w10_init005"),
        ("rabi", "final_noise_rabi_w10", "final_noise_rabi_w10_init005"),
    ]


def test_suffix_selects_a_variant(runs):
    pairs = make_sweep_figure.sweep_experiments(runs, "final_noise", "_nseed1")

    assert _names(pairs) == [
        ("rabi", "final_noise_rabi_w10_nseed1", "final_noise_rabi_w10_init005_nseed1")
    ]


def test_no_pair_is_an_error(runs):
    with pytest.raises(SystemExit, match="_nseed3"):
        make_sweep_figure.sweep_experiments(runs, "final_noise", "_nseed3")
