"""Convert legacy result pickles into a refactor-proof format.

The legacy ``.pkl`` files pickle whole ``nn.Module`` objects under their pre-package
import paths; they are read through ``pinn_jc.io.legacy.load_legacy_pickle``, which maps
those paths onto the current classes.

For every ``<name>.pkl`` it writes::

    <dst>/<name>/meta.json            provenance + what can be verified from the weights
    <dst>/<name>/run_<i>/model.pt     {"model_real": state_dict, "model_imag": state_dict}
    <dst>/<name>/run_<i>/history.npz  loss curves and learned parameters, one row per epoch

Each run is verified after writing: rebuilt networks must reproduce the legacy
networks' outputs exactly, and the history must round-trip unchanged.

Usage:
    python scripts/migrate_pickles.py [--src saved_artifacts/training] [--dst saved_artifacts/runs]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pinn_jc.io.legacy import load_legacy_pickle, models_from_legacy_state_dicts

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Experiment settings as written in the notebooks that most plausibly produced each file.
# The notebooks were edited after these runs (e.g. some runs used 50 hidden units while
# the notebooks' config says 100), so none of this is verified. Treat it as a lead only.
NOTEBOOK_HINTS = {
    "results_dict_fock": "apps/train_models.ipynb: params wc=wa=1, g1=1; picture=atom; "
    "dims 2x2; tfinal=2pi; n_time_steps=200; 10 runs, seeds 42+i",
    "results_dict_fock_superposition": "apps/train_models.ipynb: params wc=wa=1, g1=1; "
    "picture=atom; dims 2x3; tfinal=2pi; n_time_steps=200; 10 runs, seeds 42+i",
    "results_dict_coherent": "apps/train_models.ipynb: params wc=wa=1, g1=1; picture=atom; "
    "dims 2x10; tfinal=2pi; n_time_steps=500; 15 runs, seeds 42+i",
    "results_dict_squeeze": "apps/train_models.ipynb: params wc=wa=1, g1=1; picture=atom; "
    "dims 2x15; tfinal=2pi; n_time_steps=400; 5 runs, seeds 42+i",
    "results_dict_fock_single_shot": "apps/train_models_corrections.ipynb: params wc=wa=1, "
    "g1=1; picture=atom; dims 2x2; tfinal=2pi; n_time_steps=200",
    "results_dict_fock_superposition_single_shot": "apps/train_models_corrections.ipynb: "
    "params wc=wa=100, g1=1; picture=atom; dims 2x3; tfinal=2pi; n_time_steps=200",
    "results_dict_coherent_single_shot": "apps/train_models_corrections.ipynb (multi-run "
    "cell): params wc=wa=100, g1=1; picture=atom; dims 2x10; tfinal=2pi; n_time_steps=500; "
    "10 runs, seeds 42+i",
    "results_dict_rabi2_fock": "apps/train_rabi_model.ipynb: params wc=wa=100, g1=1, g2=0; "
    "picture=rabi2; dims 2x2; tfinal=8pi; n_time_steps=500; 5 runs, seeds 100+i^2. "
    "apps/generate_plots.ipynb plots these with wc=wa=10, g1=0.1 instead",
    "results_dict_rabi2_fock_jc": "same notebook as results_dict_rabi2_fock; meaning of the "
    "'_jc' suffix is not recorded",
    "results_dict_rabi2_fock_softplus": "same notebook as results_dict_rabi2_fock; '_softplus' "
    "presumably means couplings = softplus(param) rather than abs(param)",
    "results_dict_rabi2_fock_jc_softplus": "apps/train_rabi_model.ipynb writes this exact "
    "filename; see results_dict_rabi2_fock",
    "results_usc_rabi_fock": "apps/train_usc_rabi.ipynb: picture=rabi2; dims 2x4; tfinal=2pi; "
    "n_time_steps=400; seed 42",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_runs(legacy: dict) -> dict:
    """Legacy files are either {"run_i": {"models", "loss"}} or a single {"models", "loss"}."""
    if "models" in legacy:
        return {"run_0": legacy}
    return legacy


def describe_network(state_dict: dict) -> dict:
    hidden = sorted(
        k for k in state_dict if k.startswith("hidden_layers.") and k.endswith("weight")
    )
    return {
        "hidden_units": [int(state_dict[k].shape[0]) for k in hidden],
        "output_dim": int(state_dict["output_layer.weight"].shape[0]),
        "n_couplings": int(state_dict["param"].shape[0])
        if "param" in state_dict
        else 0,
    }


def history_arrays(loss: dict) -> dict:
    return {key: np.asarray(values, dtype=np.float64) for key, values in loss.items()}


def migrate_run(run: dict, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dicts = {name: model.state_dict() for name, model in run["models"].items()}
    torch.save(state_dicts, run_dir / "model.pt")
    np.savez_compressed(run_dir / "history.npz", **history_arrays(run["loss"]))

    # Verify: the current modules rebuilt from the saved weights reproduce the legacy
    # networks exactly, and the history round-trips.
    loaded = torch.load(run_dir / "model.pt", weights_only=True)
    state = models_from_legacy_state_dicts(loaded)["state"]
    time = torch.linspace(0, 30, 257).reshape(-1, 1)
    with torch.no_grad():
        for name, rebuilt in (("model_real", state.real), ("model_imag", state.imag)):
            if not torch.equal(rebuilt(time), run["models"][name].eval()(time)):
                raise RuntimeError(
                    f"{run_dir}/{name}: rebuilt network differs from legacy"
                )
    with np.load(run_dir / "history.npz") as history:
        for key, values in history_arrays(run["loss"]).items():
            if not np.array_equal(history[key], values):
                raise RuntimeError(f"{run_dir}: history key {key!r} did not round-trip")

    final_couplings = run["loss"].get("learned_param", [None])[-1]
    return {
        "epochs": len(run["loss"]["total_loss"]),
        "final_total_loss": run["loss"]["total_loss"][-1],
        "final_learned_param": final_couplings,
    }


def migrate_file(pkl_path: Path, dst_root: Path) -> None:
    legacy = load_legacy_pickle(pkl_path)
    runs = normalize_runs(legacy)
    experiment_dir = dst_root / pkl_path.stem

    run_summaries = {
        run_name: migrate_run(run, experiment_dir / run_name)
        for run_name, run in runs.items()
    }
    first_run = next(iter(runs.values()))
    meta = {
        "legacy_file": pkl_path.name,
        "legacy_sha256": sha256(pkl_path),
        "verified_from_weights": {
            **describe_network(first_run["models"]["model_real"].state_dict()),
            "activation": "sin (model_train_params default; not stored in the pickle)",
            "n_runs": len(runs),
        },
        "stored_params": first_run.get("params"),
        "notebook_hint_unverified": NOTEBOOK_HINTS.get(pkl_path.stem),
        "learned_param_note": "values as logged during training, i.e. after the positivity "
        "transform (abs or softplus) applied by the code version that ran",
        "runs": run_summaries,
    }
    (experiment_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{pkl_path.name}: {len(runs)} run(s) -> {experiment_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src", type=Path, default=PROJECT_ROOT / "saved_artifacts" / "training"
    )
    parser.add_argument(
        "--dst", type=Path, default=PROJECT_ROOT / "saved_artifacts" / "runs"
    )
    args = parser.parse_args()

    pkl_paths = sorted(args.src.resolve().glob("*.pkl"))
    if not pkl_paths:
        sys.exit(f"no .pkl files found in {args.src}")
    for pkl_path in pkl_paths:
        migrate_file(pkl_path, args.dst.resolve())


if __name__ == "__main__":
    main()
