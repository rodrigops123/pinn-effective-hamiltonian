# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics Informed Neural Networks (PINN) research project for learning quantum mechanical systems — specifically the vector state evolution and effective Hamiltonian of the Jaynes-Cummings (JC) model (atom-cavity interaction). Originally a Master's thesis codebase (dissertation defended); now being refactored in phases to support a paper.

## Setup and Running

Package management is fully uv-based (Python 3.12 from `.python-version`; exact versions in `uv.lock`; src layout, package `pinn_jc`). Do not use pip directly:
```bash
uv sync --all-groups          # dev tools + notebook kernel
uv run pre-commit install     # ruff check/format + nbstripout
uv add <pkg> / uv add --group dev <pkg>   # never hand-edit pins; uv updates pyproject + uv.lock
```
Run every command through `uv run` (e.g. `uv run pytest`, `uv run python scripts/migrate_pickles.py`).

Run the tests (golden characterization + physics invariants; known bugs are strict xfails):
```bash
uv run pytest
UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py   # only after an intentional behaviour change
```

Results are not tracked in git (`saved_artifacts/` is ignored). Legacy `.pkl` results were
converted to `saved_artifacts/runs/<name>/` (state_dict + history.npz + meta.json) by
`scripts/migrate_pickles.py`.

Experiments are YAML files in `configs/` (one per experiment; `experiment:` section with name, description and explicit seeds, plus the `ExperimentConfig` sections). Train them with:
```bash
uv run python scripts/train.py configs/atom_fock.yaml                     # all seeds, full epochs
uv run python scripts/train.py configs/atom_fock.yaml --epochs 5 --seeds 1 --allow-dirty   # quick check
```
The runner refuses uncommitted changes to tracked files unless `--allow-dirty`, so stored results map to a commit. Output: `saved_artifacts/runs/<name>/<UTC timestamp>_<commit>/` with `experiment.yaml`, `meta.json` (git commit + dirty flag, package versions, command), `train.log`, `summary.json` (per-seed final metrics, coupling mean/std) and `seed_<s>/` (`config.yaml`, `model.pt`, `history.npz`, `metrics.json`). Reload a run with `pinn_jc.io.runs.load_run(dir)`. Figures: `uv run python scripts/make_figures.py <experiment dir or seed dir>` writes PDF/PNG into `<seed dir>/figures/`; `apps/figures.ipynb` is the interactive equivalent (also covers thesis-era pickles). Never add training notebooks: add a config.

## Architecture

### Data Flow
```
src/pinn_jc/physics/  (operators, Hamiltonian registry, initial states, frames)
      ↓
src/pinn_jc/data/simulate.py
  → SimulationData (states, expectations, observables, H, time + which picture/frame)
      ↓
src/pinn_jc/training/trainer.py  train(ExperimentConfig) -> TrainingResult
  → trained neural networks (saved/loaded by experiments.py + io/runs.py)
      ↓
src/pinn_jc/evaluation.py  evaluate / evaluate_run -> Evaluation (prediction vs exact, same frame)
      ↓
src/pinn_jc/visualization/plots.py  Evaluation / history -> matplotlib Figure
```

### Key Modules

**`src/pinn_jc/physics/`** — Single source of truth for the physics. `hamiltonians.py` defines every picture (`interaction`, `atom`, `full`, `rabi`, `rabi2`, `rabi2_ip`, `geral`) once as a tuple of `Term(coefficient, operator, frequency)`; `Hamiltonian(picture, dims)` builds operators once and emits both `to_qobj(params)` (simulation) and `to_torch(params, couplings, time)` (ODE loss; `couplings` follow `Picture.couplings` order). `operators.py` names the operators (basis: `qutip.tensor(atom, field)`, atom index 1 = excited, σ- = destroy(atom)). `states.py` is the validated initial-state registry (`fock`, `fock_superposition`, `coherent`, `squeeze`). `frames.py` has the `Frame` enum and the exp(+iH0 t) rotation. Add new physics here, never inline in the trainer.

**`src/pinn_jc/data/simulate.py`** — `simulate(..., frame=None) -> SimulationData` integrates with QuTiP and returns the states in `frame` (default: the requested picture's own frame, so they satisfy that picture's ODE). Integration may use a different picture for accuracy (`rabi2_ip` → `rabi2`; resonant `full`/`rabi2` with g2 = 0 → `interaction`), recorded in `simulated_picture`; frames are converted exactly with diagonal phases (`physics/frames.py`). Invariant, tested in `tests/test_physics_consistency.py`: training data must satisfy the ODE of the Hamiltonian being trained.

**`src/pinn_jc/training/losses.py`** — Four PINN loss components:
- `loss_ic`: initial condition (state at t=0)
- `loss_norm`: quantum normalization ||ψ||² = 1
- `loss_data`: MAE of expectation values at fixed time indices (`trainer.data_indices`: first instant if `n_points_loss` is None/1, else a fixed random subset drawn once with a private generator)
- `loss_ode`: residual of dψ/dτ = -i T H ψ via autodifferentiation (rows of `nn_state` are ψ(τ); H acts as H ψ; T = `time_scale`, 1 unless `is_scaled`)

**`src/pinn_jc/training/`** — `config.py`: frozen, validated dataclasses `ExperimentConfig(physics, data, training, model)` with `to_dict`/`from_dict` (`PhysicsConfig`: picture the PINN learns in, true params, init state, dims, `learn_couplings`, `positivity`; `DataConfig`: grid, `n_data_points`, `validation_fraction` = held-out final stretch of time, `scale_time`; `ModelConfig`; `TrainingConfig`: epochs, lr, `LossWeights`, `seed`, `log_every`). `trainer.py`: `train(config) -> TrainingResult(config, state, couplings, history)`, one loop for learned or fixed couplings, validation fidelity / ⟨O⟩ MAE every `log_every` epochs, progress via `logging` (enable with `logging.basicConfig(level=logging.INFO)`). `history.py`: `TrainingHistory` (per-epoch unweighted terms, weighted `total_loss`, couplings, validation metrics; `to_arrays`, `to_legacy_dict`). The learning picture is explicit: write `rabi2_ip` (not `rabi2`) to learn the Rabi model in its interaction picture. Seeds: `pinn_jc.reproducibility.set_seed`.

**`src/pinn_jc/models/`** — `MLP` (fully connected, `Sine` activation by default), `ComplexStateNet(real, imag)` giving ψ(t) = real(t) + i·imag(t), and `Couplings(n, positivity="softplus"|"abs")` holding the learnable g's (`.raw` parameter, θ ~ U(0,1) at init). Physics parameters never live inside a network. `trainer.build_models` constructs real net → couplings → imag net; this order reproduces the legacy RNG stream, so do not reorder it.

**`src/pinn_jc/settings.py`** — Only `DEVICE="cpu"`; every other default lives in the config dataclasses.

**`src/pinn_jc/experiments.py` / `src/pinn_jc/io/runs.py`** — `Experiment` (YAML ↔ dataclass, `with_overrides`), `run_experiment` (one run per seed, logs always written at INFO), `save_run` / `load_run` (checkpoint format 1: `{"format", "state", "couplings"}`; `load_run` rebuilds the modules exactly without consuming the caller's RNG).

**`src/pinn_jc/evaluation.py`** — `evaluate(models, physics, tfinal, n_time_steps, scale_time)` and `evaluate_run(run)` compute predicted vs exact states, expectation values and fidelity |⟨ψ|ψ_nn⟩|²/‖ψ_nn‖² once, always in the frame of `physics.picture` (the training frame). Plots never simulate.

**`src/pinn_jc/visualization/plots.py`** — Pure figure functions returning `Figure` (no `plt.show`): `plot_expectations`, `plot_fidelity`, `plot_state_components`, `plot_losses`, `plot_couplings` (labels from `physics.coupling_label`, true values dashed), `plot_positivity_transforms`; `use_style()` applies SciencePlots (LaTeX) explicitly, `save_figure` exports. History arguments accept `TrainingHistory`, loaded arrays or legacy `loss_dict`s.

**`src/pinn_jc/io/legacy.py`** — Everything pre-Phase-4: `LegacyNeuralNet` (old class, only for unpickling), `load_legacy_pickle` (remaps old class paths), `models_from_legacy_state_dicts` (migrated `{"model_real","model_imag"}` weights → current modules) and `as_models` (accepts new `{"state","couplings"}` or legacy `models_dict`; used by the plots).

### Complex Number Handling

Networks output real tensors; `ComplexStateNet` combines the real- and imaginary-part `MLP`s into a complex ψ. Internally, `torch.complex64` is used for operator algebra and ODE loss computation.

### Learnable Parameters

With `PhysicsConfig.learn_couplings` (default), the picture's couplings (in `Picture.couplings` order) are learned through a `Couplings` module; `params` then only supplies their true values for simulating the data. `TrainingHistory.couplings` holds the transformed values per epoch.

## Saved Outputs

Trained models and plots are saved to `saved_artifacts/` (gitignored). Legacy pickles live in `saved_artifacts/training/`; their migrated form is in `saved_artifacts/runs/`.
