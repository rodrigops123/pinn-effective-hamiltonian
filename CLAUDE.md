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

The main entry points are Jupyter notebooks in `apps/`:
- `apps/train_models.ipynb` — standard training workflow
- `apps/train_rabi_model.ipynb` — Rabi model training
- `apps/generate_plots.ipynb` — results visualization

## Architecture

### Data Flow
```
src/pinn_jc/physics/  (operators, Hamiltonian registry, initial states, frames)
      ↓
src/pinn_jc/data/simulate.py
  → SimulationData (states, expectations, observables, H, time + which picture/frame)
      ↓
src/pinn_jc/training/trainer.py  (PINN training loop)
  → trained neural networks
      ↓
src/pinn_jc/visualization/plots.py
```

### Key Modules

**`src/pinn_jc/physics/`** — Single source of truth for the physics. `hamiltonians.py` defines every picture (`interaction`, `atom`, `full`, `rabi`, `rabi2`, `rabi2_ip`, `geral`) once as a tuple of `Term(coefficient, operator, frequency)`; `Hamiltonian(picture, dims)` builds operators once and emits both `to_qobj(params)` (simulation) and `to_torch(params, couplings, time)` (ODE loss; `couplings` follow `Picture.couplings` order). `operators.py` names the operators (basis: `qutip.tensor(atom, field)`, atom index 1 = excited, σ- = destroy(atom)). `states.py` is the validated initial-state registry (`fock`, `fock_superposition`, `coherent`, `squeeze`). `frames.py` has the `Frame` enum and the exp(+iH0 t) rotation. Add new physics here, never inline in the trainer.

**`src/pinn_jc/data/simulate.py`** — `simulate(..., frame=None) -> SimulationData` integrates with QuTiP and returns the states in `frame` (default: the requested picture's own frame, so they satisfy that picture's ODE). Integration may use a different picture for accuracy (`rabi2_ip` → `rabi2`; resonant `full`/`rabi2` with g2 = 0 → `interaction`), recorded in `simulated_picture`; frames are converted exactly with diagonal phases (`physics/frames.py`). Invariant, tested in `tests/test_physics_consistency.py`: training data must satisfy the ODE of the Hamiltonian being trained.

**`src/pinn_jc/training/losses.py`** — Four PINN loss components:
- `loss_ic`: initial condition (state at t=0)
- `loss_norm`: quantum normalization ||ψ||² = 1
- `loss_data`: MAE of expectation values at fixed time indices (`trainer.data_indices`: first instant if `n_points_loss` is None/1, else a fixed random subset drawn once with a private generator)
- `loss_ode`: residual of dψ/dτ = -i T H ψ via autodifferentiation (rows of `nn_state` are ψ(τ); H acts as H ψ; T = `time_scale`, 1 unless `is_scaled`)

**`src/pinn_jc/training/trainer.py`** — Training loops. `train()` uses a fixed Hamiltonian; `train_with_parameter()` jointly learns coupling constants (g1, g2, etc.) alongside the state; picture `rabi2` is trained as `rabi2_ip` with data simulated in that frame.

**`src/pinn_jc/models/`** — `MLP` (fully connected, `Sine` activation by default), `ComplexStateNet(real, imag)` giving ψ(t) = real(t) + i·imag(t), and `Couplings(n, positivity="softplus"|"abs")` holding the learnable g's (`.raw` parameter, θ ~ U(0,1) at init). Physics parameters never live inside a network. `trainer.build_models` constructs real net → couplings → imag net; this order reproduces the legacy RNG stream, so do not reorder it.

**`src/pinn_jc/settings.py`** — Global defaults: `SEED`, default model architecture (`units=[100,100,100]`, `Sine()` activation, lr 0.01), `DEVICE="cpu"`.

**`src/pinn_jc/io/legacy.py`** — Everything pre-Phase-4: `LegacyNeuralNet` (old class, only for unpickling), `load_legacy_pickle` (remaps old class paths), `models_from_legacy_state_dicts` (migrated `{"model_real","model_imag"}` weights → current modules) and `as_models` (accepts new `{"state","couplings"}` or legacy `models_dict`; used by the plots).

### Complex Number Handling

Networks output real tensors; `ComplexStateNet` combines the real- and imaginary-part `MLP`s into a complex ψ. Internally, `torch.complex64` is used for operator algebra and ODE loss computation.

### Learnable Parameters

`train_with_parameter(..., positivity="softplus")` learns the couplings (in `Picture.couplings` order) through a `Couplings` module and returns `({"state": ComplexStateNet, "couplings": Couplings}, loss_dict)`; `loss_dict["learned_param"]` holds the transformed values per epoch. `train()` returns `couplings=None`.

## Saved Outputs

Trained models and plots are saved to `saved_artifacts/` (gitignored). Legacy pickles live in `saved_artifacts/training/`; their migrated form is in `saved_artifacts/runs/`.
