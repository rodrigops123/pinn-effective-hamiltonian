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

**`src/pinn_jc/models/mlp.py`** — Standard multi-layer fully connected network. Real and imaginary parts of the complex quantum state are handled by separate networks.

**`src/pinn_jc/settings.py`** — Global defaults: `SEED`, default model architecture (`units=[100,100,100]`, `SIN()` activation, lr 0.01), `DEVICE="cpu"`.

**`src/pinn_jc/io/legacy.py`** — `load_legacy_pickle` reads result pickles saved before the package layout (remaps old class paths).

### Complex Number Handling

Networks output real tensors; real and imaginary parts of ψ are handled by separate `Neural_Net` instances. Internally, `torch.complex64` is used for operator algebra and ODE loss computation.

### Learnable Parameters

`train_with_parameter()` exposes coupling constants (e.g., `g1`, `g2`) as `nn.Parameter` with softplus activation to enforce positivity. Learned parameter values are stored per epoch in `loss_dict`.

## Saved Outputs

Trained models and plots are saved to `saved_artifacts/` (gitignored). Legacy pickles live in `saved_artifacts/training/`; their migrated form is in `saved_artifacts/runs/`.
