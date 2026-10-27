# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics Informed Neural Networks (PINN) research project for learning quantum mechanical systems — specifically the vector state evolution and effective Hamiltonian of the Jaynes-Cummings (JC) model (atom-cavity interaction). Originally a Master's thesis codebase (dissertation defended); now being refactored in phases to support a paper.

## Setup and Running

Python 3.12. Install the locked environment and the package (src layout, package `pinn_jc`):
```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
pre-commit install   # ruff check/format + nbstripout
```

Run the tests (golden characterization + physics invariants; known bugs are strict xfails):
```bash
pytest
UPDATE_GOLDEN=1 pytest tests/test_golden.py   # only after an intentional behaviour change
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
src/pinn_jc/data/simulate.py
  → y_train (complex quantum states), expect_t (observables), H (Hamiltonian matrix)
      ↓
src/pinn_jc/training/trainer.py  (PINN training loop)
  → trained neural networks
      ↓
src/pinn_jc/visualization/plots.py
```

### Key Modules

**`src/pinn_jc/data/simulate.py`** — Quantum simulation via QuTip. Generates training data for multiple Hamiltonian pictures (`"interaction"`, `"atom"`, `"full"`, `"rabi"`, `"rabi2"`, `"rabi2_ip"`) and initial states (Fock, coherent, squeezed, superposition).

**`src/pinn_jc/training/losses.py`** — Four PINN loss components:
- `loss_ic`: initial condition (state at t=0)
- `loss_norm`: quantum normalization ||ψ||² = 1
- `loss_data`: matches expectation values of operators
- `loss_ode`: enforces Schrödinger equation via autodifferentiation

**`src/pinn_jc/training/trainer.py`** — Training loops. `train()` uses a fixed Hamiltonian; `train_with_parameter()` jointly learns coupling constants (g1, g2, etc.) alongside the state.

**`src/pinn_jc/models/mlp.py`** — Standard multi-layer fully connected network. Real and imaginary parts of the complex quantum state are handled by separate networks.

**`src/pinn_jc/settings.py`** — Global defaults: `SEED`, default model architecture (`units=[100,100,100]`, `SIN()` activation, lr 0.01), `DEVICE="cpu"`.

**`src/pinn_jc/io/legacy.py`** — `load_legacy_pickle` reads result pickles saved before the package layout (remaps old class paths).

### Complex Number Handling

Networks output real tensors; real and imaginary parts of ψ are handled by separate `Neural_Net` instances. Internally, `torch.complex64` is used for operator algebra and ODE loss computation.

### Learnable Parameters

`train_with_parameter()` exposes coupling constants (e.g., `g1`, `g2`) as `nn.Parameter` with softplus activation to enforce positivity. Learned parameter values are stored per epoch in `loss_dict`.

## Saved Outputs

Trained models and plots are saved to `saved_artifacts/` (gitignored). Legacy pickles live in `saved_artifacts/training/`; their migrated form is in `saved_artifacts/runs/`.
