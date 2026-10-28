# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Project

Physics-informed neural networks that learn the state ψ(t) and the coupling constants
of atom–cavity Hamiltonians (Jaynes–Cummings, Rabi) from expectation-value data. The
code backs a paper: correctness and reproducibility of results come first. The README
has the user-facing documentation, including conventions.

## Commands

The environment is managed with uv only; never use pip directly.

```bash
uv sync --all-groups                      # environment (Python 3.12, uv.lock)
uv run pytest                             # all tests, ~10 s
uv run ruff check . && uv run ruff format .
uv add <pkg> / uv add --group dev <pkg>   # never hand-edit pins
uv run python scripts/train.py configs/<name>.yaml [--epochs N --seeds S --workers W --allow-dirty]
uv run python scripts/make_figures.py saved_artifacts/runs/<name>/<run>
UPDATE_GOLDEN=1 uv run pytest "tests/test_golden.py::<test>"   # only for deliberate result changes
```

## Architecture

```
physics/ (operators, PICTURES registry, INITIAL_STATES, Frame)
  -> data/simulate.py   simulate(..., frame) -> SimulationData (states in the picture's frame)
  -> training/          ExperimentConfig -> train(config) -> TrainingResult(state, couplings, history)
  -> experiments.py + io/runs.py   YAML experiments, multi-seed runner, run directories
  -> evaluation.py      evaluate_run(run) -> Evaluation (prediction vs exact, training frame)
  -> visualization/plots.py        Evaluation / history -> Figure
```

- **`physics/`** is the single source of truth. Each picture is a tuple of
  `Term(coefficient, operator, frequency)`, and `Hamiltonian(picture, dims)` emits
  `to_qobj` (simulation) and `to_torch(params, couplings, time)` (ODE loss). Add physics
  here, never inline in the trainer.
- **Invariant**, tested in `tests/test_physics_consistency.py`: training data satisfy the
  ODE of the Hamiltonian being trained. `simulate` converts frames exactly; it may
  integrate a different picture for accuracy (recorded in `simulated_picture`).
- **Learning picture:** explicit in configs. Write `rabi2_ip`, not `rabi2`, to learn the
  Rabi model in its interaction picture.
- **`training/config.py`**: frozen, validated dataclasses with `to_dict`/`from_dict`.
  All defaults live here; `settings.py` only holds `DEVICE`.
- **`trainer.build_models`** builds the real net, then the couplings, then the imaginary
  net. This order reproduces the legacy RNG stream, so the same seed gives the same
  initialisation: do not reorder it.
- **`io/legacy.py`** reads results of earlier code versions: `LegacyNeuralNet` for unpickling,
  `load_legacy_pickle`, `as_models` (accepts current `{"state", "couplings"}` and legacy
  dicts), and `models_from_legacy_state_dicts`.
- **Docstrings** are numpy-style, enforced by ruff for `src/`. State shapes, the
  row-vector ψ convention and frames wherever they matter.

## Rules

- **Experiments:** a new experiment is a new file in `configs/`, never a training
  notebook. Stored results must come from a clean tree (the runner enforces it).
- **Generated outputs** (runs, figures, pickles) live in `saved_artifacts/`, which is
  gitignored; never commit them.
- **Golden files** change only on purpose: regenerate just the affected ones and explain
  why in the commit message.
- **Git:**
  - Work on a branch and merge pull requests with merge commits, not squash: run
    directories record commit hashes.
  - Do not add AI attribution or co-author lines to commits or pull requests.
- **Notebooks:** `examples/exponential_decay_pinn.ipynb` is a self-contained
  introductory example, independent of `pinn_jc`; keep its code unchanged. Notebook outputs are stripped on commit.
