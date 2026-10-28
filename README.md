# Introduction
This is a code base used to train a neural network based on the Physics Informed Neural Networks (PINN) framework to:
- learn the vector state associated with a Jaynes-Cummings model
- learn the effective Hamiltonian of a Jaynes-Cummings model

# Code structure
```
|___apps/                     # notebooks: figures
|___configs/                  # one YAML file per experiment
|___scripts/
|   |___train.py              # train an experiment config, once per seed
|   |___migrate_pickles.py    # convert legacy result pickles to state_dict format
|___src/pinn_jc/
|   |___settings.py           # device
|   |___reproducibility.py    # set_seed
|   |___physics/              # operators, Hamiltonian registry, initial states, frames
|   |___data/simulate.py      # QuTiP simulation of the training data
|   |___models/               # MLP, complex state network, learnable couplings
|   |___training/             # experiment config, losses, training loop, history
|   |___visualization/        # plotting
|   |___experiments.py        # experiment files and the multi-seed runner
|   |___io/runs.py            # save / load run directories
|   |___io/legacy.py          # load and convert results from earlier code versions
|___tests/
|___pyproject.toml
|___uv.lock                   # exact, hashed versions of every dependency
```

# Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/) (>= 0.6.14). uv reads the
Python version from `.python-version` (3.12) and installs it if needed.
```
uv sync                       # create .venv with the locked deps + dev tools, pinn_jc editable
uv run pre-commit install     # ruff lint/format + strip notebook outputs on commit
uv run pytest
```
Use `uv sync --all-groups` to also install the Jupyter kernel for the notebooks.
Add dependencies with `uv add <pkg>` (or `uv add --group dev <pkg>`); upgrade one
deliberately with `uv lock --upgrade-package <pkg>`. Without uv, a pinned requirements
file can be produced with `uv export --format requirements-txt`.

`tests/test_golden.py` pins the current numerical behaviour; regenerate it with
`UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` only after an intentional change.
Known bugs are documented as strict `xfail` tests.

Training results are not tracked in git. Legacy `.pkl` results can be converted to a
format that survives refactoring with `uv run python scripts/migrate_pickles.py`, or loaded
directly with `pinn_jc.io.legacy.load_legacy_pickle`.

# Running experiments
Each experiment is a YAML file in `configs/`: the physics (picture, true parameters,
initial state, dimensions), the data (time grid, data points, optional held-out
validation window), the network, the training settings and an explicit list of seeds.
```
uv run python scripts/train.py configs/atom_fock.yaml
```
trains every seed and writes `saved_artifacts/runs/<name>/<UTC timestamp>_<commit>/`:
the experiment as run, the git commit and package versions, the training log, a
summary across seeds, and per seed the resolved config, weights, loss history and
final metrics. Load a run with `pinn_jc.io.runs.load_run`. The runner refuses to start
with uncommitted changes (override with `--allow-dirty`), so every result maps to a
commit. `--epochs` and `--seeds` override the config for quick checks.

The configs reproduce the experiments of the thesis notebooks, which were removed in
favour of them and remain in the git history.

