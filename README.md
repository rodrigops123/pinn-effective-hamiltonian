# Introduction
This is a code base used to train a neural network based on the Physics Informed Neural Networks (PINN) framework to:
- learn the vector state associated with a Jaynes-Cummings model
- learn the effective Hamiltonian of a Jaynes-Cummings model

# Code structure
```
|___apps/                     # notebooks: training runs and figures
|___scripts/
|   |___migrate_pickles.py    # convert legacy result pickles to state_dict format
|___src/pinn_jc/
|   |___settings.py           # global defaults (device, seed, network defaults)
|   |___data/simulate.py      # QuTiP simulation of the training data
|   |___models/               # MLP and activations
|   |___training/             # loss functions and training loops
|   |___visualization/        # plotting
|   |___io/legacy.py          # load pickles written before the package layout
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
