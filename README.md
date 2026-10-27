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
|___requirements-lock.txt     # fully pinned environment
```

# Setup
Requires Python 3.12.
```
pip install -r requirements-lock.txt
pip install -e . --no-deps
pre-commit install            # ruff lint/format + strip notebook outputs on commit
pytest
```
`tests/test_golden.py` pins the current numerical behaviour; regenerate it with
`UPDATE_GOLDEN=1 pytest tests/test_golden.py` only after an intentional change.
Known bugs are documented as strict `xfail` tests.

The lock file is generated from `pyproject.toml` with
`uv pip compile pyproject.toml --extra dev --extra notebooks --universal --python-version 3.12 -o requirements-lock.txt`.

Training results are not tracked in git. Legacy `.pkl` results can be converted to a
format that survives refactoring with `python scripts/migrate_pickles.py`, or loaded
directly with `pinn_jc.io.legacy.load_legacy_pickle`.
