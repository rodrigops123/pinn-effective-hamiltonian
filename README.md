# pinn-effective-hamiltonian

Physics-informed neural networks (PINNs) that learn, from expectation-value data, both
the state ψ(t) of an atom–cavity system and the coupling constants of its Hamiltonian
(Jaynes–Cummings and Rabi models).

## Method in one paragraph

A two-level atom couples to a cavity mode truncated to N Fock states. Two fully
connected networks give the real and imaginary parts of ψ(t) ∈ C^(2N); the learnable
couplings g live in a separate module, kept non-negative by a softplus (or |·|). The
networks and the couplings are trained together on four loss terms: the initial state,
the normalisation ‖ψ(t)‖ = 1, the measured populations ⟨a†a⟩ and ⟨σ+σ−⟩, and the
residual of the Schrödinger equation dψ/dt = −iHψ, computed by automatic
differentiation, where H contains the learned g. The training data are exact QuTiP
solutions, given in the frame of the Hamiltonian being learned.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (≥ 0.6.14), which also
installs the Python version in `.python-version` (3.12) when needed.

```bash
uv sync --all-groups          # locked dependencies, dev tools, Jupyter kernel; pinn_jc editable
uv run pre-commit install     # ruff lint/format and notebook-output stripping on commit
uv run pytest                 # about 10 s
```

`uv.lock` pins every package with hashes. Without uv, `uv export --format
requirements-txt` produces an equivalent pinned requirements file.

## Running an experiment

Every experiment is a YAML file in `configs/`:

| Config | System | Initial state | Learned | Seeds × epochs | CPU time* |
|---|---|---|---|---|---|
| `atom_fock` | JC, rotating frame, on resonance | \|e,0⟩, N = 2 | g1 | 10 × 30 000 | 17 min |
| `atom_fock_superposition` | same | \|g⟩(\|0⟩+\|1⟩+\|2⟩)/√3, N = 3 | g1 | 10 × 30 000 | 18 min |
| `atom_coherent` | same | \|g, α = 1⟩, N = 10 | g1 | 15 × 50 000 | 4.6 h |
| `atom_squeeze` | same | \|g⟩ S(r = 1)\|0⟩, N = 15 | g1 | 5 × 30 000 | 58 min |
| `rabi2_jc_fock` | Rabi in the interaction picture, JC data (g2 = 0), ω = 100 g1 | \|e,0⟩, N = 2 | g1, g2 | 5 × 50 000 | 25 min |
| `rabi2_g_equals_omega_fock` | Rabi in the interaction picture, g1 = g2 = ω | \|e,0⟩, N = 4 | g1, g2 | 5 × 30 000 | 20 min |

\* single process, measured on the development machine before dψ/dt was computed in one
forward-mode pass; since then `atom_coherent` and `atom_squeeze` train about 4–5× faster
per epoch, and the other experiments 5–20% faster.

```bash
uv run python scripts/train.py configs/atom_fock.yaml                       # every seed
uv run python scripts/train.py configs/atom_fock.yaml --epochs 200 --seeds 42 --allow-dirty   # quick check
uv run python scripts/train.py configs/atom_coherent.yaml --workers 5      # 5 seeds at a time
```

`--workers N` trains N seeds at once, each in its own process on one thread; a seed gives
the same result as when trained alone. On an M3 Pro (5 performance + 6 efficiency
cores), throughput stops growing at about 5–8 workers, roughly 3.5× a single process.

Each run writes `saved_artifacts/runs/<name>/<UTC timestamp>_<commit>/`:

```
experiment.yaml   the experiment as run (overrides applied)
meta.json         git commit and dirty flag, package versions, command, start/end time
train.log         training log of every seed
summary.json      final metrics per seed; mean and std of the learned couplings
seed_<s>/
    config.yaml   resolved configuration (seed included)
    model.pt      trained networks and couplings
    history.npz   per-epoch loss terms, couplings, validation metrics
    metrics.json  final values
```

The runner refuses to start when tracked files have uncommitted changes, so every
stored result maps to a commit; `--allow-dirty` overrides this for quick checks and is
recorded in `meta.json`. To change an experiment, edit its YAML (or copy it under a new
`name`), commit, and run it. A config can also hold out the end of the time grid
(`data.validation_fraction`) to measure extrapolation.

## Figures

```bash
uv run python scripts/make_figures.py saved_artifacts/runs/atom_fock/<timestamp>_<commit>
```

writes, for every seed, `seed_<s>/figures/{expectations,fidelity,state_components,losses,couplings}.{pdf,png}`.
Predictions are always compared with the exact solution **in the frame the model was
trained in**. `apps/figures.ipynb` does the same interactively. In Python:

```python
from pinn_jc.evaluation import evaluate_run
from pinn_jc.io.runs import load_run
from pinn_jc.visualization import plots

run = load_run("saved_artifacts/runs/atom_fock/<timestamp>_<commit>/seed_42")
evaluation = evaluate_run(run)
plots.use_style()  # SciencePlots + LaTeX
fig = plots.plot_expectations(evaluation)
plots.save_figure(fig, "figures/atom_fock_expectations")
```

## Conventions

- **Basis.** `qutip.tensor(atom, field)` order: |g,0⟩, …, |g,N−1⟩, |e,0⟩, …, |e,N−1⟩.
  Atomic level 1 is the excited state, so σ− = `destroy(atom)`.
- **States** are stored as rows: a trajectory has shape (T, D), row t being ψ(t_t).
  Operators act on column vectors (Hψ).
- **Frames** are rotations ψ_frame(t) = exp(+iGt) ψ_lab(t): lab (G = 0), rotating
  (G = ωa(a†a + σ+σ−)) and interaction (G = H0 = ωc a†a + ωa σ+σ−). Each Hamiltonian
  picture has one frame, and its training data are given in that frame.
- **Pictures** (`pinn_jc.physics.PICTURES`): `interaction` (resonant JC), `atom` (JC,
  rotating frame), `full` (JC, lab), `rabi2` (Rabi, lab), `rabi2_ip` (Rabi, interaction
  picture, time dependent), `rabi` (one coupling per process; Hermitian only for
  g1 = g2, g3 = g4) and `geral` (Rabi with two-photon and drive terms). Learned
  couplings follow the order of `Picture.couplings`.
- **Scaled time.** With `data.scale_time` the network sees τ = t/T and the ODE becomes
  dψ/dτ = −iTH(t)ψ, with H evaluated at physical time.
- **Fidelity** of a prediction: |⟨ψ|ψ_NN⟩|² / ‖ψ_NN‖².
- Figures show time as g1·t when g1 is defined.

## Results from earlier code versions

Result pickles written by earlier versions of the code (whole pickled networks) can
still be read: `pinn_jc.io.legacy.load_legacy_pickle` loads them,
`scripts/migrate_pickles.py` converts them to plain weights and histories, and
`apps/figures.ipynb` plots them. They do not record their configuration, so state
the physics they were trained with when evaluating them.

## Repository layout

```
configs/                     one YAML file per experiment
scripts/
    train.py                 train an experiment, once per seed
    make_figures.py          render the figures of saved runs
    migrate_pickles.py       convert result pickles of earlier code versions
src/pinn_jc/
    physics/                 operators, Hamiltonian registry, initial states, frames
    data/simulate.py         exact QuTiP solutions in a requested frame
    models/                  MLP, complex state network, learnable couplings
    training/                experiment config, losses, training loop, history
    experiments.py           experiment files and the multi-seed runner
    evaluation.py            predictions vs exact solution, in the training frame
    visualization/plots.py   figures
    io/runs.py, io/legacy.py run directories; formats of earlier code versions
    reproducibility.py       seeding
apps/figures.ipynb           interactive figures
examples/                    self-contained introductory PINN (exponential decay)
tests/                       physics invariants, regression and characterization tests
```

## Development

- `uv run pytest` runs three kinds of tests:
  - **physics invariants:** exact solutions satisfy the ODE the PINN enforces, in every frame;
  - **regression tests** for every fixed bug;
  - **golden tests** that pin numerical outputs.
- After a deliberate change of results, regenerate only the affected golden files with
  `UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py::<test>`, and explain why in the
  commit.
- `uv run ruff check . && uv run ruff format .`: the package follows numpy-style
  docstrings, enforced by ruff.
- **New physics:** add operators, pictures or initial states in `pinn_jc/physics/`; the
  same definition drives the simulation and the training.
- **New experiments:** add a file in `configs/`; training notebooks are not used.
