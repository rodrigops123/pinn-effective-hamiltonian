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

### Many configs at once

```bash
uv run python scripts/run_configs.py configs/final              # every config in the directory
uv run python scripts/run_configs.py configs/final --cores 10   # leave the rest of the machine free
```

`run_configs.py` trains each config with one worker per seed and starts configs,
longest first, while enough of `--cores` (default: all) are free. A config that already
has a finished run (one with `summary.json`) of the current commit is skipped, so an
interrupted batch resumes with the same command. `--epochs`, `--seeds` and `--output`
pass through to `train.py`; use a separate `--output` for quick checks, or the skip
check will treat them as finished runs. Per-config logs go to
`saved_artifacts/run_logs/`.

## Running on AWS

`scripts/aws.sh` runs batches on one EC2 machine, driven from your computer. You need
the AWS CLI (`brew install awscli`, then `aws login` and
`aws configure set region us-east-2`), an account on the paid plan with an EC2 vCPU
quota for the instance type in that region (default us-east-2 and `c7a.24xlarge`,
96 vCPU at about $5/h on demand), and the commit to run pushed to GitHub. The
step-by-step guide, with account setup, expected output and checks, is
[`docs/aws.md`](docs/aws.md).

```bash
scripts/aws.sh setup                           # once: key pair from ~/.ssh/id_rsa.pub, SSH rule for your IP
scripts/aws.sh up experiments/final-sweep      # start, clone at that commit, uv sync, run the tests
scripts/aws.sh run --stop-when-done configs/final   # run_configs.py in tmux; survives disconnects
scripts/aws.sh status                          # or: log (follow), ssh
scripts/aws.sh start                           # after the batch stopped the machine
scripts/aws.sh down                            # fetch, then terminate (asks first)
```

The machine bills until it stops or is terminated. With `--stop-when-done` it stops
itself when the batch ends: compute billing ends, and the disk with the results stays
(a few cents a day) until `start` and `down`. `down` copies `saved_artifacts/` into the
main checkout (`fetch` does only that), then terminates, which deletes the disk; it
refuses while a batch runs. Settings are environment variables (`PINN_AWS_TYPE`,
`PINN_AWS_REGION`, `PINN_AWS_SPOT=1` for a cheaper interruptible spot machine, which has
its own quota and can be blocked by an organization policy; see the script header).
Results from a different CPU are not bit-identical to the same seeds on another
machine, so run a whole study on one platform.

## Figures

```bash
uv run python scripts/make_figures.py saved_artifacts/runs/atom_fock/<timestamp>_<commit>
```

writes, for every seed, `seed_<s>/figures/{expectations,fidelity,populations,losses,couplings}.{pdf,png}`.
The fidelity figure shows the infidelity 1 − F on a log scale; the populations figure
shows |⟨s,n|ψ(t)⟩|² of every occupied basis state, which, unlike the components, does
not depend on the global phase or the frame. `--state-components` adds the heat map
of the real and imaginary parts of every component, as a diagnostic.
For an experiment directory it also writes `figures/couplings_all_seeds.{pdf,png}`
next to the seeds: the coupling trajectories of every seed, with the seed of lowest
final total loss in bold.
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

The figure of the final sweep (`docs/studies/final_sweep.md`) takes every
`final_{rabi,jc}_w<ω>` experiment with its `_init005` partner (g2 started at 0.05),
keeps per seed the start with the lower total loss, and shades the g2 the data cannot
resolve within the fit floor:

```bash
uv run python scripts/make_sweep_figure.py --prefix final --floor 2e-3
```

It writes `saved_artifacts/figures/final_g2_sweep.{pdf,png}` and the per-seed table
`final_sweep.json`, and refuses outputs of a dirty tree or starts whose settings differ
in more than the seed and `coupling_init`. `--prefix final_noise` draws the sweep with
measurement noise; `--suffix _nseed1` (or `_nseed2`) selects the experiments with another
noise draw, `final_noise_{rabi,jc}_w<ω>[_init005]_nseed1`.

The g2 start map (`docs/studies/noise_sweep_basins.md`) plots the learned g2 against its
initial value for every `g2basin_rabi_w<ω>_g2init<g2>` experiment, one panel per ω:

```bash
uv run python scripts/make_basin_figure.py
```

It writes `saved_artifacts/figures/g2basin_rabi.{pdf,png}` and the per-seed table
`g2basin_rabi.json`.

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
- **Truncation.** `physics.truncation: field` (default) keeps the full product basis;
  `excitation` keeps |s,n⟩ with s + n < N, which removes the stationary truncation
  artefact |e, N−1⟩ of Jaynes–Cummings (see `docs/studies/coherent_squeeze.md`).
- **Training options** (all off by default): `training.ode_reduction` (`sum` or `mean`
  over components), `training.lr_schedule` (`cosine`, `exponential`) with
  `final_learning_rate`, a growing time window (`window_start`, `window_epochs`),
  `training.freeze_couplings_epochs` (couplings held at their start while the state
  trains), `model.hard_initial_condition` and `model.input_scale` (the networks see
  input_scale × t; 10 lets them represent the fast counter-rotating terms, see
  `docs/studies/g2_bias.md`).
- **Measurement noise.** `data.noise_std` adds Gaussian noise of that standard deviation
  to the expectation values the data loss fits, drawn from `data.noise_seed` (the same
  realisation for every training seed); the initial state and the evaluation stay exact.
- **Learned couplings** are non-negative, g = f(θ): `physics.positivity` is `softplus`
  (default) or `abs`. `physics.coupling_draw: softplus` starts abs couplings where
  softplus ones start (0.69 to 1.31) instead of in U(0, 1);
  `physics.coupling_init: {g2: 0.05}` sets a start, and
  `physics.fixed_couplings: {g2: 1.0}` holds a coupling fixed while the others are
  learned.
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
configs/studies/             method studies (see docs/studies/)
configs/reproduction/        reruns of the legacy experiments
configs/rwa_sweep/           Rabi vs JC across g/ω, including g = ω (docs/studies/rwa_sweep.md)
configs/final/               the sweep with 10 seeds and two g2 starts (docs/studies/final_sweep.md)
configs/final_noise/         the final sweep with measurement noise σ = 0.01 (noise seeds 0–2 at ω = 10, 15; docs/studies/noise_sweep_basins.md)
configs/g2_basins/           Rabi data at ω = 10, 15 with g2 started at 0.05–1.5 (docs/studies/noise_sweep_basins.md)
docs/studies/                study reports (index in docs/studies/README.md)
scripts/
    train.py                 train an experiment, once per seed
    run_configs.py           train many configs at once, skipping finished ones
    aws.sh                   run batches on an AWS EC2 machine
    make_figures.py          render the figures of saved runs
    migrate_pickles.py       convert result pickles of earlier code versions
    summarize_runs.py        per-seed couplings, losses and fidelity (the study tables)
    make_sweep_figure.py     learned g2 against g/ω, one start per seed by lowest loss
    make_basin_figure.py     learned g2 against its initial value (g2 start map)
    rwa_sweep_preparation.py field dimension, Rabi–JC difference and g2 resolution per ω
    g2_bias_diagnostics.py   loss landscape and supervised-fit diagnostics of g2
    g2_symmetry_check.py     the data are even in g2
src/pinn_jc/
    physics/                 operators, Hamiltonian registry, initial states, frames
    data/simulate.py         exact QuTiP solutions in a requested frame
    models/                  MLP, complex state network, learnable couplings
    training/                experiment config, losses, training loop, history
    experiments.py           experiment files and the multi-seed runner
    evaluation.py            predictions vs exact solution, in the training frame
    sweep.py                 two-start selection and g2 resolution band of the sweep
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
