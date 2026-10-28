# Study: why `atom_coherent` and `atom_squeeze` fail to recover g1

Configs: `configs/studies/study_*`, `study2_*`, `study3_*`. Each config holds out
nothing, uses the physics of `configs/atom_coherent.yaml` or `configs/atom_squeeze.yaml`
(Jaynes–Cummings, rotating frame, on resonance, g1 = 1, data ⟨a†a⟩ and ⟨σ+σ−⟩ on every
grid point) and trains the same four seeds (42–45), so variants are compared on
identical initialisations.

## Summary

- **The failures are not a coupling-search problem first.** With g1 fixed at its true
  value, the original setup still converges to a near-stationary state, built from
  |g,0⟩ and the top field level |e, n_max⟩. That level is stationary only because the
  field truncation removes |g, n_max+1⟩, and it supplies photon number cheaply.
- **`atom_coherent` is solved** by three changes together: excitation-number
  truncation, a loss balance that does not grow with the Hilbert-space dimension D, and
  learning rate 1e-3. All 4 seeds give g1 = 0.999–1.000 and fidelity 1.000.
- **`atom_squeeze` is not solved.** With the same recipe it converges to g1 ≈ 0.6–0.7
  with fidelity ≈ 0.49, and training longer does not move it. With g1 fixed at its true
  value the network mostly learns the state, and both solutions reach about the same
  total loss: with ⟨a†a⟩ and ⟨σ+σ−⟩ as the only data, g1 is weakly identifiable at the
  accuracy the network reaches.

## Metrics

Per seed: the learned g1, and the fidelity |⟨ψ|ψ_NN⟩|²/‖ψ_NN‖² against the exact state
on 400 points of [0, 2π], in the training frame. A seed **succeeds** when
|g1 − 1| < 0.01 and the mean fidelity exceeds 0.99. Losses are the final epoch's values;
the expectation-value error is the mean absolute error of the two observables.

## Round 1: training-procedure variants

Commit `f1f6b1b`. One change per variant against the experiment as configured.

| Variant | Change |
|---|---|
| `baseline` | none |
| `fixed_g` | g1 fixed at its true value (diagnostic) |
| `ic10`, `ic100` | initial-condition loss weight 10, 100 |
| `scaled_time` | network input τ = t / tfinal |
| `lr1e-3` | learning rate 1e-3 (default 1e-2) |
| `cosine` | cosine decay of the learning rate, 1e-2 → 1e-4 |
| `hard_ic` | ψ(t) = ψ0 + t·NN(t), exact initial condition |
| `window` | growing time window, 10 % of the grid at first, all of it after 60 % of the epochs |

| Variant | Coherent: success | Coherent: g1 per seed | Coherent: mean fidelity | Squeezed: success | Squeezed: g1 per seed | Squeezed: mean fidelity |
|---|---|---|---|---|---|---|
| baseline | 0/4 | 0.546 0.896 0.547 0.101 | 0.40 | 0/4 | 0.027 0.032 0.035 0.027 | 0.47 |
| fixed_g | 0/4 | (1) | 0.69 | 0/4 | (1) | 0.60 |
| ic10 | 2/4 | 0.654 0.621 0.993 0.994 | 0.71 | 0/4 | 0.139 0.141 0.141 0.140 | 0.47 |
| ic100 | 0/4 | 0.161 0.565 0.160 0.567 | 0.28 | 0/4 | 0.085 0.086 0.084 0.082 | 0.47 |
| scaled_time | 0/4 | 0.044 0.050 0.039 0.055 | 0.21 | 0/4 | 0.090 0.107 0.136 0.085 | 0.59 |
| lr1e-3 | 1/4 | 0.551 1.000 0.999 1.001 | 0.82 | 0/4 | 0.235 0.640 0.608 0.224 | 0.47 |
| cosine | 0/4 | 0.553 0.997 0.553 0.112 | 0.45 | 0/4 | 0.048 0.149 0.155 0.043 | 0.48 |
| hard_ic | 0/4 | 0.163 0.558 0.162 0.332 | 0.25 | 0/4 | 0.141 0.144 0.145 0.141 | 0.47 |
| window | 0/4 | 0.548 0.548 0.993 0.547 | 0.49 | 0/4 | 0.028 0.032 0.036 0.028 | 0.47 |

The baselines reproduce results of earlier code versions: then, 1 of 15 coherent seeds
and none of 5 squeezed seeds recovered g1.

### Diagnosis

`fixed_g` fails on every seed, with flat fidelity (0.69 coherent, 0.60 squeezed) at all
times, and its losses stop moving by epoch 1000. The predicted state is nearly constant:

| | \|g,0⟩ | \|e, n_max⟩ |
|---|---|---|
| coherent, population | 0.55 | 0.054 (n_max = 9) |
| squeezed, population | 0.92 | 0.086 (n_max = 14) |

Both states are stationary in the truncated Hamiltonian (H|g,0⟩ = 0, and
H|e, n_max⟩ = 0 because its partner |g, n_max+1⟩ is truncated away). The ODE loss is
therefore satisfied, and the small weight on the top level reproduces the mean photon
number (0.054 × 9 ≈ 0.5; 0.086 × 14 ≈ 1.2). Two further effects make this state hard to
escape:

- the initial-condition loss averages over the D components while the ODE loss sums
  over them, so its relative weight falls as 1/D (D = 20 and 30 here, against 4 for the
  Fock experiments);
- ⟨a†a⟩ + ⟨σ+σ−⟩ is conserved, so the data are one time series for a 20–30-dimensional
  complex state.

Other observations: a lower learning rate or a moderate initial-condition weight delays
the collapse; decaying the learning rate refines whatever solution was reached; and the
coherent runs return repeatedly to a second spurious solution, g1 ≈ 0.55 with fidelity
≈ 0.32. Scaled time is probably worse because sine networks with the default
initialisation cannot represent the needed frequencies on τ ∈ [0, 1] (not verified).

## Round 2: truncation and loss balance

Commit `2dcbc8b`. Two new options:

- `physics.truncation: excitation` keeps the states |s, n⟩ with s + n < n_field, which
  drops |e, n_max⟩ (D = 19 coherent, 29 squeezed). Simulation still integrates the full
  basis and projects; it raises if a dropped state becomes populated or if an operator
  couples kept and dropped states.
- `training.ode_reduction: mean` averages the ODE residual over components instead of
  summing it, so the balance between loss terms no longer depends on D. The alternative
  `ic D/4` scales the initial-condition weight instead (4.75 coherent, 7.25 squeezed).

| Variant | Coherent: success | Coherent: g1 per seed | Coherent: mean fidelity | Squeezed: success | Squeezed: g1 per seed | Squeezed: mean fidelity |
|---|---|---|---|---|---|---|
| trunc | 0/4 | 0.547 0.547 0.547 0.547 | 0.33 | 0/4 | 0.046 0.044 0.052 0.051 | 0.46 |
| trunc, g1 fixed (lr 1e-2) | 0/4 | (1) | 0.67 | 0/4 | (1) | 0.65 |
| trunc + lr 1e-3 | 0/4 | 0.552 0.552 0.552 0.552 | 0.32 | 0/4 | 0.642 0.650 0.683 0.687 | 0.37 |
| ODE mean + lr 1e-3 | 3/4 | 0.528 1.000 1.000 1.000 | 0.85 | 0/4 | 0.571 0.632 0.726 0.623 | 0.48 |
| **trunc + ODE mean + lr 1e-3** | **4/4** | 0.999 1.000 0.999 1.000 | **1.000** | 0/4 | 0.639 0.594 0.579 0.597 | 0.49 |
| **trunc + ic D/4 + lr 1e-3** | **4/4** | 0.999 0.999 0.999 0.999 | **1.000** | 0/4 | 0.383 0.396 0.402 0.388 | 0.43 |

For the coherent state no single change is enough: the loss balance with the lower
learning rate reaches 3/4, and the truncation recovers the last seed. Both ways of
balancing the losses work. The successful runs reach expectation-value errors of
0.002–0.003.

The squeezed state changes failure mode: the data and initial condition are now fitted
well (expectation-value error 0.005–0.006, initial-condition loss below 1e-3), but g1
stays near 0.6 and the fidelity decays from 1 at t = 0 to about 0.2 by t ≈ 4–6. At epoch
30 000 all losses and g1 were still changing.

## Round 3: the squeezed state, trained longer

Commits `fa5967b` and `7c780eb`. The round-2 recipe (truncation, ODE mean, lr 1e-3).

| Variant | Success | g1 per seed | Mean fidelity | Expectation-value error |
|---|---|---|---|---|
| 100k epochs | 0/4 | 0.646 0.595 0.644 0.640 | 0.49 | 0.0039 |
| 150k epochs | 0/4 | 0.649 0.601 0.650 0.648 | 0.49 | 0.0033 |
| 100k epochs, ODE weight 3 | 0/4 | 0.710 0.714 0.659 0.670 | 0.49 | 0.0061 |

g1 settles by about epoch 50 000 and moves by less than 0.01 afterwards; the data fit
keeps improving while the fidelity curve stays the same. The squeezed runs are therefore
not undertrained: they converge to a stable wrong solution.

Diagnostic: the same recipe with g1 fixed at its true value (30k epochs).

| Seed | Mean fidelity | Minimum fidelity | Expectation-value error |
|---|---|---|---|
| 42 | 0.764 | 0.648 | 0.0090 |
| 43 | 0.930 | 0.909 | 0.0060 |
| 44 | 0.980 | 0.962 | 0.0056 |
| 45 | 0.988 | 0.974 | 0.0052 |

With the coupling given, three of four seeds come close to the exact state, and the
ODE loss was still decreasing at the end. The final losses of the two kinds of run are
nearly equal:

| | Data loss | ODE loss | Initial-condition loss |
|---|---|---|---|
| g1 fixed at 1 (seeds 44, 45) | ≈ 4 × 10⁻³ | ≈ 6 × 10⁻³ | ≈ 9 × 10⁻⁴ |
| g1 learned, ≈ 0.65 (150k) | ≈ 3 × 10⁻³ | ≈ 8 × 10⁻³ | ≈ 4 × 10⁻⁴ |

At the accuracy the network reaches, a wrong coupling with a wrong state costs about as
much as the true one. The obstacle for the squeezed vacuum is weak identifiability of
g1 from these two observables, not the network's capacity or the length of training.

## Round 4: the recipe on the Fock experiments

Commit `eed7687`. Does the coherent recipe (truncation, ODE mean, lr 1e-3) also suit the
experiments that already worked? Four seeds, 30k epochs, against their previous
settings.

| Experiment | Settings | g1 per seed | g1 jitter (std, last 1000 epochs) | Mean fidelity | Expectation-value error |
|---|---|---|---|---|---|
| `atom_fock` | previous | 0.9993 1.0010 0.9983 0.9990 | 1.1 × 10⁻³ | 0.99995 | 0.0045 |
| `atom_fock` | recipe | 0.9999 0.9998 0.9996 0.9998 | 7.4 × 10⁻⁵ | 1.00000 | 0.0023 |
| `atom_fock_superposition` | previous | 0.9991 0.9977 0.9987 0.9985 | 9.3 × 10⁻⁴ | 0.99992 | 0.0049 |
| `atom_fock_superposition` | recipe | 0.9993 1.0000 1.0000 0.9999 | 1.8 × 10⁻⁴ | 1.00000 | 0.0020 |

The recipe reduces the g1 error about fivefold, the epoch-to-epoch jitter of g1 by 5–15×
and the expectation-value error by half, at the same cost per epoch. All `atom_*`
experiments now use it. The `rabi2_ip` experiments cannot use the excitation truncation
(the counter-rotating terms change the excitation number) and keep their settings.

## Round 5: the loss balance and learning rate on the `rabi2_ip` experiments

Commit `df5dd0b`. The excitation truncation does not apply to `rabi2_ip`, so the check
compares the previous settings with the mean ODE reduction and learning rate 1e-3.
Four seeds; true couplings g1 = 1 and g2 = 0 (`rabi2_jc_fock`) or g1 = g2 = 1
(`rabi2_g_equals_omega_fock`).

| Experiment | Settings | g1 per seed | g2 per seed | Mean fidelity (min) |
|---|---|---|---|---|
| `rabi2_jc_fock` | previous | 0.999 0.998 0.457 1.000 | 0.079 0.096 0.060 0.083 | 0.881 (0.42) |
| `rabi2_jc_fock` | ODE mean + lr 1e-3 | 1.000 1.000 1.000 1.000 | 0.649 0.713 0.532 0.720 | 1.000 (1.000) |
| `rabi2_g_equals_omega_fock` | previous | 0.993 0.993 0.992 0.991 | 0.995 0.994 0.993 0.997 | 0.9998 (0.9994) |
| `rabi2_g_equals_omega_fock` | ODE mean + lr 1e-3 | 0.999 0.978 0.999 0.847 | 1.000 0.990 1.000 0.021 | 0.885 (0.010) |

- `rabi2_g_equals_omega_fock`: the change hurts (one seed fails, g1 and g2 fluctuate
  10× more), so the previous settings stay.
- `rabi2_jc_fock`: the change recovers g1 and the state on every seed (the previous
  settings fail on seed 104), but g2 is **not converged in either setting**. From its
  initial value (1.19 for seed 100) it drifts slowly towards 0: 0.52, 0.20, 0.11, 0.08 at
  epochs 1k, 5k, 20k, 50k with lr 1e-2, and 1.13, 1.03, 0.85, 0.65 with lr 1e-3. The
  final g2 thus measures training length rather than the data. This is expected: at
  ωc = ωa = 100 the counter-rotating terms oscillate as e^{±200it}, about 10 rad per
  grid step (Δt = 0.05), so the grid does not resolve them and their effect on the
  dynamics is of order g2/2ω. The previous settings stay; the experiment needs a
  redesign (smaller ω, a finer grid, or longer training with g2 reported as a
  trajectory) before its g2 supports a claim.

## Conclusions and open questions

1. For multi-photon initial states, use the excitation-number truncation, a loss
   balance independent of D (`ode_reduction: mean`, or an initial-condition weight that
   grows with D), and learning rate 1e-3. `configs/atom_coherent.yaml`,
   `atom_fock.yaml` and `atom_fock_superposition.yaml` use this recipe.
2. Hard initial conditions, learning-rate decay, scaled time, a growing time window and
   larger initial-condition weights did not help in this setting. The options remain
   available (all off by default) so that these results can be reproduced.
3. For the squeezed vacuum, the open question is the data. Observables sensitive to
   coherences (⟨a⟩, ⟨σx⟩) or the photon-number distribution P(n, t) would make the
   spurious solution costly; whether they can be assumed measured is a modelling
   decision. Multi-start or warm-up strategies for g1 would not remove the flat region.
4. `rabi2_jc_fock` recovers g1 but cannot determine g2 as designed (round 5).

## Reproducing

```bash
uv run python scripts/train.py configs/studies/study2_coherent_trunc_odemean_lr1e-3.yaml --workers 4
```

Each variant takes 4–7 minutes per seed on one core (about 7 ms per epoch for
coherent, 4 ms for squeezed); the 100k and 150k runs take 15–25 minutes per seed.
