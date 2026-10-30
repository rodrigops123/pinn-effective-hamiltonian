# Measurement noise

Configs in `configs/studies/noise_*`. Does the result of `rwa_sweep.md` survive noisy
data? Settings of the sweep (`rabi2_ip`, abs positivity with the softplus start, input
10t, ODE mean reduction, lr 1e-3, 50k epochs, seeds 100–104) with `data.noise_std:
0.01`: Gaussian noise of standard deviation σ = 0.01 on ⟨a†a⟩ and ⟨σ+σ-⟩ at every
point, one realisation (`noise_seed: 0`) for all seeds. For a population measured with
N shots σ ≤ 0.5/√N, so σ = 0.01 corresponds to about 2500 shots per point. The initial
state is exact (a known preparation), and fidelity and ⟨O⟩ MAE are measured against
the exact, noise-free solution.

## What noise does to the resolution

The data loss is an MAE, so a model difference δ much smaller than the noise raises it
only by about δ²/σ. The gap between the JC and the Rabi fit to noisy Rabi data, from
the exact trajectories (no training; `scripts/rwa_sweep_preparation.py --noise-stds
0.003 0.01 0.03`, noise seed 0):

| ω | g/ω | σ = 0 | σ = 3e-3 | σ = 1e-2 | σ = 3e-2 |
|---|---|---|---|---|---|
| 100 | 0.01 | 3.0e-5 | < 1e-7 | < 1e-6 | < 1e-6 |
| 20 | 0.05 | 9.0e-4 | 2.2e-4 | 1.1e-4 | 6.9e-5 |
| 15 | 0.067 | 1.6e-3 | 5.6e-4 | 2.7e-4 | 1.5e-4 |
| 10 | 0.1 | 3.7e-3 | 2.1e-3 | 1.0e-3 | 5.3e-4 |
| 7 | 0.14 | 7.6e-3 | 5.8e-3 | 3.3e-3 | 1.7e-3 |
| 5 | 0.2 | 1.6e-2 | 1.4e-2 | 1.0e-2 | 5.3e-3 |
| 2 | 0.5 | 0.21 | 0.20 | 0.20 | 0.19 |

Against a fit floor of about 1e-3, σ = 0.01 makes ω = 10 marginal and leaves ω ≤ 7
resolved: the switch to the effective model should move to stronger coupling.

## Results (σ = 0.01)

| ω | g/ω | Rabi data: g2 | Rabi data: fidelity | JC data: g2 | noise-free (`rwa_sweep.md`): Rabi / JC g2 |
|---|---|---|---|---|---|
| 100 | 0.01 | 0.000 (5/5) | 1.0000 | 0.000 (5/5) | 0.000 / 0.000 |
| 10 | 0.1 | 0.002–0.110 | 0.9965–0.9970 | 0.005–0.047 | 0.001–0.551 / 0.003–0.032 |
| 7 | 0.14 | 0.870–0.933 (4/5), 0.001 | 0.9998–0.9999, 0.9927 | 0.003–0.083 | 0.972–0.985 (4/5), 0.001 / 0.001–0.039 |
| 5 | 0.2 | 0.981–0.993 (5/5) | 1.0000 | 0.003–0.052 | 0.989–0.995 / 0.003–0.022 |

g1 = 0.997–1.000 in every run. All runs from the clean commit e9ab777.

- **The PINN does not fit the noise.** The data loss of the successful runs is
  7.9–8.0e-3, the noise level itself (an MAE of Gaussian noise is 0.80 σ = 8.0e-3),
  while their ⟨O⟩ MAE against the exact solution is 1.2–2.2e-3, as without noise: the
  ODE term filters the noise, and the prediction is about five times closer to the
  truth than the data are.
- **The claim holds at both ends.** At ω = 100 g2 = 0 from both data; at ω = 5 the full
  model is learned on every seed. The JC control is passed everywhere.
- **Noise slows the decisions near the switch.** At ω = 7 the learned g2 is 0.87–0.93
  after 50k epochs, still rising (0.43–0.90 at 20k), against 0.97–0.99 without noise; with
  JC data g2 decays more slowly (up to 0.083 against 0.039 at ω = 7). Both move in the
  direction of the truth; the data's pull is weaker, as the resolution table predicts.
- **ω = 10 is unchanged:** g2 ≈ 0 on every seed, as on four of five seeds without noise.

### The two-start protocol under noise

Rabi data with g2 started at 0.05 (`noise_rabi_w*_init005`) give g2 = 0.000 at ω = 7
and 5, as without noise. Per seed, the lower total loss picks:

| ω | from ≈ 1: total | from 0.05: total | selected |
|---|---|---|---|
| 5 | 1.33–1.47e-2 (g2 ≈ 0.98–0.99) | 1.71–1.84e-2 (g2 = 0) | g2 ≈ 0.98–0.99 on 5/5 |
| 7 | 1.40–1.44e-2 (g2 ≈ 0.87–0.93), 1.53e-2 (seed 103, g2 = 0) | 1.45–1.53e-2 (g2 = 0) | g2 ≈ 0.87–0.93 on 4/5; seed 103 trapped from both starts |

The margins shrink with noise (at ω = 7, 2–9% of the total loss, against 17–25% at
ω = 5), but the selection is right wherever one start finds the truth. The data loss
separates more clearly: the right model fits to the noise floor (7.9–8.0e-3 against
0.80 σ = 8.0e-3), a trapped one stays above it (9.6e-3 at ω = 7, 1.0–1.1e-2 at ω = 5).
With a known noise level this is a goodness-of-fit check of the learned model: data
loss above 0.8 σ means the model misses part of the dynamics. At ω = 10 the
JC-like fits to Rabi data sit at 8.3–8.4e-3, 4–5% above the floor, against 7.9–8.0e-3
for JC data.

## Conclusions

1. With σ = 0.01 (about 2500 shots per point) the method still learns the effective
   model where the RWA holds and the full model where it does not, and the two-start
   protocol still selects correctly.
2. The PINN filters the noise: its predictions are as close to the truth as without
   noise, about five times closer than the data.
3. Noise weakens the data's pull near the switch: g2 converges more slowly at ω = 7 and
   with JC data. The switch does not visibly move at this level (ω = 10 already gave
   JC without noise), as expected from the resolution table, which predicts that
   σ = 0.01 makes ω = 10 marginal.
4. Knowing σ adds a model check: a correct model fits the data to 0.8 σ.

## Open questions

- Longer training at ω = 7 with noise (g2 was still rising at 50k).
- A larger noise level (σ = 0.03, about 280 shots) at ω = 7, where the resolution table
  predicts the switch would move.
- Noise realisations (`noise_seed`) as a second source of variation next to the
  training seeds.

## Reproducing

```bash
uv run python scripts/train.py configs/studies/noise_<name>.yaml --workers 5
uv run python scripts/summarize_runs.py saved_artifacts/runs/noise_<name> --trajectory 20000
```

Each config takes about 5 min with five workers at ω = 5–10 and 15–16 min at ω = 100.
