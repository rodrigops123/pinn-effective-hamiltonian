# Final sweep with noise, and the g2 start map at the switch

Two questions left by `final_sweep.md`:

1. **Does the paper's result survive measurement noise across the whole sweep?**
   `measurement_noise.md` checked σ = 0.01 at ω = 5 and 7 with 5 seeds. Here the full
   final sweep is rerun with the same noise.
2. **Where does the full model's basin start at the switch?** The final sweep uses two
   g2 starts, near 1 and 0.05. At ω = 10 the start near 1 reached the truth on only
   3/10 seeds. Here g2 starts at eight values from 0.05 to 1.5, at ω = 10 and 15.

All runs: AWS EC2 c7a (x86, us-east-2), from clean commit 9033ed5, seeds 100–109. The
noiseless final sweep ran on Apple Silicon. Seeds give the same initialisation on both
platforms, but training differs in the last digits, so a seed near a basin boundary
can end differently. The two platforms are compared by outcome counts here, never pooled.

## Configs

- `configs/final_noise/final_noise_{rabi,jc}_w<ω>[_init005].yaml`: `configs/final/`
  with `data.noise_std: 0.01` and `data.noise_seed: 0`. Same two starts, epochs and
  selection (lowest total loss per seed).
- `..._w{10,15}[_init005]_nseed{1,2}.yaml`: two more noise draws at the switch, so that
  one draw does not decide g2 there.
- `configs/g2_basins/g2basin_rabi_w{10,15}_g2init<g2>.yaml`: noiseless Rabi data,
  `configs/final/final_rabi_w<ω>.yaml` with `physics.coupling_init: {g2: <g2>}` for g2
  in 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5; g1 from the softplus draw; 150k epochs.

## Figures

```bash
uv run python scripts/make_sweep_figure.py --prefix final_noise                    # noise seed 0
uv run python scripts/make_sweep_figure.py --prefix final_noise --suffix _nseed1   # and _nseed2
uv run python scripts/make_basin_figure.py                                         # start map
```

`final_noise_g2_sweep.{pdf,png}` is the sweep figure of `final_sweep.md` for noisy
data. Its shaded band keeps the noiseless definition: exact trajectories within 2e-3
data loss of the true ones. It measures what the PINN resolves, and the PINN's ⟨O⟩
error against the exact solution stays at 1–2e-3 with noise (as in
`measurement_noise.md`). The noisy data losses are in the table below.
`g2basin_rabi.{pdf,png}` shows learned g2 against its initial value, one panel per ω,
with the number of seeds reaching the full model over each start.

## Results: the sweep with noise

Rabi data (true g2 = 1), selected start per seed. In brackets, the noiseless final
sweep's count.

| ω | g/ω | full model (g2 > 0.5) | selected g2 | fidelity of the selected runs |
|---|---|---|---|---|
| 100 | 0.01 | 0/10 (0/10) | 0.000 | 1.0000 |
| 20 | 0.05 | 0/10 (0/10) | 0.000 | 0.9991 |
| 15 | 0.067 | 0/10 (0/10); 0/10, 0/10 for noise seeds 1, 2 | 0.000 | 0.9984–0.9985 |
| 10 | 0.1 | **0/10 (3/10)**; 0/10, 0/10 for noise seeds 1, 2 | ≤ 0.019 | 0.9965 |
| 7 | 0.14 | 8/10 (8/10) | 0.934–0.975; 0.000 on seeds 103, 107 | 0.9999–1.0000; 0.993 |
| 5 | 0.2 | 10/10 (10/10) | 0.973–0.996 | 1.0000 |
| 2 | 0.5 | 9/10 (10/10) | 0.994–1.000; 0.001 on seed 105 | 1.0000; 0.69 |
| 1 | 1 | 7/10 at the truth (7/10); 3 stalled at 0.74–0.89 | 0.996–0.999 | 1.0000 on 7; 0.75, 0.77, 0.99 |

g1 = 0.997–1.001 on every selected run, except ω = 1 (0.87–1.00) and ω = 2 seed 105
(0.971).

JC data (true g2 = 0), the null control: selected g2 ≤ 0.10 at every ω on every seed
(80/80, plus 40/40 for the extra noise draws). At ω = 5 the largest value is 0.10,
inside the band 0–0.28. g1 = 0.999–1.001, fidelity ≥ 0.9999, ⟨O⟩ MAE against the
exact solution 0.5–1.9e-3.

### What changes with noise

- **Both ends hold.** ω ≥ 15 gives JC on every seed and every noise draw; ω = 5 gives
  the full model 10/10; ω = 7 gives 8/10. The control passes everywhere.
- **The two failing seeds at ω = 7 are 103 and 107, as without noise.** Their g2 ≈ 1
  start falls to 0 in both cases, so the failure follows the initialisation, not the
  noise draw.
- **The switch moves: ω = 10 is JC on 30/30 seeds over three noise draws** (3/10 full
  without noise). From the start near 1, g2 is at 0.00–0.20 after 50k epochs on 9
  seeds, for every draw. On seed 109 (start 1.26) it stalls at 0.58, 0.45 and 0.32 for the three draws,
  still falling at 150k. Those runs fit the data better than the trapped start
  (data loss 8.1 vs 8.3e-3) but have a higher ODE loss, so the selection keeps
  g2 = 0, as with seed 108 without noise.
- **ω = 2, seed 105 fails.** The start near 1 falls to g2 = 0.001 (fidelity 0.69).
  Its ODE loss is 7.6× that of the other seeds (1.8e-2 vs 2.2–2.4e-3).

### Is ω = 10 still resolvable with noise?

Data loss of the *exact* solution at each g2 against the actual noisy training data
(×10⁻³, noise seed 0):

| ω | g2 = 0 | 0.5 | 0.8 | 0.9 | 1.0 (truth) | PINN, trapped at 0 | PINN, full model |
|---|---|---|---|---|---|---|---|
| 7 | 11.21 | 9.66 | 8.31 | 8.00 | 7.88 | 9.62 | 7.86–7.88 |
| 10 | 8.90 | 8.42 | 8.00 | 7.91 | 7.88 | 8.30–8.32 | none |
| 15 | 8.15 | 8.01 | 7.91 | 7.89 | 7.88 | 7.96 | none |

The truth sits at 7.88e-3 ≈ 0.8σ, the mean absolute value of the noise. At ω = 10
the exact JC trajectory is 1.0e-3 worse than the truth, much more than the scatter of
the correct runs (correct runs at ω = 5 and 7 land within 5e-5 of the truth's loss). The
data still separate the models, so the 0/30 is an optimisation failure. It can be
detected: the trapped runs are 0.4e-3 above the floor, and the JC-data runs at the
same ω reach 7.86e-3. For noise seeds 1 and 2 the trapped runs reach 8.49e-3 and
8.18e-3, against a truth of 8.10e-3 and 7.86e-3.

Noise weakens the pull towards the truth. For a residual δ much smaller than σ, the
mean absolute error grows like δ²/σ instead of |δ|. The gap between JC and the truth
in the PINN's data loss shrinks from 1.6e-3 without noise (2.4e-3 vs 0.8e-3) to about
0.4e-3. The ODE term's preference for g2 = 0 (`rwa_sweep.md`) is unchanged, so at
ω = 10 it now wins. This reading fits the start map below but was not tested
separately.

Noise draws differ in detail. For noise seed 2 the exact data loss is lowest at
g2 ≈ 0.9 (7.85e-3) rather than at the true 1.0 (7.86e-3): at ω = 10 one draw moves
the best fit by about 0.1.

### The data loss as a fit check

- ω = 7, 10: trapped runs sit 0.4–1.7e-3 above the floor and above the JC-data runs.
  The check works, as without noise.
- ω = 2: the failed seed is 0.3e-3 above the others. That is weak, but its ODE loss is
  7.6× higher.
- **ω = 1: the data loss does not work as a check at all.** The 0.05 start collapses g1
  to 0.49–0.79 (once to 0; g2 = 0, fidelity 0.18–0.38) on every seed, yet it fits the noisy data
  *better* than the correct runs (7.65–8.04e-3 vs 8.19–8.41e-3). Only the ODE loss
  separates them (9.7–14.6e-3 vs 2.0e-3). The total-loss selection is right on all 10
  seeds. The three stalled runs (g1 0.87–0.99, g2 0.74–0.89) have ODE losses 2–4×
  higher, and data losses (8.29–8.91e-3) that overlap those of the good runs.

## Results: the g2 start map (noiseless)

Seeds reaching the full model (g2 > 0.5 at 150k epochs) for each initial g2:

| initial g2 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|---|
| ω = 10 (g/ω = 0.1) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 6/10 | 8/10 |
| ω = 15 (g/ω = 0.067) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

- **ω = 10:** where it is reached, the full model is the lower minimum. Total loss is
  4.4–4.7e-3 vs 5.6–5.8e-3 at g2 = 0, and data loss 0.7–0.8e-3 vs 2.4e-3, so the
  selection would pick it (medians over the seeds). Its basin starts between 0.5 and 1 and covers only part of
  the seeds even from 1.5. Three runs are still climbing at 150k: seed 108 from 0.75
  (0.21 at 50k, 0.38 at 150k), seed 106 from 1.0 (0.18, 0.58) and seed 101 from 1.5
  (0.21, 0.90). Some trapped runs might escape with longer training, so the counts
  are what happens within 150k epochs.
- **ω = 15:** every start ends at g2 = 0, 80/80, with the same total loss
  (4.2–4.5e-3) and data loss (1.2e-3) whatever the start. No start reaches the full
  model, so here the JC result does not depend on the start.
- **The start near 1 in the final sweep** is the softplus draw (0.73–1.26 over the
  seeds), not exactly 1. With it, 3/10 reach the truth at ω = 10 (Apple Silicon), against
  6/10 from exactly 1.0 here. The comparison mixes the start values and the platforms.
  Both point to the same reading: at ω = 10 the outcome depends on the start, and
  starts below about 0.75 never reach the full model.

## Conclusions

1. The paper's claim holds with σ = 0.01 noise at both ends. The effective JC model is
   learned on every seed and noise draw for g/ω ≤ 0.067, and the full model on 10/10
   at g/ω = 0.2 and 9/10 at 0.5. The JC-data control passes 120/120.
2. Noise moves the switch to between g/ω = 0.1 and 0.14. At ω = 10 the data still
   resolve g2 (the exact JC trajectory is 1e-3 worse than the truth), but no run reaches
   the full model. The failure is detectable by the data loss.
3. At ω = 10, without noise, the full model is the global minimum, but its basin
   starts at g2 ≈ 0.75–1. At ω = 15 (g/ω = 0.067) no start up to 1.5 reaches it, so
   there the parsimony bias of `rwa_sweep.md` is a property of the loss landscape,
   not of the start protocol.
4. With noise, the total loss (through the ODE term) is the reliable criterion. At
   ω = 1 the data loss alone would prefer a wrong model.

## Open questions

- Longer training at ω = 10: three basin runs and noisy seed 109 are still moving at
  150k.
- Whether ⟨a²⟩, which is odd in g2 (`g2_start.md`), restores the pull towards g2 = 1
  under noise.
- The δ²/σ explanation of the noisy switch could be tested with an L2 data loss scaled
  by 1/σ².

## Reproducing

```bash
scripts/aws.sh up experiments/noise-sweep-basins
scripts/aws.sh run --stop-when-done configs/final_noise configs/g2_basins
scripts/aws.sh down                       # fetches the runs into saved_artifacts/runs/
uv run python scripts/summarize_runs.py saved_artifacts/runs/final_noise_*   # fidelities
```

Then the figure commands above. The 64 configs (640 runs) ran in one batch of about
6 h on one c7a machine (see `docs/aws.md`), or locally with
`scripts/run_configs.py configs/final_noise configs/g2_basins`. The extra noise draws
are named `..._nseed<n>` after the start suffix, which is why `make_sweep_figure.py`
takes `--suffix`.
