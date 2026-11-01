# Final sweep: 10 seeds, two g2 starts

Configs in `configs/final/`. This is the paper's result: the RWA sweep (`rwa_sweep.md`)
rerun with the protocol of `g2_start.md`, from one clean commit. Settings of the sweep
(`rabi2_ip`, abs positivity with the softplus start, input 10t, ODE mean reduction,
lr 1e-3, the sweep's field dimensions and time grids), with:

- seeds 100–109;
- two starts per seed: g2 near 1 from the softplus draw (`final_<data>_w<ω>`) and
  g2 = 0.05 (`final_<data>_w<ω>_init005`), g1 from the softplus draw in both;
- per seed, the start with the lower total loss (sum of the unweighted loss terms, each
  averaged over the last 1k epochs) is kept;
- 150k epochs at ω = 15, 10, 7 (near the switch) and ω = 1 (g = ω), 50k elsewhere.

g1 = 1, ωc = ωa = ω, initial state |e,0⟩, t ∈ [0, 2π]. Rabi data have g2 = 1, JC data
g2 = 0.

## The figure

`scripts/make_sweep_figure.py` writes `saved_artifacts/figures/final_g2_sweep.{pdf,png}`
(learned g2 against g/ω, Rabi and JC data side by side) and the per-seed table
`final_sweep.json`. Filled points are each seed's selected start, open grey points the
start it beat. The shaded band is the range of g2 the data cannot tell from the truth:
the values whose exact trajectory stays within a data loss of 2e-3 (the fit floor) of
the training data, on a grid of step 0.02 over [0, 1.5]
(`pinn_jc.sweep.resolution_band`). At g/ω ≤ 0.05 the band covers the whole grid.

## Results

Rabi data (true g2 = 1). "Reached from ≈ 1": runs from the g2 ≈ 1 start ending within
0.05 of the truth. Fidelity against the exact Rabi state, mean over the trajectory.

| ω | g/ω | band | selected g2 | reached from ≈ 1 | fidelity of the selected runs |
|---|---|---|---|---|---|
| 100 | 0.01 | 0–1.5 | 0.000 (10/10) | 0/10 | 1.0000 |
| 20 | 0.05 | 0–1.5 | 0.000 (10/10) | 0/10 | 0.9991 |
| 15 | 0.067 | 0–1.38 | 0.000 (10/10) | 0/10 | 0.9985 |
| 10 | 0.1 | 0.60–1.22 | 0.96–0.98 (3/10); ≤ 0.004 (7/10) | 3/10 (+ one at 0.77) | 1.0000; 0.9966 at g2 = 0 |
| 7 | 0.14 | 0.82–1.14 | 0.989–0.996 (8/10); 0.000 (2/10) | 8/10 | 1.0000; 0.991 at g2 = 0 |
| 5 | 0.2 | 0.92–1.08 | 0.989–0.999 (10/10) | 10/10 | 1.0000 |
| 2 | 0.5 | 1.00 | 0.993–1.003 (10/10) | 10/10 | 1.0000 |
| 1 | 1 | 1.00 | 0.995–0.999 (7/10); see below (3/10) | 7/10 | 1.0000 on 7; 0.76, 0.92, 0.38 |

g1 = 0.998–1.000 on every selected run except at ω = 1.

JC data (true g2 = 0), the null control: the selected g2 is 0.000–0.025 at every ω on
every seed (10/10 × 8), g1 = 0.999–1.000, fidelity 1.0000, ⟨O⟩ MAE against the exact
solution at most 2.2e-3. Both starts end at g2 ≈ 0. The selection takes either (the
0.05 start on 3–7 seeds per ω), so the control does not depend on it.

### What the selection did

- **ω ≥ 15 (g/ω ≤ 0.067):** both starts end at g2 = 0 on every seed. The data do not
  resolve g2 (the band spans the whole grid), and the PINN learns the effective JC
  model. The missing counter-rotating admixture costs 0.1–0.15% of fidelity at ω = 15–20.
- **ω = 5, 2:** the g2 ≈ 1 start reaches the truth on every seed, and the 0.05 start is
  trapped at 0 with 1.6–3.6 times the total loss (14 times on one seed, where g1 collapses too). The selection is right 20/20.
- **ω = 7:** 8/10 right. On the two failing seeds (103, 107) the g2 ≈ 1 start *also*
  falls to g2 = 0, so no selection can help. They are detectable: their data loss is
  3.4e-3 against 0.7–0.9e-3 for the runs that reach the truth, and 0.7e-3 for JC data
  at the same ω.
- **ω = 10:** the switch. The g2 ≈ 1 start reaches 0.96–0.98 on 3 seeds, which win the
  selection (total 4.3–4.8e-3 against 5.3–5.7e-3 at g2 = 0). On 6 seeds it falls to 0.
  On seed 108 it stalls at 0.77: closer to the data (data loss 1.4e-3 against 2.4e-3)
  and the state (fidelity 0.9998 against 0.9966), but with a higher total loss
  (6.1e-3 against 5.9e-3), so the selection keeps the trapped run. With 5 seeds,
  `g2_start.md` found 1/5 at 150k; here 3/10.
- **ω = 1 (g = ω):** the g2 ≈ 1 start reaches the truth on 7/10 seeds. On 3 it stalls
  at g1 ≈ 0.81–0.92, g2 ≈ 0.70–0.77 (fidelity 0.76–0.92, ODE loss 4–6 times that of the
  good runs). The 0.05 start fails on every seed, and here it collapses g1 too
  (g1 = 0.45–0.65 with g2 = 0, once g1 = 0; fidelity 0.18–0.38). The selection keeps the stalled run
  on seeds 104 and 105 and the collapsed one on seed 100, where both starts fail.

### The data loss is a fit check except at ω = 1

At ω = 7 and 10 the runs trapped at g2 = 0 fit the data 3–4 times worse than the runs
at the truth, and worse than the same ω's JC-data runs. A failed seed is visible without
knowing the answer, as with noise (`measurement_noise.md`). At ω = 1 it is not: the good
runs end at a data loss of 3.2e-3 and the failed ones at 2.1–3.9e-3. There only the
ODE loss separates them (ODE term 1.7–1.8e-3 against 7–15e-3).

## Conclusions

1. The claim holds at both ends with 10 seeds and the two-start protocol. Where the
   data cannot resolve the counter-rotating term (g/ω ≤ 0.067) the PINN learns the
   effective JC model on every seed, from either start. Where they resolve it clearly
   (g/ω = 0.2, 0.5) it learns the full model on every seed. The JC-data control is
   passed on all 80 seeds.
2. The switch lies at g/ω ≈ 0.1. There the data resolve g2 only within 0.60–1.22, and
   the PINN returns the full model on 3/10 seeds and JC on 7/10. At 0.14 it is 8/10
   full. This matches `rwa_sweep.md` (switch between 0.067 and 0.1) and its reading: a
   parsimony bias of the method, which prefers the slower JC dynamics until the data
   clearly demand the counter-rotating term.
3. Failures come from the g2 ≈ 1 start falling into the g2 = 0 basin, not from the
   selection. Away from ω = 1 they are flagged by an excess data loss. At g = ω the
   training itself is the limit: 7/10 at 150k epochs.

## Open questions

- At ω = 10 the selection rejects a run closer to the truth (seed 108, g2 = 0.77). The
  total loss is the right criterion only once runs have converged.
- At ω = 1 the stalled runs might converge with longer training; the 0.05 start
  collapsing g1 is new (with 50k epochs, `g2_start.md` saw g1 = 0.47–0.68 too, on 5/5).
- More starts per seed, or ⟨a²⟩ as an extra observable (`g2_start.md`), might raise the
  success rate at ω = 7–10.

## Reproducing

```bash
for c in configs/final/*.yaml; do uv run python scripts/train.py $c --workers 5; done
uv run python scripts/make_sweep_figure.py            # figure + per-seed JSON
uv run python scripts/summarize_runs.py saved_artifacts/runs/final_*   # fidelities
```

The 32 configs took 16.7 h with five workers on an M3 Pro: from 10 min (50k epochs,
ω = 20) to 84 min (150k epochs, ω = 1) per config. Runs from clean commit
f9f1dad. The figure script's selection refuses starts that differ in more than the seed
and `coupling_init`, and outputs from a dirty tree.
