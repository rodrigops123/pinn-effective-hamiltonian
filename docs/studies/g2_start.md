# Starting g2 far from the truth

Configs in `configs/studies/g2start_*`. The RWA sweep (`rwa_sweep.md`) starts g2 near 1,
the full model. The question here: what happens when g2 starts near 0 instead, and can
the training be changed so that the result does not depend on the start? Settings of
the sweep (`rabi2_ip`, abs positivity, input 10t, ODE mean reduction, lr 1e-3, 50k
epochs, seeds 100–104), with `physics.coupling_init: {g2: 0.05}`; g1 starts from the
softplus draw as in the sweep.

## Results: g2 started at 0.05

Rabi data (true g2 = 1). Total loss is the mean of the last 1k epochs; the column "from
≈ 1" repeats the sweep's runs with the same seeds and settings.

| ω | g/ω | final g2 | fidelity | total loss, from 0.05 | total loss, from ≈ 1 |
|---|---|---|---|---|---|
| 100 | 0.01 | 0.000 | 1.0000 | 6.5–7.1e-3 | 6.4–6.9e-3 (also g2 = 0) |
| 20 | 0.05 | 0.000 | 0.9991 | 6.6–6.8e-3 | 6.5–7.1e-3 (also g2 = 0) |
| 15 | 0.067 | 0.000 | 0.9985 | 6.6–7.1e-3 | 6.8–7.4e-3 (also g2 = 0) |
| 10 | 0.1 | 0.000 | 0.9965 | 7.8–8.3e-3 | 7.6–8.0e-3 (g2 = 0.55 on one seed) |
| 7 | 0.14 | 0.000–0.001 | 0.9908–0.9910 | 0.93–1.0e-2 | 7.1–7.8e-3 (g2 ≈ 0.98, 4/5) |
| 5 | 0.2 | 0.000–0.002 | 0.981–0.982 | 1.2–1.3e-2 | 6.5–8.1e-3 (g2 ≈ 0.99) |
| 2 | 0.5 | 0.000–0.001 | 0.73–0.78 | 2.6–3.0e-2 | 7.1–8.3e-3 (g2 ≈ 1.00) |
| 1 | 1 | 0.000–0.003, g1 = 0.47–0.68 | 0.35–0.38 | 1.7–2.3e-2 | 1.3e-2 (g2 ≈ 1.00, 3/5) |

From 0.05 the PINN learns g2 ≈ 0 at every ω, on every seed: a start near 0 is a trap
everywhere, not only at ω = 5 (`g2_bias.md`). JC data from 0.05 (ω = 1, 5, 10) give
g2 = 0.000–0.002 with fidelity 0.9992–1.0000 and no collapse of g1 (one seed collapsed
from the ≈ 1 start at ω = 1).

The trap is visible in the total loss wherever the counter-rotating term is resolved
(ω ≤ 7): it is 1.2–3.5 times that of the runs that find the truth. At ω = 10 the two
tie after 50k epochs; after 150k the full model is lower (4.3e-3 against 5.6e-3,
`rwa_sweep.md`).

## Why: g2 = 0 is a stationary point of the data loss

The rotation a → i a, σ+ → -i σ+ leaves the JC term g1 (a†σ- + a σ+), the initial
state |e,0⟩ and both observables (⟨a†a⟩, ⟨σ+σ-⟩) unchanged, and flips the sign of the
counter-rotating term g2 (a σ- + a†σ+). So the data are even in g2: the simulated
observables for g2 = ±0.3 agree exactly (difference 0.0), and near 0 the data change
as g2² (at ω = 5: 0.026 g2², so 6.5e-5 at g2 = 0.05 and 2.6e-4 at 0.1). The gradient
of the data loss with respect to g2 vanishes at g2 = 0, while the ODE term favours the
slower JC dynamics (`g2_bias.md`). Any gradient method started near 0 sees almost no
pull towards the truth. This is a property of the inverse problem with these
observables and this initial state, not of the optimiser.

## Attempted remedies (g2 from 0.05, Rabi data)

| Remedy | ω = 2 | ω = 5 | ω = 10 |
|---|---|---|---|
| none | 0 / 5 | 0 / 5 | 0 / 5 |
| cosine learning rate, 1e-3 → 1e-5 (`*_cosine`) | | 0 / 5 | 0 / 5 |
| growing time window, 10% → 100% over 25k epochs (`*_window`) | | 0 / 5 | 0 / 5 |
| couplings frozen for 10k epochs (`*_freeze`, `training.freeze_couplings_epochs`) | 1 / 5 | 0 / 5 | 0 / 5 |

Runs reaching g2 ≈ 1. With the couplings frozen, the state fits the data first; once
released, g2 falls from 0.05 to ≈ 0.02 within 5k epochs on 14 of 15 runs, the ODE
term's pull outweighing the data's. The one escape (ω = 2, seed 100) climbs to 0.99 by
epoch 20k.

The cosine schedule lowers the loss floor of every run (total 2.7e-3 at ω = 10 against
about 8e-3 at a constant rate): **total losses are comparable only between runs with
the same training settings**, a condition for any lowest-loss selection.

## Protocol for the paper: two starts, lowest total loss

Since no change to the training removes the trap, the start is part of the method:

1. Train from the full model (g2 near 1, the sweep's start) and from the effective one
   (g2 = 0.05), with the same settings and seeds.
2. Keep, per seed, the run with the lower total loss.

On the runs above this selects the truth at every ω, with one caveat:

| ω | from ≈ 1 | from 0.05 | selected |
|---|---|---|---|
| ≥ 15 | g2 = 0 | g2 = 0 | JC (effective model) |
| 10 | g2 ≈ 0 (4/5) or 0.55 | g2 = 0 | tie after 50k; the full model after 150k (1/5 seeds) |
| ≤ 7 | g2 ≈ 1 | g2 = 0, higher loss | full model, except the seeds failing from both starts |

The two-start protocol makes the conclusion independent of the start, but not of the
training length near the switch: at ω = 10 the run has to be long enough for the loss
to separate the two models, and most starts from the full model still fall into the JC
basin there.

## Conclusions

1. A start near g2 = 0 is a trap at every ω, from a symmetry of the problem: the data
   loss is even in g2 and flat at 0. It is not fixed by the learning-rate schedule, a
   growing time window or freezing the couplings while the state trains.
2. Where the counter-rotating term is resolved, the trap has a clearly higher total
   loss, so a two-start protocol with lowest-loss selection recovers the right model.
   This should be the paper's protocol, stated with its reason.
3. Near the switch (ω = 10), 150k epochs are needed for the losses to separate.

## Open questions

- An observable odd under the rotation would give the data a gradient at g2 = 0 and
  might remove the trap. With |e,0⟩, parity conservation keeps ⟨a + a†⟩ and ⟨σx⟩ at 0,
  so they do not help, but ⟨a²⟩ does: it changes sign with g2 (at ω = 5, g2 = ±0.3,
  |⟨a²⟩| up to 4e-2). An initial state without definite parity is the other route.
  Either changes the experiment and belongs to the backlog.
- More starts (e.g. g2 in {0.05, 0.3, 0.5, 1}) to map the basin boundary per ω.

## Reproducing

```bash
uv run python scripts/train.py configs/studies/g2start_<name>.yaml --workers 5
uv run python scripts/summarize_runs.py saved_artifacts/runs/g2start_<name>
uv run python scripts/g2_symmetry_check.py --omega 5 --field 6   # the symmetry
```

Each 50k-epoch config takes about 5 min with five workers at ω = 5–20, 7 min at ω = 2,
13 min at ω = 1 and 16 min at ω = 100. Runs from clean commits 24e7403 (all but the
freeze variants) and 3235322 (the freeze variants).
