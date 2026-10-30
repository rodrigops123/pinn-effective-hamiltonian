# RWA sweep: does the PINN learn the effective model where the RWA holds?

Configs in `configs/rwa_sweep/`, preparation checks in `scripts/rwa_sweep_preparation.py`.
The paper's claim: the PINN, learning the Rabi model (g1 and g2 free), recovers the
effective Jaynes–Cummings dynamics (g2 → 0) where the rotating-wave approximation holds,
and the full model (g2 ≈ g) where it does not.

Setting: `rabi2_ip`, ωc = ωa = ω, g = 1, initial state |e,0⟩, t in [0, 2π]; Rabi data
(g1 = g2 = 1) and, as the null control, JC data (g1 = 1, g2 = 0) at every ω. Settings
of `positivity_benchmark.md`: abs positivity with the softplus start (g1 and g2 start in
0.69 to 1.31, the full model), input 10t (`g2_bias.md`), ODE mean reduction, lr 1e-3,
50k epochs unless stated, seeds 100–104. One input scale for every ω: tuning it per ω
would tune the method to the expected answer.

## Preparation (no training)

| ω | g/ω | field dim | time steps (points per CR period) | Δ MAE | Δ max |
|---|---|---|---|---|---|
| 100 | 0.01 | 3 | 2001 (10) | 3.0e-5* | 8.9e-5* |
| 20 | 0.05 | 4 | 500 (12) | 9.0e-4 | 2.6e-3 |
| 15 | 0.067 | 4 | 500 (17) | 1.6e-3 | 4.6e-3 |
| 10 | 0.1 | 5 | 500 (25) | 3.7e-3 | 1.1e-2 |
| 7 | 0.14 | 5 | 500 (36) | 7.6e-3 | 2.2e-2 |
| 5 | 0.2 | 6 | 500 (50) | 1.6e-2 | 4.1e-2 |
| 2 | 0.5 | 10 | 500 (125) | 0.21 | 0.65 |
| 1 | 1 | 18 | 500 (250) | 1.1 | 4.3 |

\* At the solver's accuracy: with g2 ≠ 0 the data are integrated in the lab frame, where
the error at ω = 100 is about 4e-5 over [0, 2π] (measured with field dimension 2, where
g2 has no effect from |e,0⟩). Δ at ω = 100 is thus an upper bound, far below the fit
floor either way.

- Field dimension: the smallest that reproduces the Rabi observables of a 40-level field
  to 1e-4 (at least 3, for |e,0⟩ → |g,1⟩ → |e,2⟩).
- Δ: the difference between Rabi and JC data, as the data loss (MAE of ⟨a†a⟩, ⟨σ+σ-⟩,
  the measure of the PINN's fit floor) and as the largest pointwise difference.
- The PINN's fit floor (data loss) is about 1–2e-3 after 50k epochs and 6–8e-4 after
  150k. Against it, the RWA is valid at ω = 100 clearly and at ω = 20 marginally
  (Δ MAE 9e-4); ω = 15 is at the floor, ω = 10 two to three times above it, ω ≤ 7
  clearly above.

Resolution bands, the data loss of the exact trajectory of each g2 against the data, at
a floor of 1e-3:

| ω | JC data: g2 indistinguishable from 0 up to | Rabi data: g2 indistinguishable from 1 down to |
|---|---|---|
| 100 | 1 (whole range) | 0 (whole range) |
| 20 | 1 (whole range) | 0 (whole range) |
| 15 | 0.7 | 0.5 |
| 10 | 0.45 | 0.75 |
| 7 | 0.3 | 0.85 |
| 5 | 0.2 | 0.9 |
| 2 | 0.05 | 0.97 |
| 1 | 0.02 | 0.99 |

## Results

Final g2 (5 seeds each); fidelity and ⟨O⟩ MAE against the exact trajectory of the
data's model, on the training grid; losses are the mean of the last 1k epochs.

| ω | g/ω | Rabi data: g2 | Rabi data: fidelity | JC data: g2 | JC data: fidelity |
|---|---|---|---|---|---|
| 100 | 0.01 | 0.000 (5/5) | 1.0000 | 0.000 (5/5) | 1.0000 |
| 20 | 0.05 | 0.000 (5/5) | 0.9991 | 0.000 (5/5) | 1.0000 |
| 15 | 0.067 | 0.000 (5/5) | 0.9985 | 0.000 (5/5) | 1.0000 |
| 10 | 0.1 | 0.001, 0.009, 0.057, 0.075, 0.551 | 0.9965–0.9991 | 0.003–0.032 | 1.0000 |
| 10, 150k epochs | 0.1 | 0.000–0.004 (4 seeds), **0.965** | 0.9965, **1.0000** | 0.001–0.007 | 1.0000 |
| 7 | 0.14 | 0.972–0.985 (4/5), 0.001 | 1.0000, 0.9909 | 0.001–0.039 | 1.0000 |
| 5 | 0.2 | 0.989–0.995 | 1.0000 | 0.003–0.022 | 1.0000 |
| 2 | 0.5 | 0.995–1.000 | 1.0000 | 0.001–0.003 | 1.0000 |
| 1 | 1 | 0.990–1.001 (3/5) | 0.9999–1.0000 | 0.000–0.001 (4/5) | 0.9961–1.0000 |

g1 = 1.000 ± 0.002 in every run except the trapped ones at ω = 1. The runs come from
clean commits: 93c41e4 (ω = 100, 20, 10, 5, 2, 1), 99c0e91 (ω = 15, 7, ω = 10 at 150k),
52ab312 and f084966 (the g2-fixed checks below).

### Where the RWA holds (ω = 100, 20, 15): the effective model

g2 → 0 within 5k epochs on every seed, with both data: the PINN learns JC from Rabi data.
At ω = 100 the learned state matches the Rabi trajectory with fidelity 1.0000 and
⟨O⟩ MAE 0.9–1.5e-3, the same as from JC data: the effective model reproduces the full
dynamics to the data's precision. At ω = 20 and 15 the fidelity against the Rabi state
is 0.9991 and 0.9985 (the counter-rotating part of the state the effective model
drops), while ⟨O⟩ MAE stays at 1–2e-3.

### Where it does not (ω = 7, 5, 2, 1): the full model

With Rabi data g2 = 0.97–1.00, with JC data g2 ≤ 0.039; the null control is passed at
every ω. Four runs fail, each recognisable by its total loss:

- ω = 7, Rabi data, seed 103: g2 = 0.001, total loss 9.8e-3 against 7.1–7.8e-3 for the
  four runs that find g2 ≈ 1; fidelity 0.9909.
- ω = 1, Rabi data, seeds 100 and 104: trapped at g1 ≈ 0.75, g2 ≈ 0.67, total loss
  2.0–2.3e-2 against 1.3e-2 for the three runs that find g1 = g2 = 1; fidelity 0.73–0.80.
- ω = 1, JC data, seed 102: g1 collapsed to 0 (the near-zero trap of
  `positivity_benchmark.md`), total loss 4.0e-2 against 0.7–2.4e-2.

The fit floor is higher at ω = 1 (data loss 5.7e-3 with Rabi data, against 1–2e-3
elsewhere), and three JC runs (seeds 100, 101, 104) have an ODE loss of 0.5–1.8e-2
and fidelity 0.996–0.999: 50k epochs are not quite enough at this field dimension.

### The boundary (ω = 10): a trap, found with longer training

After 50k epochs, g2 falls on four of five seeds (0.001–0.075, still falling) and stays
at 0.55 on one. The data resolve g2 here (Rabi band g2 ≳ 0.75), so g2 ≈ 0 is not what
the data say: those runs reach a data loss of 2.5–2.6e-3, below the 3.7e-3 of the exact
JC trajectory, by bending the state off the JC dynamics at a cost in the ODE term. At
50k the total losses tie (7.6–8.0e-3), so lowest-loss selection cannot choose.

After 150k epochs the seed at 0.55 reaches g2 = 0.965 with fidelity 1.0000 and a total
loss of 4.3e-3 (data 7.2e-4), against 5.6e-3 (data 2.4e-3) for the four seeds at
g2 ≈ 0. So g2 ≈ 0 is a trap, not a tie: the full model is the lower minimum, but four
of five starts from the full model fall into the JC basin. With enough training,
lowest-loss selection among starts finds the truth.

### Why g2 → 0 in the RWA regime: the cost of the counter-rotating dynamics

Rabi data with g2 fixed at its true value 1 (`rwa_rabi_w*_g2_fixed*`), against the free
runs (which reach g2 ≈ 0):

| ω | epochs | g2 fixed at 1: total (data, ODE) | free, g2 → 0: total (data, ODE) |
|---|---|---|---|
| 10 | 50k | 7.2–8.1e-3 (1.3–1.6e-3, 2.6–2.8e-3) | 7.6–8.0e-3 (2.0–2.6e-3, 2.2–2.6e-3) |
| 15 | 50k | 8.0–8.4e-3 (1.2–1.3e-3, 3.3–3.8e-3) | 6.8–7.4e-3 (1.6–1.8e-3, 2.1–2.3e-3) |
| 15 | 150k | 4.5–4.9e-3 (6.6–7.2e-4, 2.0–2.2e-3) | 4.1–4.7e-3 (1.2–1.3e-3, 1.3–1.4e-3) |
| 20 | 50k | 0.12 on 4 seeds (ODE 0.11), 1.8e-2 on 1 | 6.5–7.1e-3 (1.4–1.6e-3, 1.9–2.2e-3) |

- **ω = 15:** the data favour the full model (data loss about half that of the JC
  solution), but representing the counter-rotating dynamics costs more in the ODE term
  than it gains, so the JC solution has the lower total loss. The margin shrinks with
  training (about 1e-3 at 50k, about 3e-4 at 150k); at ω = 10 it has already flipped.
- **ω = 20:** with input 10t the network cannot represent the full dynamics at all
  (ODE loss 0.11 on four seeds with g2 forced to 1; the supervised fits of the
  preparation give a dψ/dt error of 0.022 at ω = 20, against 0.003–0.005 for ω ≤ 10).
  g2 = 0 is then the only solution the network can fit.

So the PINN learns the effective model where the counter-rotating contribution to the
data is smaller than what it costs the network to represent it. This is an Occam-like
resolution of the method, not only a property of the data, and it depends on the
training budget and on the network's ability to represent fast dynamics (the input
scale).

## The g = ω experiment (`configs/rabi2_g_equals_omega_fock.yaml`, now removed)

That config used a field dimension of 4. At g = ω, ⟨a†a⟩ reaches 4.3, and 4 levels
change the observables by up to 3.3 against a 40-level field (18 levels: 7e-5). Its
earlier result (g1 ≈ g2 ≈ 0.99) was the truncated model's. With 18 levels and its
original settings (softplus, input t, lr 1e-2, 30k epochs) it fails on all five seeds:
g1 = 0.02–0.56, fidelity 0.28–0.64, total loss 0.25–0.42. With the sweep's settings the
same physics is the ω = 1 point above: three of five seeds succeed, the failures
detectable by loss.

## Conclusions

1. **The claim holds at both ends.** At g/ω ≤ 0.067 the PINN learns the effective JC
   model from Rabi data (15/15), and at g/ω ≥ 0.14 it learns the full Rabi model
   (17/20, the three failures detectable by loss). The JC null control is passed at
   every ω (39/40, the failure detectable by loss).
2. **In the RWA regime the data do not decide alone.** At ω ≥ 20 every g2 in [0, 1]
   fits the data below the fit floor, and the network cannot represent the full model;
   at ω = 15 the data favour it slightly, but the ODE cost of the counter-rotating
   terms outweighs that. The PINN's g2 = 0 is a parsimony bias of the method, which
   coincides with the physically motivated effective model. State it as such.
3. **The switch lies between g/ω = 0.067 and 0.1**, where Δ is one to three times the
   fit floor. At g/ω = 0.1 the full model is the lower minimum but most starts fall
   into the JC basin; longer training and lowest-loss selection among starts recover
   it. The switch location depends on the training budget and the input scale.
4. The g = ω result of `rabi2_g_equals_omega_fock` does not hold at a converged field
   dimension with its settings; the config is removed and the sweep's ω = 1 point
   replaces it.

## The legacy experiment `rabi2_jc_fock` (now removed)

JC data at ω = 100 learned with the Rabi model, like `rwa_jc_w100`, but with field
dimension 2. From |e,0⟩ the counter-rotating term only leads to |e,2⟩, which that
truncation removes, so g2 has no effect on the dynamics at all and cannot be learned
(`coherent_squeeze.md`, round 5: g2 = 0.53–0.72 with the recipe). `rwa_jc_w100`
replaces it.

## Open questions

- More seeds (10) for the final figures, and a multi-start protocol (lowest total loss)
  with 150k epochs near the switch and at g = ω.
- Whether ω = 15 flips with even longer training, as ω = 10 did.
- Whether a larger input scale or Fourier features at Σ move the switch (it would make
  the input scale depend on ω, see above).
- A start far from the truth: answered in `g2_start.md`. From g2 near 0 the PINN stays
  at 0 at every ω (the data are even in g2); two starts with lowest-loss selection
  give the right model.

## Reproducing

```bash
uv run python scripts/rwa_sweep_preparation.py --omegas 100 20 15 10 7 5 2 1
uv run python scripts/train.py configs/rwa_sweep/<name>.yaml --workers 5
uv run python scripts/summarize_runs.py saved_artifacts/runs/<name> --trajectory 5000 20000
```

With five workers each 50k-epoch config takes about 5 min at ω = 5–20, 7–8 min at
ω = 2, 13 min at ω = 1 and 15–16 min at ω = 100 (2001 time steps); 150k epochs take
16–19 min at ω = 10 and 15.
