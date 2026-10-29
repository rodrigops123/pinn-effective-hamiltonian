# RWA sweep: does the PINN learn the effective model where the RWA holds?

Configs in `configs/rwa_sweep/`, preparation checks in `scripts/rwa_sweep_preparation.py`.
The paper's claim: the PINN, learning the Rabi model (g1 and g2 free), recovers the
effective Jaynes–Cummings dynamics (g2 → 0) where the rotating-wave approximation holds,
and the full model (g2 ≈ g) where it does not.

Setting: `rabi2_ip`, ωc = ωa = ω, g = 1, initial state |e,0⟩, t in [0, 2π]; Rabi data
(g1 = g2 = 1) and, as the null control, JC data (g1 = 1, g2 = 0) at every ω. Settings
of `positivity_benchmark.md`: abs positivity with the softplus start (g1 and g2 start in
0.69 to 1.31, the full model), input 10t (`g2_bias.md`), ODE mean reduction, lr 1e-3,
50k epochs, seeds 100–104. One input scale for every ω: tuning it per ω would tune the
method to the expected answer.

## Preparation (no training)

| ω | g/ω | field dim | time steps (points per CR period) | Δ MAE | Δ max |
|---|---|---|---|---|---|
| 100 | 0.01 | 3 | 2001 (10) | 3.0e-5 | 8.9e-5 |
| 20 | 0.05 | 4 | 500 (12) | 9.0e-4 | 2.6e-3 |
| 10 | 0.1 | 5 | 500 (25) | 3.7e-3 | 1.1e-2 |
| 5 | 0.2 | 6 | 500 (50) | 1.6e-2 | 4.1e-2 |
| 2 | 0.5 | 10 | 500 (125) | 0.21 | 0.65 |
| 1 | 1 | 18 | 500 (250) | 1.1 | 4.3 |

- Field dimension: the smallest that reproduces the Rabi observables of a 40-level field
  to 1e-4 (at least 3, for |e,0⟩ → |g,1⟩ → |e,2⟩).
- Δ: the difference between Rabi and JC data, as the data loss (MAE of ⟨a†a⟩, ⟨σ+σ-⟩,
  the measure of the PINN's fit floor) and as the largest pointwise difference.
- **"RWA valid" means Δ below the fit floor**, about 1–2e-3 here: ω = 100 clearly,
  ω = 20 marginally (Δ MAE 9e-4); ω ≤ 5 clearly not; ω = 10 is at the boundary (Δ MAE
  3.7e-3, two to three times the floor).

Resolution bands, the data loss of the exact trajectory of each g2 against the data:

| ω | JC data: g2 indistinguishable from 0 up to | Rabi data: g2 indistinguishable from 1 down to |
|---|---|---|
| 100 | 1 (whole range) | 0 (whole range) |
| 20 | 1 (whole range) | 0 (whole range) |
| 10 | 0.45 | 0.75 |
| 5 | 0.2 | 0.9 |
| 2 | 0.05 | 0.97 |
| 1 | 0.02 | 0.99 |

At ω ≥ 20 the data do not determine g2 at all; what the PINN learns there is decided by
the method (the ODE term and the network's implicit bias), not by the data. The paper
must say so: "the PINN learns the effective model" there means "of the models that fit
the data equally, the PINN picks the one without the counter-rotating term".

## Results

Final g1, g2 (5 seeds each); fidelity and ⟨O⟩ MAE against the exact trajectory of the
data's model, on the training grid; losses are the mean of the last 1k epochs.

| ω | g/ω | Rabi data: g2 | Rabi data: fidelity | JC data: g2 | JC data: fidelity | runs off target |
|---|---|---|---|---|---|---|
| 100 | 0.01 | 0.000 (5/5) | 1.0000 | 0.000 (5/5) | 1.0000 | 0 / 10 |
| 20 | 0.05 | 0.000 (5/5) | 0.9991 | 0.000 (5/5) | 1.0000 | 0 / 10 |
| 10 | 0.1 | 0.001, 0.009, 0.057, 0.075, 0.551 | 0.9965–0.9991 | 0.003–0.032 | 1.0000 | see below |
| 5 | 0.2 | 0.989–0.995 | 1.0000 | 0.003–0.022 | 1.0000 | 0 / 10 |
| 2 | 0.5 | 0.995–1.000 | 1.0000 | 0.001–0.003 | 1.0000 | 0 / 10 |
| 1 | 1 | 0.990–1.001 (3/5) | 0.9999–1.0000 | 0.000–0.001 (4/5) | 0.9961–1.0000 | 3 / 10, all detectable |

g1 = 1.000 ± 0.002 in every run except the trapped ones at ω = 1. All 60 runs come from
the clean commit 93c41e4.

### Where the RWA holds (ω = 100, 20): the effective model

g2 → 0 within 5k epochs on every seed, with both data: the PINN learns JC from Rabi data.
At ω = 100 the learned state matches the Rabi trajectory with fidelity 1.0000 and
⟨O⟩ MAE 0.9–1.5e-3, the same as from JC data: the effective model reproduces the full
dynamics to the data's precision. At ω = 20 the fidelity against the Rabi state is
0.9991 (the counter-rotating part of the state the effective model drops), while
⟨O⟩ MAE stays at 1.0–1.4e-3.

### Where it does not (ω = 5, 2, 1): the full model

With Rabi data g2 = 0.99–1.00, with JC data g2 ≤ 0.022, on every seed at ω = 5 and 2;
the null control is passed at every ω. At ω = 1 (g = ω, field dimension 18) three
runs fail, each recognisable by its total loss:

- Rabi data, seeds 100 and 104: trapped at g1 ≈ 0.75, g2 ≈ 0.67, total loss 2.0–2.3e-2
  against 1.3e-2 for the three runs that find g1 = g2 = 1; fidelity 0.73–0.80.
- JC data, seed 102: g1 collapsed to 0 (the near-zero trap of `positivity_benchmark.md`),
  total loss 4.0e-2 against 0.7–2.4e-2.

The fit floor is higher at ω = 1 (data loss 5.7e-3 with Rabi data, against 1–2e-3
elsewhere), and three JC runs (seeds 100, 101, 104) have an ODE loss of 0.5–1.8e-2
and fidelity 0.996–0.999: 50k epochs are not quite enough at this field dimension.

### The boundary (ω = 10): seed-dependent, biased towards JC

With Rabi data, g2 falls on four of five seeds (0.001–0.075, still falling at 50k) and
stays at 0.55 on one. The data resolve g2 here (the Rabi resolution band is
g2 ≳ 0.75), so g2 ≈ 0 is not what the data say. The PINN compensates: its data loss,
2.5–2.6e-3, is below the 3.7e-3 of the exact JC trajectory, because the state bends off
the JC dynamics at a cost in the ODE term. The seed with g2 = 0.55 has the lowest data
loss (2.0e-3) and fidelity (0.9991 against 0.9965), but its total loss (7.6e-3) ties
with a seed at g2 = 0.057: lowest-loss selection does not pick the truth here.

This is the method's bias of `g2_bias.md` reappearing where Δ is only two to three times
the fit floor: with Σ = 20 the counter-rotating amplitude g/Σ = 0.05 is small, and the ODE
term prefers the slower JC dynamics. The JC control at ω = 10 is passed (g2 ≤ 0.032).

## Conclusions

1. **The claim holds at both ends.** At g/ω ≤ 0.05 the PINN learns the effective JC
   model from Rabi data (10/10), and at g/ω ≥ 0.2 it learns the full Rabi model
   (13/15, the two failures at g = ω detectable by loss). The JC null control is passed
   at every ω (29/30, the failure detectable by loss).
2. **In the RWA regime the data do not decide.** At ω ≥ 20 every g2 in [0, 1] fits the
   data to below the fit floor; the PINN's choice of g2 = 0 is a parsimony bias of the
   method (ODE term plus the network's spectral bias), which happens to coincide with
   the physically motivated effective model. This should be stated as such.
3. **The switch happens around g/ω = 0.1, where Δ is two to three times the fit floor**, and
   there the PINN errs towards JC: the transition is shifted to stronger coupling
   compared with what the data alone resolve. Reported honestly, the transition
   location is a property of the PINN's fit floor and bias, not only of the physics.
4. g = ω needs either more epochs or a multi-start (lowest total loss) protocol.

## Open questions

- Resolve the transition: ω = 7, 15 (g/ω ≈ 0.14, 0.067), and longer training at ω = 10
  (g2 was still falling on four seeds).
- More seeds (10) for the final figures, and a multi-start protocol for g = ω.
- Whether a larger input scale or Fourier features at Σ move the switch at ω = 10
  towards what the data resolve (it would make the input scale depend on ω, see above).
- A robust start far from the truth, e.g. g2 from near 0 (backlog).

## Reproducing

```bash
uv run python scripts/rwa_sweep_preparation.py --omegas 100 20 10 5 2 1
uv run python scripts/train.py configs/rwa_sweep/rwa_<rabi|jc>_w<ω>.yaml --workers 5
```

With five workers each config takes about 5 min at ω = 5, 10, 20, 7–8 min at ω = 2,
13 min at ω = 1 and 15–16 min at ω = 100 (2001 time steps).
