# Why the PINN switches off a visible counter-rotating term (g2 bias)

Configs in `configs/studies/g2bias_*`, diagnostics in `scripts/g2_bias_diagnostics.py`.
The question comes from `legacy_reproduction.md`: at ωc = ωa = 5 (g/ω = 0.2, field
dimension 6) the counter-rotating terms change ⟨σ+σ-⟩ by up to 4e-2, well above the
PINN's fit floor (~5e-3), yet with Rabi data (g1 = g2 = 1) the PINN learns g2 ≈ 0.01,
as with JC data. The paper's claim ("the PINN recovers the effective model, JC, where
the RWA holds and the full model where it does not") needs this case to work.

All runs use the ω = 5 recipe controls (`configs/reproduction/control_*_w5_recipe.yaml`:
`rabi2_ip`, softplus, ODE mean reduction, lr 1e-3, 50k epochs, 500 points on [0, 2π])
unless stated.

## Hypotheses

| | Hypothesis | Verdict |
|---|---|---|
| H1 | The network cannot represent the fast counter-rotating components (frequency Σ = ωa + ωc = 10, amplitude ~g/Σ), so the ODE term favours g2 = 0 | **confirmed, the cause** |
| H2 | The data term is outweighed by the ODE term | not needed once H1 is fixed |
| H3 | The optimisation path: g2 falls before the state learns the fast components | **secondary**: a start near g2 = 0 is a trap once H1 is fixed |
| H4 | The learning frame makes the counter-rotating term the only fast one | not tested; H1's fix suffices |

The time grid is not the issue at ω = 5: 50 points per period of the counter-rotating
terms (it is at ω = 100).

## Diagnostics

### D. The data do not move g2 (existing control runs)

g2 follows the same trajectory with Rabi data (true g2 = 1) and JC data (true g2 = 0),
seed 100: 0.56 / 0.56 at epoch 1k, 0.31 / 0.31 at 2k, 0.105 / 0.108 at 5k, 0.008 / 0.009
at 50k. g2 starts near 1 (softplus of U(0, 1)), so it is actively pushed away from the
truth, not missed.

### C. The ideal landscape favours the truth

The exact trajectory of the model with a given g2 has zero ODE loss; its data loss
against the training data is the profile an ideal PINN would see:

| g2 | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|
| data loss vs Rabi data (true g2 = 1) | 1.6e-2 | 1.5e-2 | 1.4e-2 | 1.0e-2 | 5.2e-3 | 0 |
| data loss vs JC data (true g2 = 0) | 0 | 2.6e-4 | 1.6e-3 | 5.7e-3 | 1.1e-2 | 1.6e-2 |

So the loss is not biased: with an exact state, g2 = 1 wins by 1.6e-2. The JC row also
gives the resolution of g2 at ω = 5: g2 ≲ 0.1–0.15 changes the data by less than the
PINN's fit floor (~6e-4–1.5e-3), so the data cannot tell it from 0.

### B. The network cannot represent the Rabi trajectory (supervised fit)

The PINN's network (same architecture, seed 100) fitted directly to the exact
interaction-picture trajectory, 20k epochs, no physics:

| Fitted to | Network input | State MAE | dψ/dt MAE | ODE loss at g2 = 0 | at g2 = 1 |
|---|---|---|---|---|---|
| JC trajectory | t | 9e-5 | 5e-4 | **5e-4** | 0.075 |
| Rabi trajectory | t | 1e-2 | 9.6e-2 | **0.015** | 0.089 |
| Rabi trajectory | 3t | 5e-4 | 9.0e-3 | 0.091 | **0.009** |
| Rabi trajectory | 10t | 1e-4 | 3.5e-3 | 0.091 | **0.003** |
| Rabi trajectory | t + Fourier features at Σ | 4e-4 | 7.4e-3 | 0.093 | **0.007** |

With input t, even told the answer, the network's time derivative misses the
counter-rotating part entirely (its error, 0.096, is the size of the whole term); for
the smoothed trajectory it can represent, the ODE loss is smallest at g2 = 0. This is
spectral bias. Scaling the input (as ω0 does in SIREN) or Fourier features fix the fit
and flip the ODE landscape to favour the truth.

### A. The PINN's profile is inverted (g2 fixed, g1 and the state learned)

| g2 fixed at | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| total loss, Rabi data | **1.4e-2** | 3.5e-2 | 5.4e-2 | 7.3e-2 | 9.2e-2 |
| total loss, JC data | **7.6e-3** | 2.6e-2 | 4.5e-2 | 6.4e-2 | 8.2e-2 |

Mean of seeds 100 and 101; at g2 = 1 seed 101 only, because seed 100 collapsed to
g1 ≈ 0.01 with both data. With Rabi data and g2 from 0.25 to 1 the learned state is the same
(fidelity 0.9853, data loss 1.03e-2): the network does not follow g2, and the ODE loss
grows linearly in it. The minimum is at g2 = 0 for both data, the opposite of C. No
choice of optimisation path can reach a minimum the network cannot represent: H1 is
the cause, H3 alone cannot be.

## The fix: `model.input_scale`

The networks see `input_scale` × t (`MLP`, default 1, no extra parameters). Five seeds
per row, final g2:

| Input | Rabi data, g2 from ≈ 1 | JC data, g2 from ≈ 1 | Rabi, from 0.05 | JC, from 0.05 | `rabi2_g_equals_omega_fock` (g1, g2) |
|---|---|---|---|---|---|
| t (before) | 0.008, 0.009 (2 seeds) | 0.008, 0.009 (2 seeds) | – | – | ≈ 0.99, 0.99 |
| 3t | 0.011–0.030 ✗ | 0.010–0.028 | ≈ 0.001 ✗ | ≈ 0.001 | 0.99, 0.99–1.00 |
| 10t | **0.987–0.997** | 0.125–0.151 (falling) | 0.010–0.016 ✗ | 0.010–0.015 | 0.99, 0.99–1.00 |
| 10t, from 0.3 | **0.986–0.996** | | | | |
| 10t, from 0.5 | **0.986–0.995** | | | | |
| 10t, 150k epochs | | 0.091–0.100 (falling) | | | |

With input 10t and Rabi data started at g2 ≥ 0.3, all 15 runs learn g2 = 0.99 ± 0.01
with g1 = 1.000, fidelity 1.0000 and expectation-value MAE 1.0–1.7e-3, the level of the
JC case: the counter-rotating term is learned when it is real and visible. Input 3t
fixes the supervised fit but not the PINN; 10t is needed.

Two caveats remain.

1. **A start near g2 = 0 is a trap.** From g2 = 0.05 the PINN learns g2 ≈ 0.01 with
   Rabi data. The basin boundary is between 0.05 and 0.3. The trap is recognisable: its
   total loss is 1.25e-2 against 7e-3 for the runs that find g2 = 1, and its fidelity
   0.982 against 1.0000. Selecting the lowest-loss run among starts chooses correctly.
2. **With JC data g2 decays slowly.** From g2 ≈ 1: 0.33 at 5k epochs, 0.16 at 20k,
   0.125 at 50k, 0.103 at 100k, 0.092 at 150k, the same on all seeds. Diagnostic C
   explains it: g2 ≈ 0.1 changes the data by 2.6e-4, below the fit floor (6e-4 at 150k),
   so only the weak ODE pressure pushes it further. g2 ≈ 0.1 is within what the data
   resolve at ω = 5, not a wrong answer, but a strict "g2 < 0.05" is not met.

## Conclusions

1. The g2 bias of the legacy settings is spectral bias of the network, not a property
   of the loss or of the method: the exact landscape favours the truth, and a network
   that can represent the counter-rotating dynamics learns it.
2. With `input_scale: 10` and the recipe, the PINN learns g2 ≈ 1 from Rabi data and
   drives g2 down towards 0 from JC data at ω = 5, while the g = ω experiment is
   unaffected (that config used an unconverged field dimension, 4, and has since been
   removed; see `rwa_sweep.md`). This is the behaviour the paper's claim needs.
3. The result depends on the start: from g2 near 0 the PINN stays there. The protocol
   must say how g2 starts (decided in `g2_start.md`: the second option below, after
   the trap turned out to hold at every ω for a symmetry reason). Two defensible
   options:
   - **start from the full model** (all couplings near 1) and let the data switch off
     what they do not support; this matches the claim and the default initialisation;
   - **multi-start** (e.g. g2 from 0.05 and from 1) and keep the run with the lowest
     total loss, which picks the right basin in every case here.
4. Report g2 with its resolution: the exact data-loss profile (diagnostic C) gives the
   range of g2 the data cannot distinguish from 0 at the PINN's fit floor (here
   ≈ 0.1–0.15). A learned g2 inside that range means "no counter-rotating term
   resolved".

## Open questions for the sweep

Followed up in `rwa_sweep.md` (one input scale, 10, for every ω; the grid at ω = 100)
and `positivity_benchmark.md` (the slow decay with JC data).


- Input scale per regime: 10t was chosen for Σ = 10. A fixed scale relative to the
  fastest frequency (e.g. input_scale ≈ Σ) or Fourier features at Σ would carry over to
  other ω; the supervised fit (diagnostic B) is a cheap test before each sweep point.
- At ω = 100 the grid also has to resolve Σ = 200 (≥ 10 points per period: Δt ≲ 3e-3).
- Whether the slow decay of g2 with JC data can be sped up (cosine learning-rate
  schedule, longer training) without changing the result.

## Reproducing

```bash
uv run python scripts/g2_bias_diagnostics.py configs/reproduction/control_rabi_data_w5_recipe.yaml configs/reproduction/control_jc_data_w5_recipe.yaml
uv run python scripts/train.py configs/studies/g2bias_<name>.yaml --workers 5
```

Each 50k-epoch run takes about 6 min per seed on one core; the diagnostics script about
2 min.
