# Reproduction of the legacy experiments

Configs in `configs/reproduction/`. The JC experiments (Fock, Fock superposition,
coherent) and the Rabi-loss experiments (Rabi and JC data learned with the Rabi model)
of the legacy code were rerun with their original parameters and the legacy settings
(field truncation, ODE loss summed over components), two seeds each. Seeds 42, 43 (JC)
and 100, 101 (Rabi loss) are the first two seeds of the legacy runs, and `build_models`
keeps the legacy RNG stream, so the two can be compared run by run. On top of that, a
control (`control_*_w5*`) tests the Rabi-loss claim in a regime where the
counter-rotating terms are visible in the data.

## Results

g1 and g2 are the final learned couplings; fidelity is |⟨ψ|ψ_nn⟩|²/‖ψ_nn‖² on the
training grid; ⟨O⟩ MAE is the mean error of ⟨a†a⟩ and ⟨σ+σ-⟩.

| Experiment | Seed | g1 | g2 | Fidelity mean (min) | ⟨O⟩ MAE |
|---|---|---|---|---|---|
| `legacy_jc_fock` | 42 | 0.184 | – | 0.46 (0.00) | 0.39 |
| | 43 | **0.9999** | – | 1.0000 (1.0000) | 1.6e-3 |
| `legacy_jc_fock_superposition` | 42 | 0.164 | – | 0.26 (0.00) | 0.17 |
| | 43 | 0.322 | – | 0.23 (0.00) | 0.21 |
| `legacy_jc_coherent` | 42 | 0.048 | – | 0.21 (0.01) | 0.12 |
| | 43 | 0.048 | – | 0.21 (0.01) | 0.12 |
| `legacy_rabi_data_abs` (true g2 = 1) | 100 | 0.003 | 0.000 | 0.53 (0.00) | 0.30 |
| | 101 | 0.430 | 0.000 | 0.56 (0.00) | 0.37 |
| `legacy_rabi_data_softplus` (true g2 = 1) | 100 | **0.996** | 0.155 | 0.9999 (0.9997) | 4.9e-3 |
| | 101 | **0.998** | 0.146 | 1.0000 (0.9999) | 5.6e-3 |
| `legacy_jc_data_abs` (true g2 = 0) | 100 | 0.000 | 0.000 | 0.53 (0.00) | 0.30 |
| | 101 | 0.426 | 0.000 | 0.56 (0.00) | 0.37 |
| `legacy_jc_data_softplus` (true g2 = 0) | 100 | **0.999** | 0.156 | 0.9999 (0.9997) | 5.4e-3 |
| | 101 | **0.999** | 0.149 | 1.0000 (0.9999) | 5.7e-3 |
| `control_rabi_data_w5_recipe` (ω = 5, true g2 = 1) | 100 | **0.991** | 0.008 | 0.982 (0.963) | 6.5e-3 |
| | 101 | **0.999** | 0.009 | 0.978 (0.929) | 6.6e-3 |
| `control_jc_data_w5_recipe` (ω = 5, true g2 = 0) | 100 | **1.000** | 0.009 | 1.0000 (1.0000) | 1.8e-3 |
| | 101 | **1.000** | 0.008 | 1.0000 (1.0000) | 2.8e-3 |

The controls with the legacy settings (`control_*_w5`, abs; `control_*_w5_softplus`)
collapsed on all eight runs (g1 ≤ 0.39, ⟨O⟩ MAE ≈ 0.35); the `_recipe` versions use
the ODE mean reduction and learning rate 1e-3 from `coherent_squeeze.md`.

The legacy pickles (`saved_artifacts/runs/results_dict_*`) give the success rate of the
original runs, counting g1 within 1% of the truth:

| Legacy result | Converged runs |
|---|---|
| Fock, [50, 50, 50] | 4 / 10 (the others: g1 ≈ 0.244) |
| Fock superposition, [50, 50, 50] | 6 / 10 (g1 ≈ 0.17 or 0) |
| Coherent, [50, 50, 50] | 1 / 15, at g1 = 0.995 (g1 ≈ 0.09 or 0.61) |
| Coherent, [100, 100, 100] | 1 / 10 (g1 ≈ 0.06 or 0.55) |
| Rabi loss, abs (`rabi2_fock`, `rabi2_fock_jc`) | 2 / 5 each (g1 ≈ 0, 0.45) |
| Rabi loss, softplus | 4 / 5 each (g2 = 0.06–0.10 at 50k epochs) |

The reruns land in the same basins: Fock seed 42 fails and 43 converges as legacy
runs 0 and 1 did; with abs, Rabi-loss seeds 100 and 101 give g1 ≈ 0 and ≈ 0.43 as
legacy runs 0 and 1 (0.0007, 0.45). Fock superposition seed 42 fails here but converged
in the legacy run, so the legacy notebook did not use exactly the recorded parameters
(most likely a different learning rate).

## Comparison with the legacy figures

- **Fock.** Seed 43 reproduces them: loss hierarchy L_eq > L_norm ≈ L_data > L_ic,
  fidelity above 0.99999, observable errors ~1e-3, g1 = 0.9999. Seed 42 is a
  self-consistent slower Rabi oscillation (g1 = 0.18): low ODE loss (2e-3), data loss
  0.39. The legacy figures show the successful case only.
- **Fock superposition.** Not reproduced on these two seeds.
- **Coherent.** Not reproduced: both seeds fit an almost static state (g1 = 0.05). Even
  the legacy figure has fidelity 0.965–0.985, 0.97 at t = 0, where the initial condition
  is imposed; that is a 3% state error, not "close to unity".
- **Rabi data and JC data.** The softplus runs reproduce the legacy coupling curves
  closely: g1 dips to ~0.55, then locks at 1; g2 decays monotonically and is still 0.15
  at 50k epochs. With abs both seeds fail, as 3 of 5 legacy runs did. **The Rabi-data
  and JC-data runs give the same couplings to three decimals** (next section explains
  why).
- The legacy coupling plot for Rabi data draws the true counter-rotating parameter at 0
  although the data have g2 = 1, and the Rabi-data and JC-data plots are near copies.
  One legacy caption gives the initial state as |g⟩ ⊗ |1⟩ instead of |e⟩ ⊗ |0⟩.

## Challenges to the conclusions

### 1. In the Rabi-loss experiments, the data cannot tell Rabi from Jaynes-Cummings

With cavity dimension 2 and |e,0⟩, the dynamics stays in {|e,0⟩, |g,1⟩}. The
counter-rotating operators aσ- and a†σ+ have no matrix element inside it (a†σ+|g,1⟩ =
√2|e,2⟩ is truncated away), so **Rabi data with g2 = 1 and JC data are identical**.
`simulate` gives a 4e-5 difference, which is the integrator's error in the lab frame.
Even with an untruncated field, at ω = 100 and t ≤ 2π the difference is ~1e-4 in the
observables (table below), 50 times below the PINN's best data fit (5e-3).

| ωc = ωa | max \|Δ⟨O⟩\| (Rabi g2 = 1 vs JC), field dim 6 |
|---|---|
| 100 | 1.1e-4 |
| 20 | 2.6e-3 |
| 10 | 1.1e-2 |
| 5 | 4.1e-2 |
| 2 | 6.5e-1 |

So the Rabi-data and JC-data experiments are the same experiment, and the g2 the PINN
reports there says nothing about the data. In dimension 2, g2 multiplies only
components the true state never populates; it acts as a penalty on the network's
leakage into them and decays at a rate set by the learning rate: to exactly 0 with abs
(kink), slowly with softplus (0.15 here, 0.08 in the legacy runs after the same number
of epochs).

### 2. The PINN switches g2 off even when it is real and visible

At ω = 5 the counter-rotating terms change ⟨σ+σ-⟩ by up to 4e-2: its minima reach ~0.03
instead of 0. With true g2 = 1, both seeds learn **g2 = 0.008**, the same value as with
JC data. They pay for it: ⟨O⟩ MAE 6.6e-3 against 1.8e-3 with JC data, systematic errors
up to 0.025 at the minima of ⟨σ+σ-⟩, fidelity down to 0.93. The ODE residual of a smooth
network is smallest when the e^{±iΣt} terms are switched off; the data term is too weak
to overrule it. The ability to "switch off unsupported terms" is therefore not shown: the
method switches off the counter-rotating term whether it is supported or not, as long as
its effect is perturbative. When it is not perturbative (g = ω = 1,
`rabi2_g_equals_omega_fock`, round 5 of `coherent_squeeze.md`), g2 is learned (0.99 on 4
seeds).

### 3. Success depends on the seed, and the legacy results are single runs

With the legacy settings, 5 of the 14 runs here converge, and the legacy pickles give
1/15 (coherent), 4/10 (Fock), 6/10 (superposition), 2/5 (Rabi, abs). The legacy figures
show converged runs without saying so. The failures are not noise around the truth but
distinct wrong solutions (g1 ≈ 0.24, 0.17, 0.09, 0.61, 0.45 recur across seeds); a
converged run is easy to recognize afterwards by its data loss (1e-3 vs 1e-1).

### 4. "abs gives values closer to the effective model"

Of the legacy conclusions, this one is reversed by the data: with abs, 4 of 4 runs here
and 6 of 10 legacy runs collapse g1 to a wrong value; softplus converges on 4 of 4 here
and 8 of 10 legacy runs. abs only gives g2 = 0 exactly, which (points 1 and 2) is not
evidence for anything. The kink explanation is plausible for g2, but the dominant effect
of abs is on g1.

### 5. Smaller points

- The JC experiments use ωa = ωc = 100, but on resonance the JC interaction picture does
  not contain ω, so these experiments do not test anything about ω ≫ g.
- In Fock dimension 2, ⟨a†a⟩ + ⟨σ+σ-⟩ = 1 and the data are cos²(gt): g is read off the
  period directly. The experiment checks the code, not the method.
- L_eq has a higher floor because it is summed over D components and has the units of
  dψ/dt, not because "the regularization of the couplings is accounted for within this
  loss term": there is no regularization term. Loss values of different terms are not
  comparable in absolute value.
- Fidelity above 1 comes from not normalizing ψ_nn. Normalized fidelity (as computed
  here) removes it; the excess measures the norm error, which is part of the result.
- The field truncation leaves |e, n_max⟩ as a spurious stationary state
  (`physics/basis.py`); it is one of the identified causes of the coherent-state failures.

## What the results support

1. A PINN with a JC ODE loss and two observables can recover g1 and the full state
   (fidelity > 0.9999) for low-dimensional initial states, **when training converges**;
   with the legacy settings this happens on a minority to a majority of seeds depending
   on the state, and on 1 in 10–15 seeds for the coherent state.
2. Convergence is the bottleneck, and it is fixable: excitation-number truncation, the
   ODE loss averaged over components and lr 1e-3 solve the coherent state on 4/4 seeds
   and reduce the g1 error 5× for the Fock states (`coherent_squeeze.md`, rounds 2 and 4),
   and rescue the ω = 5 controls here (4/4 vs 0/8).
3. With the split Rabi ODE, g1 is learned in every converged run. g2 is learned when the
   counter-rotating dynamics is large (g = ω); when it is perturbative the PINN sets g2
   ≈ 0 **independently of the true value**. This is an effective-model *bias* of the
   method, not model selection from data.
4. The positivity map changes convergence much more than it changes g2: softplus is the
   safer choice.

## What needs refining

- Report success rates (or all seeds) and select runs by a stated criterion, such as the
  final data loss; 10+ seeds for the headline experiments.
- Redesign the Rabi-loss experiments: field dimension ≥ 3 (better 6), a range of ω/g
  (2, 5, 10, 20, 100) so that g2's effect on the data runs from dominant to below the fit
  floor, and both true g2 = 0 and g2 = 1 at each point. The claim worth testing is "the
  learned g2 follows the truth where the data resolve it", which point 2 currently
  contradicts.
- To make g2 learnable in the perturbative regime: more weight on L_data (or adaptive
  weighting), collocation points resolving 2π/Σ, Fourier features at Σ in the network,
  or learning in a frame where the counter-rotating term is slow. A profile of the data
  loss with g2 fixed at several values would show directly whether the data or the ODE
  term decides g2.
- Use the recipe settings (excitation truncation, ODE mean, lr 1e-3) for all JC
  experiments, and state the coherent-state limitation (fixed by them) and the squeezed
  vacuum (still unsolved) explicitly.
- Fix the true-g2 line and the initial-state caption of the legacy plots; use the
  normalized fidelity.

## Reproducing

```bash
uv run python scripts/train.py configs/reproduction/<name>.yaml --workers 2
uv run python scripts/make_figures.py saved_artifacts/runs/<name>/<run> --formats png
```

Runs take 2.5 min (Fock, 30k epochs), 8 min (Rabi loss, 50k) and 9 min (coherent and
controls, 50k) per seed on one core.
