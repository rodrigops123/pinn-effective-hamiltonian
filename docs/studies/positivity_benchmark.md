# Positivity benchmark: abs against softplus for g2

Configs in `configs/studies/positivity_*`. With the fix of `g2_bias.md` (input 10t),
softplus learns g2 correctly at ω = 5 but drives it towards 0 slowly when the data have
no counter-rotating term (g2 ≈ 0.09 after 150k epochs). The question: does abs,
g = |θ|, converge faster without losing accuracy?

Setting of `g2_bias.md`: `rabi2_ip` at ωc = ωa = 5, field dimension 6, input 10t, ODE
mean reduction, lr 1e-3; JC data (true g2 = 0) for 100k epochs, Rabi data (true
g2 = 1) for 50k; g2 started at exactly 1 (`coupling_init`) for both maps; seeds
100–104.

## Results

| | softplus | abs, abs start | abs, softplus start |
|---|---|---|---|
| JC data: g2 at 20k epochs | 0.16–0.25 | 0.003–0.06 | 0.009–0.066 |
| JC data: g2 at 50k epochs | 0.12–0.17 | 0.000–0.019 | 0.002–0.022 |
| JC data: g2 at 100k epochs | 0.103–0.127 | 0.000–0.008 | 0.000–0.010 |
| JC data: epochs until g2 < 0.05 | never (100k) | 5k–23k | 5k–25k |
| Rabi data: g2 at 50k epochs | 0.989–0.996 | 0.988–0.995 | 0.989–0.996 |
| Failed runs (g1 collapsed) | 0 / 10 | 2 / 10 (seed 104, both data) | **0 / 10** |
| fidelity, ⟨O⟩ MAE of the good runs | 1.0000, 0.7–2.0e-3 | 1.0000, 0.7–2.0e-3 | 1.0000, 0.6–2.0e-3 |

- **abs is 4 to 10 times faster** at removing an unsupported g2 and gives the same
  accuracy otherwise: with Rabi data g2 = 0.99 ± 0.01 either way.
- **The abs failures come from the start, not from abs.** Its default start is
  g = θ ~ U(0, 1); seed 104 drew g1 = 0.055 and stayed trapped near 0 (fidelity 0.55,
  total loss 0.10 against ≈ 0.005), the same near-zero trap as g2 in `g2_bias.md`.
  `physics.coupling_draw: softplus` starts abs couplings at softplus(θ) (0.69 to 1.31),
  the values a softplus run of the same seed starts from; with it all 10 runs succeed.
- Why softplus is slow: softplus(θ) → 0 only as θ → -∞, and its gradient factor
  sigmoid(θ) shrinks with g, so the already weak pull on an unresolved g2 fades further.
  abs reaches 0 at finite θ and then oscillates around it (the kink), which is what
  "switching a term off" needs.

This revises point 4 of `legacy_reproduction.md` ("softplus is the safer choice"):
that held with input t and abs's default start, where g1 collapsed often. With the
softplus start and input 10t, abs is as reliable and much faster.

## Recommendation for the sweep

`positivity: abs` with `coupling_draw: softplus`, g2 started near 1 (full model). An
unresolved g2 then reaches ≲ 0.02 within 50k epochs, so "g2 switched off" is a clear
outcome rather than a slow drift to be read against a resolution band (which is still
worth reporting).

## Reproducing

```bash
uv run python scripts/train.py configs/studies/positivity_<name>.yaml --workers 5
```

JC runs take about 12 min per seed (100k epochs), Rabi runs about 6 min.
