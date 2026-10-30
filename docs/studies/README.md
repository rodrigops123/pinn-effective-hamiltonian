# Study reports

Each report states its question, configs, results, conclusions and how to reproduce
them. In the order they build on each other:

| Report | Question | Main result |
|---|---|---|
| [coherent_squeeze.md](coherent_squeeze.md) | Why do `atom_coherent` and `atom_squeeze` fail to recover g1? | Excitation truncation, ODE mean reduction and lr 1e-3 fix the coherent state; the squeezed state stays at g1 ≈ 0.65 |
| [legacy_reproduction.md](legacy_reproduction.md) | Do the legacy results reproduce, and does the Rabi-loss claim hold where the counter-rotating terms are visible? | g1 and the state are recovered when training converges (seed-dependent with the legacy settings); at ω = 5 the PINN sets g2 ≈ 0 whatever its true value |
| [g2_bias.md](g2_bias.md) | Why does the PINN switch off a visible g2? | Spectral bias: the network cannot represent the counter-rotating dynamics; input 10t (`model.input_scale`) fixes it |
| [positivity_benchmark.md](positivity_benchmark.md) | abs or softplus for the couplings? | abs with the softplus start (`physics.coupling_draw`) is as accurate and 4–10× faster at removing an unsupported g2 |
| [rwa_sweep.md](rwa_sweep.md) | Does the PINN learn the effective JC model where the RWA holds and the full model where it does not? | Yes at both ends; the switch lies between g/ω = 0.067 and 0.1 and reflects a parsimony bias of the method |
| [g2_start.md](g2_start.md) | What if g2 starts near 0? | A trap at every ω, from a symmetry (the data are even in g2); protocol: two starts, lowest total loss |

## Conventions shared by the reports

- Per-seed numbers (final couplings; losses as the mean of the last 1k epochs;
  fidelity and ⟨O⟩ MAE against the exact solution on the training grid) come from
  `scripts/summarize_runs.py saved_artifacts/runs/<experiment>`.
- Every quoted run comes from a clean commit, recorded in its `meta.json` and in the
  output directory's name.
- "Fit floor": the data loss (MAE of ⟨a†a⟩, ⟨σ+σ-⟩) a successful run reaches, about
  1–2e-3 after 50k epochs. A difference in the data below it is not resolved.
- Total losses are compared only between runs with the same training settings.
