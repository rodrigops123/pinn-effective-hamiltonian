"""Preparation checks of the RWA sweep (docs/studies/rwa_sweep.md), without training.

For each ωc = ωa = ω, with g1 = g = 1 and the initial state |e,0>, on t in [0, 2π]:

- field dimension: the smallest one whose Rabi (g2 = g) observables agree with a much
  larger truncation to ``--dim-tolerance``;
- Δ: how far the Rabi data (g2 = g) are from the JC data (g2 = 0), as the data loss
  (MAE of ⟨a†a⟩, ⟨σ+σ-⟩, the measure the PINN's fit floor is quoted in) and as the
  largest pointwise difference;
- resolution bands: the data loss, against JC data and against Rabi data, of the exact
  trajectory of every g2 on a grid. The range of g2 whose data loss stays below the fit
  floor is what the data cannot tell from the truth;
- the time grid: points per period of the counter-rotating terms (frequency Σ = 2ω);
- with ``--noise-stds``: the same Rabi–JC gap against noisy Rabi data, MAE(data, JC)
  - MAE(data, Rabi), for each noise level (docs/studies/measurement_noise.md).

    uv run python scripts/rwa_sweep_preparation.py --omegas 100 20 10 5 2 1
"""

import argparse
import json

import numpy as np
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.training.trainer import noisy_expectations

G2_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)


def _expectations(omega, g2, field, n_time_steps, g=1.0):
    data = simulate(
        params={"wc": omega, "wa": omega, "g1": g, "g2": g2},
        tfinal=2 * np.pi,
        n_time_steps=n_time_steps,
        init_state="fock",
        picture="rabi2_ip",
        dims={"atom": 2, "field": field},
    )
    return data.expectations.numpy()


def _mae(a, b):
    return float(np.abs(a - b).mean())


def _noisy_gap(rabi, jc, std, seed):
    """MAE(noisy Rabi data, JC) - MAE(noisy Rabi data, Rabi)."""
    data = noisy_expectations(torch.from_numpy(rabi), std, seed).numpy()
    return _mae(data, jc) - _mae(data, rabi)


def field_dimension(omega, n_time_steps, tolerance, reference=40):
    """Smallest field dimension whose Rabi observables match ``reference`` levels."""
    exact = _expectations(omega, 1.0, reference, n_time_steps)
    for field in range(2, reference):
        error = float(
            np.abs(_expectations(omega, 1.0, field, n_time_steps) - exact).max()
        )
        if error < tolerance:
            return field, error
    return reference, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--omegas", type=float, nargs="+", default=[100, 20, 10, 5, 2, 1]
    )
    parser.add_argument("--points-per-period", type=int, default=10)
    parser.add_argument("--min-time-steps", type=int, default=500)
    parser.add_argument("--dim-tolerance", type=float, default=1e-4)
    parser.add_argument("--noise-stds", type=float, nargs="*", default=[])
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    results = {}
    for omega in args.omegas:
        sigma = 2 * omega
        n_time_steps = max(
            args.min_time_steps, args.points_per_period * round(sigma) + 1
        )
        field, dim_error = field_dimension(omega, n_time_steps, args.dim_tolerance)
        field = max(field, 3)  # the counter-rotating terms need |e,0> -> |g,1> -> |e,2>
        jc = _expectations(omega, 0.0, field, n_time_steps)
        rabi = _expectations(omega, 1.0, field, n_time_steps)
        profiles = {"jc": {}, "rabi": {}}
        for g2 in G2_GRID:
            trajectory = _expectations(omega, g2, field, n_time_steps)
            profiles["jc"][g2] = _mae(trajectory, jc)
            profiles["rabi"][g2] = _mae(trajectory, rabi)
        results[omega] = {
            "g_over_omega": 1 / omega,
            "n_time_steps": n_time_steps,
            "points_per_cr_period": (n_time_steps - 1) / sigma,
            "field": field,
            "field_error": dim_error,
            "delta_mae": _mae(rabi, jc),
            "delta_max": float(np.abs(rabi - jc).max()),
            "profile_vs_jc_data": profiles["jc"],
            "profile_vs_rabi_data": profiles["rabi"],
            "gap_vs_noisy_rabi_data": {
                std: _noisy_gap(rabi, jc, std, args.noise_seed)
                for std in args.noise_stds
            },
        }
        print(json.dumps({omega: results[omega]}, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
