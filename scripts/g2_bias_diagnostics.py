"""Diagnostics B and C of the g2-bias study (docs/studies/g2_bias.md).

C, exact landscape: for each g2 on a grid, the exact trajectory of the model with
that g2 has zero ODE loss, so its total loss is its data loss against the training
data. This is the profile an ideal PINN would see. Also the ODE loss of the true
trajectory under H(g2), which measures how strongly the ODE term pins g2.

B, representation: the PINN's network (same architecture and seed) is fitted to the
exact trajectory directly, without physics. Its ODE loss as a function of g2 then
shows whether the network, at its best, can carry the counter-rotating dynamics: if
its ODE loss is lower at g2 = 0 than at the true g2, the network's time derivative
misses the fast components and the ODE term will push g2 to 0.

    uv run python scripts/g2_bias_diagnostics.py configs/reproduction/control_rabi_data_w5_recipe.yaml
"""

import argparse
import dataclasses
import json

import numpy as np
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.experiments import load_experiment
from pinn_jc.physics import Hamiltonian
from pinn_jc.reproducibility import set_seed
from pinn_jc.training.losses import loss_data, loss_ode, state_and_time_derivative
from pinn_jc.training.trainer import build_models

G2_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _simulate(physics, data_config, g2):
    return simulate(
        params={**physics.params, "g2": g2},
        tfinal=data_config.tfinal,
        n_time_steps=data_config.n_time_steps,
        init_state=physics.init_state,
        picture=physics.picture,
        dims=physics.dims,
        init_state_options=physics.init_state_options,
        truncation=physics.truncation,
    )


def _ode_profile(hamiltonian, params, psi, dpsi_dt, time, reduction):
    """ODE loss of a fixed trajectory under H(g1, g2) for every g2 on the grid."""
    return {
        g2: loss_ode(
            hamiltonian.to_torch(
                params, couplings=torch.tensor([params["g1"], g2]), time=time
            ),
            psi,
            dpsi_dt,
            reduction=reduction,
        ).item()
        for g2 in G2_GRID
    }


def exact_landscape(config):
    """Diagnostic C."""
    physics, data_config = config.physics, config.data
    reduction = config.training.ode_reduction
    data = _simulate(physics, data_config, physics.params["g2"])
    time = data.time.detach()
    hamiltonian = Hamiltonian(
        physics.picture, physics.dims, truncation=physics.truncation
    )
    true_h = hamiltonian.to_torch(physics.params, time=time)
    psi = data.states
    dpsi_dt = -1j * torch.einsum("nij,nj->ni", true_h, psi)

    data_loss = {}
    for g2 in G2_GRID:
        candidate = _simulate(physics, data_config, g2)
        data_loss[g2] = loss_data(
            candidate.states, data.observables, data.expectations
        ).item()
    return {
        "data_loss_of_exact_g2_trajectory": data_loss,
        "ode_loss_of_true_trajectory": _ode_profile(
            hamiltonian, physics.params, psi, dpsi_dt, time, reduction
        ),
    }


def supervised_fit(config, seed, epochs, learning_rate):
    """Diagnostic B."""
    physics, data_config = config.physics, config.data
    reduction = config.training.ode_reduction
    data = _simulate(physics, data_config, physics.params["g2"])
    time = data.time.detach()
    target = data.states

    set_seed(seed)
    state, _ = build_models(
        physics.hilbert_dim, physics.n_couplings, physics.positivity, config.model
    )
    optimizer = torch.optim.Adam(state.parameters(), lr=learning_rate, amsgrad=True)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (state(time) - target).abs().pow(2).mean()
        loss.backward()
        optimizer.step()

    hamiltonian = Hamiltonian(
        physics.picture, physics.dims, truncation=physics.truncation
    )
    psi, dpsi_dt = state_and_time_derivative(state, time)
    psi, dpsi_dt = psi.detach(), dpsi_dt.detach()
    true_dpsi_dt = -1j * torch.einsum(
        "nij,nj->ni", hamiltonian.to_torch(physics.params, time=time), target
    )
    fidelity = (psi.conj() * target).sum(1).abs() ** 2 / (psi.abs() ** 2).sum(1)
    return {
        "state_mae": (psi - target).abs().mean().item(),
        "derivative_mae": (dpsi_dt - true_dpsi_dt).abs().mean().item(),
        "fidelity_min": fidelity.min().item(),
        "data_loss": loss_data(psi, data.observables, data.expectations).item(),
        "ode_loss": _ode_profile(
            hamiltonian, physics.params, psi, dpsi_dt, time, reduction
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()

    results = {}
    for path in args.configs:
        config = load_experiment(path).config
        if args.hidden_units:
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(
                    config.model, hidden_units=tuple(args.hidden_units)
                ),
            )
        results[path] = {
            "exact": exact_landscape(config),
            "supervised": supervised_fit(
                config, args.seed, args.epochs, args.learning_rate
            ),
        }
        print(path)
        print(json.dumps(results[path], indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
