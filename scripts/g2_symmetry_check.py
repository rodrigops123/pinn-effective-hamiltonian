"""Check that the training data are even in g2 (docs/studies/g2_start.md).

The rotation a -> i a, σ+ -> -i σ+ leaves the JC term, |e,0> and the observables
⟨a†a⟩, ⟨σ+σ-⟩ unchanged and flips the sign of the counter-rotating term, so the data
of g2 and -g2 agree and the data loss is flat at g2 = 0. Prints:

- the largest difference between the observables of g2 and -g2;
- the data change against JC data, MAE(g2) / g2², near 0 (constant: quadratic);
- ⟨a²⟩, an observable odd in g2 that the rotation does not leave invariant.

    uv run python scripts/g2_symmetry_check.py --omega 5 --field 6
"""

import argparse

import numpy as np
import qutip

from pinn_jc.data.simulate import simulate


def _expectations(omega, g2, field, n_time_steps):
    data = simulate(
        params={"wc": omega, "wa": omega, "g1": 1.0, "g2": g2},
        tfinal=2 * np.pi,
        n_time_steps=n_time_steps,
        init_state="fock",
        picture="rabi2_ip",
        dims={"atom": 2, "field": field},
    )
    return data.expectations.numpy()


def _a_squared(omega, g2, field, n_time_steps):
    """⟨a²⟩(t) of the Rabi model from |e,0>, in the lab frame."""
    a = qutip.tensor(qutip.qeye(2), qutip.destroy(field))
    sm = qutip.tensor(qutip.destroy(2), qutip.qeye(field))
    hamiltonian = (
        omega * a.dag() * a
        + omega * sm.dag() * sm
        + (a.dag() * sm + a * sm.dag())
        + g2 * (a * sm + a.dag() * sm.dag())
    )
    psi0 = qutip.tensor(qutip.basis(2, 1), qutip.basis(field, 0))
    times = np.linspace(0, 2 * np.pi, n_time_steps)
    return qutip.sesolve(hamiltonian, psi0, times, e_ops=[a * a]).expect[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omega", type=float, default=5.0)
    parser.add_argument("--field", type=int, default=6)
    parser.add_argument("--n-time-steps", type=int, default=500)
    parser.add_argument("--g2", type=float, default=0.3)
    args = parser.parse_args()
    grid = (args.field, args.n_time_steps)

    plus = _expectations(args.omega, args.g2, *grid)
    minus = _expectations(args.omega, -args.g2, *grid)
    print(f"max |O(g2) - O(-g2)| at g2 = {args.g2}: {np.abs(plus - minus).max():.1e}")

    jc = _expectations(args.omega, 0.0, *grid)
    for g2 in (0.01, 0.02, 0.05, 0.1, 0.2):
        change = np.abs(_expectations(args.omega, g2, *grid) - jc).mean()
        print(
            f"g2 = {g2:<5} data change vs JC {change:.2e}, / g2² = {change / g2**2:.3f}"
        )

    a2_plus = _a_squared(args.omega, args.g2, *grid)
    a2_minus = _a_squared(args.omega, -args.g2, *grid)
    print(
        f"⟨a²⟩: max |⟨a²⟩(g2)| = {np.abs(a2_plus).max():.2e}, "
        f"max |⟨a²⟩(g2) + ⟨a²⟩(-g2)| = {np.abs(a2_plus + a2_minus).max():.1e}"
    )


if __name__ == "__main__":
    main()
