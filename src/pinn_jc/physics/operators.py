"""Operators of the atom-cavity system in the product basis |atom> ⊗ |n>.

The basis ordering follows ``qutip.tensor(atom, field)``::

    [|g,0>, |g,1>, ..., |g,N-1>, |e,0>, ..., |e,N-1>]

with atom index 0 the ground state and 1 the excited state, so σ- = destroy(atom).
``dims`` is a mapping ``{"atom": n_atom_levels, "field": n_fock_states}``.
"""

from collections.abc import Callable, Mapping

import qutip


def annihilation(dims: Mapping[str, int]) -> qutip.Qobj:
    """Cavity annihilation operator a."""
    return qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))


def sigma_minus(dims: Mapping[str, int]) -> qutip.Qobj:
    """Atomic lowering operator σ-."""
    return qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))


def sigma_x(dims: Mapping[str, int]) -> qutip.Qobj:
    """Atomic σx (two-level atom only)."""
    if dims["atom"] != 2:
        raise ValueError(
            f"sigma_x needs a two-level atom, got atom dimension {dims['atom']}"
        )
    return qutip.tensor(qutip.sigmax(), qutip.qeye(dims["field"]))


def _build(product: Callable[[qutip.Qobj, qutip.Qobj], qutip.Qobj]):
    return lambda dims: product(annihilation(dims), sigma_minus(dims))


# Every operator a Hamiltonian term can multiply, by name.
OPERATORS: dict[str, Callable[[Mapping[str, int]], qutip.Qobj]] = {
    "photon_number": _build(lambda a, sm: a.dag() * a),  # a†a
    "atom_excitation": _build(lambda a, sm: sm.dag() * sm),  # σ+σ-
    "jc_emission": _build(lambda a, sm: a.dag() * sm),  # a†σ-
    "jc_absorption": _build(lambda a, sm: a * sm.dag()),  # aσ+
    "cr_lowering": _build(lambda a, sm: a * sm),  # aσ-
    "cr_raising": _build(lambda a, sm: a.dag() * sm.dag()),  # a†σ+
    "two_photon": lambda dims: (
        (annihilation(dims) + annihilation(dims).dag()) ** 2 * sigma_x(dims)
    ),  # (a + a†)² σx
    "atom_drive": sigma_x,  # σx
}


def operator(name: str, dims: Mapping[str, int]) -> qutip.Qobj:
    """Build a named operator of ``OPERATORS`` for the given dimensions."""
    if name not in OPERATORS:
        raise ValueError(f"unknown operator {name!r}; choose from {sorted(OPERATORS)}")
    return OPERATORS[name](dims)


def observables(dims: Mapping[str, int]) -> list[qutip.Qobj]:
    """Observables fitted by the data loss: photon number a†a and excitation σ+σ-."""
    return [operator("photon_number", dims), operator("atom_excitation", dims)]
