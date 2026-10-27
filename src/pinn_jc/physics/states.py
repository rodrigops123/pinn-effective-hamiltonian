"""Initial states of the atom-cavity system, as named in experiment configurations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import qutip


@dataclass(frozen=True)
class InitialState:
    description: str
    min_field_dim: int
    build: Callable[..., qutip.Qobj]


def _fock(dims):
    return qutip.tensor(qutip.basis(dims["atom"], 1), qutip.basis(dims["field"], 0))


def _fock_superposition(dims):
    field = (
        qutip.basis(dims["field"], 0)
        + qutip.basis(dims["field"], 1)
        + qutip.basis(dims["field"], 2)
    )
    return qutip.tensor(qutip.basis(dims["atom"], 0), field)


def _coherent(dims, alpha=1.0):
    return qutip.tensor(
        qutip.basis(dims["atom"], 0), qutip.coherent(dims["field"], alpha)
    )


def _squeeze(dims, r=1.0):
    field = qutip.squeeze(dims["field"], r) * qutip.basis(dims["field"], 0)
    return qutip.tensor(qutip.basis(dims["atom"], 0), field)


INITIAL_STATES = {
    "fock": InitialState("|e> ⊗ |0>", 1, _fock),
    "fock_superposition": InitialState(
        "|g> ⊗ (|0> + |1> + |2>)/√3", 3, _fock_superposition
    ),
    "coherent": InitialState(
        "|g> ⊗ |α>, truncated to the field dimension", 1, _coherent
    ),
    "squeeze": InitialState(
        "|g> ⊗ S(r)|0>, truncated to the field dimension", 1, _squeeze
    ),
}


def initial_state(name: str, dims: Mapping[str, int], **options) -> qutip.Qobj:
    """Normalized initial state. ``options`` are state parameters (``alpha``, ``r``)."""
    if name not in INITIAL_STATES:
        raise ValueError(
            f"unknown initial state {name!r}; choose from {sorted(INITIAL_STATES)}"
        )
    spec = INITIAL_STATES[name]
    if dims["atom"] < 2:
        raise ValueError(
            f"initial states need at least two atomic levels, got {dims['atom']}"
        )
    if dims["field"] < spec.min_field_dim:
        raise ValueError(
            f"initial state {name!r} needs a field dimension >= {spec.min_field_dim}, "
            f"got {dims['field']}"
        )
    state = spec.build(dims, **options).unit()
    if not np.isclose(state.norm(), 1.0):
        raise ValueError(f"initial state {name!r} could not be normalized")
    return state
