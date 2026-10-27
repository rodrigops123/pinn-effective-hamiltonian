"""Reference frames of the simulated states."""

from collections.abc import Mapping
from enum import StrEnum

import numpy as np


class Frame(StrEnum):
    LAB = "lab"  # Schrödinger picture of the full Hamiltonian
    ROTATING = "rotating"  # rotating at the atomic frequency ωa for both subsystems
    INTERACTION = "interaction"  # interaction picture w.r.t. H0 = ωc a†a + ωa σ+σ-


def free_energies(params: Mapping[str, float], dims: Mapping[str, int]) -> np.ndarray:
    """Diagonal of H0 = ωc a†a + ωa σ+σ- in the |atom> ⊗ |n> basis, shape (D,)."""
    photons = np.arange(dims["field"], dtype=np.float64)
    atom_levels = np.arange(dims["atom"], dtype=np.float64)
    wc, wa = float(params["wc"]), float(params["wa"])
    return (wc * photons[None, :] + wa * atom_levels[:, None]).ravel()


def rotate_to_interaction_frame(
    states: np.ndarray,
    times: np.ndarray,
    params: Mapping[str, float],
    dims: Mapping[str, int],
) -> np.ndarray:
    """Apply exp(+i H0 t) to states of shape (T, D), removing the fast carrier phases.

    Only meaningful for lab-frame states.
    """
    phases = np.exp(1j * np.outer(times, free_energies(params, dims))).astype(
        np.complex64
    )
    return (phases * states).astype(np.complex64)
