"""Reference frames of the simulated states.

Every frame used here is a diagonal rotation of the lab frame,
ψ_frame(t) = exp(+i E t) ψ_lab(t), with E the diagonal of the frame's generator in the
|atom> ⊗ |n> basis. Changing frames is therefore a phase per basis state.
"""

from collections.abc import Mapping
from enum import StrEnum

import numpy as np


class Frame(StrEnum):
    # Generators: LAB none; ROTATING ωa (a†a + σ+σ-), i.e. rotating at ωa for both
    # subsystems; INTERACTION H0 = ωc a†a + ωa σ+σ-.
    LAB = "lab"
    ROTATING = "rotating"
    INTERACTION = "interaction"


def frame_energies(
    frame: Frame, params: Mapping[str, float], dims: Mapping[str, int]
) -> np.ndarray:
    """Diagonal of the frame's generator in the |atom> ⊗ |n> basis, shape (D,)."""
    photons = np.arange(dims["field"], dtype=np.float64)[None, :]
    atom_levels = np.arange(dims["atom"], dtype=np.float64)[:, None]
    if frame is Frame.LAB:
        energies = 0.0 * photons + 0.0 * atom_levels
    elif frame is Frame.ROTATING:
        energies = float(params["wa"]) * photons + float(params["wa"]) * atom_levels
    elif frame is Frame.INTERACTION:
        energies = float(params["wc"]) * photons + float(params["wa"]) * atom_levels
    else:
        raise ValueError(f"unknown frame {frame!r}")
    return energies.ravel()


def change_frame(
    states: np.ndarray,
    times: np.ndarray,
    source: Frame,
    target: Frame,
    params: Mapping[str, float],
    dims: Mapping[str, int],
) -> np.ndarray:
    """Re-express states of shape (T, D) given in ``source`` in ``target``."""
    if source is target:
        return states
    energies = frame_energies(target, params, dims) - frame_energies(
        source, params, dims
    )
    phases = np.exp(1j * np.outer(times, energies)).astype(np.complex64)
    return (phases * states).astype(np.complex64)
