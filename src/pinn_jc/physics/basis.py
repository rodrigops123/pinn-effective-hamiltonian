"""Truncations of the atom-cavity basis.

``"field"`` keeps the whole product basis |s, n>, s < n_atom, n < n_field. In it the
top level |e, n_field - 1> is an artefact: Jaynes-Cummings couples it to
|g, n_field>, which the truncation removes, so it is a stationary state that a PINN
can use to fake photon number without any dynamics. ``"excitation"`` keeps only the
states with s + n < n_field (s counts atomic excitations), the usual truncation by
total excitation number, which has no such artefact.
"""

from collections.abc import Mapping

import numpy as np

TRUNCATIONS = ("field", "excitation")


def basis_indices(dims: Mapping[str, int], truncation: str = "field") -> np.ndarray:
    """Return the indices of the kept basis states in the product basis, in order.

    Parameters
    ----------
    dims : Mapping[str, int]
        ``{"atom": n_atom, "field": n_field}``.
    truncation : str
        ``"field"`` (all states) or ``"excitation"`` (s + n < n_field).

    Returns
    -------
    np.ndarray
        Sorted integer indices into the ``n_atom * n_field`` product basis.
    """
    if truncation not in TRUNCATIONS:
        raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {truncation!r}")
    n_atom, n_field = dims["atom"], dims["field"]
    levels = np.arange(n_atom * n_field)
    if truncation == "field":
        return levels
    atom, photons = np.divmod(levels, n_field)
    return levels[atom + photons < n_field]


def check_closed(operator: np.ndarray, indices: np.ndarray, name: str = "operator"):
    """Raise if ``operator`` couples the kept basis states to the dropped ones."""
    dropped = np.setdiff1d(np.arange(operator.shape[0]), indices)
    if dropped.size == 0:
        return
    leak = max(
        np.abs(operator[np.ix_(dropped, indices)]).max(),
        np.abs(operator[np.ix_(indices, dropped)]).max(),
    )
    if leak > 1e-12:
        raise ValueError(
            f"{name} couples the kept basis states to the dropped ones; the "
            "excitation truncation needs an excitation-conserving Hamiltonian"
        )
