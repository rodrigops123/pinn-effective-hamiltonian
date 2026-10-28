"""Compare a trained PINN with the exact solution, in the frame it was trained in.

Everything the figures need is computed once here, so plotting never simulates and
the predicted and exact states are always in the same frame (that of
``physics.picture``, as in training).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.io.legacy import as_models
from pinn_jc.io.runs import LoadedRun
from pinn_jc.physics import Frame
from pinn_jc.training.config import PhysicsConfig

OBSERVABLE_LABELS = (
    r"\langle a^\dagger a \rangle",
    r"\langle \sigma_+ \sigma_- \rangle",
)


@dataclass(frozen=True)
class Evaluation:
    time: np.ndarray  # (T,) physical time
    predicted_state: np.ndarray  # (T, D) complex
    exact_state: np.ndarray  # (T, D) complex, same frame
    predicted_expectations: np.ndarray  # (T, n_observables)
    exact_expectations: np.ndarray  # (T, n_observables)
    fidelity: np.ndarray  # (T,) |<ψ|ψ_nn>|² / ||ψ_nn||²
    physics: PhysicsConfig
    frame: Frame
    train_end: float | None  # last training time when a validation window was held out

    @property
    def basis_labels(self) -> list[str]:
        atom_names = ("g", "e") if self.physics.dims["atom"] == 2 else None
        labels = []
        for level in range(self.physics.dims["atom"]):
            name = atom_names[level] if atom_names else str(level)
            labels += [
                rf"$|{name},{n}\rangle$" for n in range(self.physics.dims["field"])
            ]
        return labels

    @property
    def observable_labels(self) -> tuple[str, ...]:
        return OBSERVABLE_LABELS

    @property
    def time_unit(self) -> tuple[np.ndarray, str]:
        """Dimensionless time g1 t (as in the thesis) when g1 is defined, else t."""
        g1 = float(self.physics.params.get("g1", 0.0))
        if g1 > 0:
            return self.time * g1, r"$g_1 t$"
        return self.time, r"$t$"


def evaluate(
    models: Mapping,
    physics: PhysicsConfig,
    tfinal: float,
    n_time_steps: int,
    scale_time: bool = False,
    train_end: float | None = None,
) -> Evaluation:
    """Evaluate ``models`` (current ``{"state", ...}`` or legacy dict) on a fresh grid."""
    data = simulate(
        params=physics.params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=physics.init_state,
        picture=physics.picture,
        dims=physics.dims,
        init_state_options=physics.init_state_options,
    )
    time = data.time.detach()
    network_time = time / time.max() if scale_time else time
    state = as_models(models)["state"]
    was_training = state.training
    state.eval()
    with torch.no_grad():
        psi = state(network_time)
    state.train(was_training)

    predicted_expectations = torch.einsum(
        "ni,kij,nj->nk", psi.conj(), data.observables, psi
    ).real
    overlap = torch.sum(psi.conj() * data.states, dim=1)
    fidelity = overlap.abs() ** 2 / (psi.abs() ** 2).sum(dim=1)

    return Evaluation(
        time=time.numpy().ravel(),
        predicted_state=psi.numpy(),
        exact_state=data.states.numpy(),
        predicted_expectations=predicted_expectations.numpy(),
        exact_expectations=data.expectations.numpy(),
        fidelity=fidelity.numpy(),
        physics=physics,
        frame=data.frame,
        train_end=train_end,
    )


def evaluate_run(run: LoadedRun, n_time_steps: int | None = None) -> Evaluation:
    """Evaluate a saved run on its own time grid (or a finer one of the same span)."""
    data = run.config.data
    n_time_steps = n_time_steps or data.n_time_steps
    train_end = None
    if data.n_train < data.n_time_steps:
        train_end = data.tfinal * (data.n_train - 1) / (data.n_time_steps - 1)
    return evaluate(
        run.models,
        run.config.physics,
        data.tfinal,
        n_time_steps,
        scale_time=data.scale_time,
        train_end=train_end,
    )
