"""Training data: exact QuTiP solutions of the Schrödinger equation."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import qutip
import torch

from pinn_jc.physics.frames import Frame, change_frame
from pinn_jc.physics.hamiltonians import Hamiltonian, get_picture
from pinn_jc.physics.operators import observables
from pinn_jc.physics.states import initial_state
from pinn_jc.settings import DEVICE


@dataclass(frozen=True)
class SimulationData:
    """Simulated trajectory, as torch tensors on ``settings.DEVICE``."""

    states: torch.Tensor  # (T, D) complex64, in `frame`
    expectations: torch.Tensor  # (T, n_observables) float64, frame independent
    observables: torch.Tensor  # (n_observables, D, D) complex64: a†a, σ+σ-
    time: torch.Tensor  # (T, 1) float32, requires_grad
    picture: str  # requested picture
    simulated_picture: str  # picture actually integrated
    frame: Frame  # frame of `states`


def _is_resonant(params: Mapping[str, float]) -> bool:
    return abs(float(params["wc"]) - float(params["wa"])) < 1e-12


def simulation_picture(picture: str, params: Mapping[str, float]) -> str:
    """Picture to integrate for the requested one.

    - ``rabi2_ip`` is time dependent, so its lab-frame form ``rabi2`` is integrated.
    - On resonance, ``full`` and ``rabi2`` with g2 = 0 are exactly H0 + g1 (a†σ- + aσ+);
      they are integrated in the Jaynes-Cummings interaction picture instead, which
      avoids the stiffness of ωc, ωa >> g1.

    The states are converted to the requested frame afterwards, so this choice only
    affects numerical accuracy, never the returned data.
    """
    if picture == "rabi2_ip":
        picture = "rabi2"
    jaynes_cummings = picture == "full" or (
        picture == "rabi2" and float(params.get("g2", 0.0)) == 0.0
    )
    if jaynes_cummings and _is_resonant(params):
        return "interaction"
    return picture


def simulate(
    params: Mapping[str, float],
    tfinal: float,
    n_time_steps: int,
    init_state: str,
    picture: str = "interaction",
    dims: Mapping[str, int] | None = None,
    frame: Frame | str | None = None,
    init_state_options: Mapping[str, float] | None = None,
) -> SimulationData:
    """Simulate the atom-cavity system on ``n_time_steps`` points of [0, tfinal].

    ``frame`` selects the frame of the returned states; by default it is the frame of
    ``picture``, so the states satisfy dψ/dt = -i H ψ with that picture's Hamiltonian.
    ``init_state_options`` are passed to the initial state (e.g. ``alpha``, ``r``).
    """
    if dims is None:
        dims = {"atom": 2, "field": 2}
    if picture == "interaction" and not _is_resonant(params):
        raise ValueError(
            "the 'interaction' picture is the resonant Jaynes-Cummings interaction "
            "picture and needs wc == wa"
        )
    target = get_picture(picture).frame if frame is None else Frame(frame)
    simulated_picture = simulation_picture(picture, params)

    times = np.linspace(0, tfinal, n_time_steps)
    hamiltonian = Hamiltonian(simulated_picture, dims).to_qobj(params)
    measured = observables(dims)

    result = qutip.sesolve(
        hamiltonian,
        initial_state(init_state, dims, **(init_state_options or {})),
        times,
        e_ops=measured,
        options={"store_states": True},
    )

    states = np.array(
        [state.full().ravel() for state in result.states], dtype=np.complex64
    )
    source = get_picture(simulated_picture).frame
    states = change_frame(states, times, source, target, params, dims)

    return SimulationData(
        states=torch.tensor(states, device=DEVICE),
        expectations=torch.tensor(np.array(result.expect), device=DEVICE).transpose(
            0, 1
        ),
        observables=torch.tensor(
            np.array([op.full() for op in measured], dtype=np.complex64), device=DEVICE
        ),
        time=torch.tensor(
            times, dtype=torch.float32, requires_grad=True, device=DEVICE
        ).reshape(-1, 1),
        picture=picture,
        simulated_picture=simulated_picture,
        frame=target,
    )
