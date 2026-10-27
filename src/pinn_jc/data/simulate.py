"""Training data: exact QuTiP solutions of the Schrödinger equation."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import qutip
import torch

from pinn_jc.physics.frames import Frame, rotate_to_interaction_frame
from pinn_jc.physics.hamiltonians import Hamiltonian, get_picture
from pinn_jc.physics.operators import observables
from pinn_jc.physics.states import initial_state
from pinn_jc.settings import DEVICE


@dataclass(frozen=True)
class SimulationData:
    """Simulated trajectory, as torch tensors on ``settings.DEVICE``.

    ``states`` are in ``simulation_frame``, additionally multiplied by exp(+i H0 t)
    when ``rotated_to_interaction_frame`` is set. That rotation only yields the
    interaction picture for lab-frame data (see bug B in tests/).
    """

    states: torch.Tensor  # (T, D) complex64
    expectations: torch.Tensor  # (T, n_observables) float64, frame independent
    observables: torch.Tensor  # (n_observables, D, D) complex64: a†a, σ+σ-
    hamiltonian: torch.Tensor  # (D, D) complex64, of `simulated_picture`
    time: torch.Tensor  # (T, 1) float32, requires_grad
    picture: str  # requested picture
    simulated_picture: str  # picture actually integrated
    simulation_frame: Frame
    rotated_to_interaction_frame: bool


def simulation_picture(picture: str, params: Mapping[str, float]) -> str:
    """Picture to integrate for the requested one.

    On resonance and without counter-rotating coupling, the lab-frame models are
    integrated in the Jaynes-Cummings interaction picture instead, which avoids the
    stiffness of ωc, ωa >> g.
    """
    resonant = abs(float(params["wc"]) - float(params["wa"])) < 1e-12
    no_counter_rotating = float(params.get("g2", 0.0)) == 0.0
    if picture in ("full", "rabi2", "rabi") and no_counter_rotating and resonant:
        return "interaction"
    return picture


def simulate(
    params: Mapping[str, float],
    tfinal: float,
    n_time_steps: int,
    init_state: str,
    picture: str = "interaction",
    dims: Mapping[str, int] | None = None,
    state_frame: str = "lab",
) -> SimulationData:
    """Simulate the atom-cavity system on ``n_time_steps`` points of [0, tfinal].

    state_frame:
      - "lab": the states as integrated, in the simulated picture's frame
      - "interaction": the states multiplied by exp(+i H0 t), H0 = ωc a†a + ωa σ+σ-
    """
    if state_frame not in ("lab", "interaction"):
        raise ValueError(
            f"state_frame must be 'lab' or 'interaction', got {state_frame!r}"
        )
    if dims is None:
        dims = {"atom": 2, "field": 2}

    times = np.linspace(0, tfinal, n_time_steps)
    simulated_picture = simulation_picture(picture, params)
    hamiltonian = Hamiltonian(simulated_picture, dims).to_qobj(params)
    measured = observables(dims)

    result = qutip.sesolve(
        hamiltonian,
        initial_state(init_state, dims),
        times,
        e_ops=measured,
        options={"store_states": True},
    )

    states = np.array(
        [state.full().ravel() for state in result.states], dtype=np.complex64
    )
    rotate = state_frame == "interaction"
    if rotate:
        states = rotate_to_interaction_frame(states, times, params, dims)

    return SimulationData(
        states=torch.tensor(states, device=DEVICE),
        expectations=torch.tensor(np.array(result.expect), device=DEVICE).transpose(
            0, 1
        ),
        observables=torch.tensor(
            np.array([op.full() for op in measured], dtype=np.complex64), device=DEVICE
        ),
        hamiltonian=torch.tensor(
            hamiltonian.full(), dtype=torch.complex64, device=DEVICE
        ),
        time=torch.tensor(
            times, dtype=torch.float32, requires_grad=True, device=DEVICE
        ).reshape(-1, 1),
        picture=picture,
        simulated_picture=simulated_picture,
        simulation_frame=get_picture(simulated_picture).frame,
        rotated_to_interaction_frame=rotate,
    )
