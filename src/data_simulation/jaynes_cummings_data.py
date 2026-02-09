import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import qutip
from typing import Union
from config.global_variables import DEVICE


def to_interaction_frame_numpy(
    y_lab: np.ndarray, tlist: np.ndarray, params: dict, dims: dict
) -> np.ndarray:
    """
    y_lab: (T, D) complex amplitudes in the product basis |atom>⊗|n>
    returns y_ip: (T, D) where y_ip(t) = exp(+i H0 t) y_lab(t)
    with H0 = wc a†a + wa σ+σ-.

    Basis ordering assumed from qutip.tensor(atom, field):
      [|0,0>, |0,1>, ..., |0,Nf-1>, |1,0>, ..., |1,Nf-1>]
    where atom index 0 is ground, 1 is excited.
    """
    wc = float(params["wc"])
    wa = float(params["wa"])
    Nf = int(dims["field"])

    n = np.arange(Nf, dtype=np.float64)  # 0..Nf-1
    E_g = wc * n  # ground manifold energies
    E_e = wc * n + wa  # excited manifold energies
    E = np.concatenate([E_g, E_e], axis=0)  # (D,)

    # phases(t,k) = exp(+i E_k t)
    phases = np.exp(1j * np.outer(tlist, E)).astype(np.complex64)  # (T, D)
    return (phases * y_lab).astype(np.complex64)


def choose_init_state(init_state: str, dims: dict) -> qutip.Qobj:

    if init_state == "fock_superposition":
        psi0a = qutip.tensor(
            qutip.basis(dims["atom"], 0),
            (
                qutip.basis(dims["field"], 0)
                + qutip.basis(dims["field"], 1)
                + qutip.basis(dims["field"], 2)
            ),
        )
        psi0 = psi0a.unit()

    if init_state == "fock":
        psi0a = qutip.tensor(
            qutip.basis(dims["atom"], 1), qutip.basis(dims["field"], 0)
        )
        psi0 = psi0a.unit()

    if init_state == "coherent":
        alpha = 1.0
        psi0a = qutip.tensor(
            qutip.basis(dims["atom"], 0), qutip.coherent(dims["field"], alpha)
        )
        psi0 = psi0a.unit()

    if init_state == "squeeze":
        r = 1.0
        psi0a = qutip.tensor(
            qutip.basis(dims["atom"], 0),
            qutip.squeeze(dims["field"], r) * qutip.basis(dims["field"], 0),
        )
        psi0 = psi0a.unit()

    return psi0


# ------------------------------------------------------------
# CHANGED: chooses_hamiltonian (fix "atom" convention and keep
# all models consistent with σ+σ- (sm†sm), avoiding spurious
# global phase mismatches between sim and ODE loss.
# ------------------------------------------------------------
def chooses_hamiltonian(picture: str, params: dict, dims: dict) -> qutip.Qobj:
    """
    picture options:
      - "interaction": JC interaction-picture at resonance (no fast free terms)
      - "atom": rotating at ωa for BOTH subsystems -> detuning term (wc-wa) a†a + interactions (time-independent JC)
      - "full": lab-frame JC: wc a†a + wa σ+σ- + g1(JC)
      - "rabi2": lab-frame split: g1(JC) + g2(counter-rotating)
      - others unchanged
    """
    a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))
    sx = qutip.tensor(qutip.sigmax(), qutip.qeye(dims["field"]))

    if picture == "interaction":
        # JC interaction picture at resonance: H_I = g1 (a†σ- + aσ+)
        hamiltonian = params["g1"] * (a.dag() * sm + a * sm.dag())

    elif picture == "atom":
        # Rotating frame at ωa applied to BOTH (a†a + σ+σ-):
        # H = (wc - wa) a†a + g1 (a†σ- + aσ+)
        delta = params["wc"] - params["wa"]
        hamiltonian = delta * (a.dag() * a) + params["g1"] * (
            a.dag() * sm + a * sm.dag()
        )

    elif picture == "full":
        # Lab JC with σ+σ- (projector). Keep this consistent everywhere.
        hamiltonian = (
            params["wc"] * (a.dag() * a)
            + params["wa"] * (sm.dag() * sm)
            + params["g1"] * (a.dag() * sm + a * sm.dag())
        )

    elif picture == "rabi":
        hamiltonian = (
            params["wc"] * a.dag() * a
            + params["wa"] * sm.dag() * sm
            + params["g1"] * a.dag() * sm
            + params["g2"] * a * sm.dag()
            + params["g3"] * a * sm
            + params["g4"] * a.dag() * sm.dag()
        )

    elif picture == "rabi2":
        hamiltonian = (
            params["wc"] * (a.dag() * a)
            + params["wa"] * (sm.dag() * sm)
            + params["g1"] * (a.dag() * sm + a * sm.dag())
            + params["g2"] * (a * sm + a.dag() * sm.dag())
        )

    elif picture == "geral":
        hamiltonian = (
            params["wc"] * a.dag() * a
            + params["wa"] * sm.dag() * sm
            + params["g1"] * (a.dag() * sm + a * sm.dag())  # jaynes-cummings
            + params["g2"] * (a * sm + a.dag() * sm.dag())  # rabi
            + params["g3"] * (a + a.dag()) ** 2 * sx  # two-photon
            + params["g0"] * sx  # classic field
        )

    else:
        raise ValueError(
            "Invalid picture. Choose from 'interaction', 'atom', 'full', 'rabi', 'rabi2', or 'geral'."
        )

    return hamiltonian


# ------------------------------------------------------------
# CHANGED: data_jc
# - optionally simulate with a "slow" Hamiltonian when possible
# - return the state in the interaction frame to eliminate fast
#   ω oscillations that your NN doesn't need to learn.
# ------------------------------------------------------------
def data_jc(
    params: dict,
    tfinal: float,
    n_time_steps: int,
    init_state: str,
    picture: str = "interaction",
    dims: dict[str, int] = {"atom": 2, "field": 2},
    state_frame: str = "lab",  # "lab" or "interaction"
):
    """
    Generates data for the Jaynes-Cummings model.

    state_frame:
      - "interaction": returns y_train(t) = exp(+iH0 t) |ψ_lab(t)>
        with H0 = wc a†a + wa σ+σ-
        (removes fast carrier phases)
      - "lab": returns the raw Schrödinger-picture state
    """

    tlist = np.linspace(0, tfinal, n_time_steps)

    a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))

    field_op = a.dag() * a
    atom_op = sm.dag() * sm
    operators_list = [field_op, atom_op]

    # --- High-impact: if you're in the JC regime and on-resonance, simulate directly
    # in the interaction picture to avoid stiffness from wc,wa >> g.
    picture_sim = picture
    if picture in ("full", "rabi2", "rabi") and float(params.get("g2", 0.0)) == 0.0:
        if abs(float(params["wc"]) - float(params["wa"])) < 1e-12:
            picture_sim = "interaction"

    hamiltonian = chooses_hamiltonian(picture_sim, params, dims)
    psi0 = choose_init_state(init_state, dims)

    options = qutip.Options(store_states=True)
    result = qutip.sesolve(
        hamiltonian, psi0, tlist, e_ops=operators_list, options=options
    )

    operators_list_np = np.array(
        [op.full() for op in operators_list], dtype=np.complex64
    )
    y_train = np.array(
        [result.states[i].full().flatten() for i in range(n_time_steps)],
        dtype=np.complex64,
    )

    # --- Key change: strip the fast free-evolution phases (carrier).
    if state_frame == "interaction":
        y_train = to_interaction_frame_numpy(y_train, tlist, params, dims)

    expect = np.array(result.expect)

    # torch conversions
    hamiltonian_t = torch.tensor(
        hamiltonian.full(), dtype=torch.complex64, device=DEVICE
    )
    operators_list_t = torch.tensor(operators_list_np, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    expect_t = torch.tensor(expect, device=DEVICE).transpose(0, 1)
    time_t = torch.tensor(
        tlist, dtype=torch.float32, requires_grad=True, device=DEVICE
    ).reshape(-1, 1)

    return y_train_t, expect_t, hamiltonian_t, operators_list_t, time_t








################################ IMPLEMENTACAO ANTIGA #################################

# def choose_init_state(init_state: str, dims: dict) -> qutip.Qobj:

#     if init_state == "fock_superposition":
#         psi0a = qutip.tensor(
#             qutip.basis(dims["atom"], 0),
#             (
#                 qutip.basis(dims["field"], 0)
#                 + qutip.basis(dims["field"], 1)
#                 + qutip.basis(dims["field"], 2)
#             ),
#         )
#         psi0 = psi0a.unit()

#     if init_state == "fock":
#         psi0a = qutip.tensor(
#             qutip.basis(dims["atom"], 1), qutip.basis(dims["field"], 0)
#         )
#         psi0 = psi0a.unit()

#     if init_state == "coherent":
#         alpha = 1.0
#         psi0a = qutip.tensor(
#             qutip.basis(dims["atom"], 0), qutip.coherent(dims["field"], alpha)
#         )
#         psi0 = psi0a.unit()

#     if init_state == "squeeze":
#         r = 1.0
#         psi0a = qutip.tensor(
#             qutip.basis(dims["atom"], 0),
#             qutip.squeeze(dims["field"], r) * qutip.basis(dims["field"], 0),
#         )
#         psi0 = psi0a.unit()

#     return psi0


# def chooses_hamiltonian(picture: str, params: dict, dims: dict) -> qutip.Qobj:
#     """
#     Chooses the Hamiltonian based on the specified picture.

#     Args:
#         picture: The picture of the Hamiltonian to be used.
#         params: in this order: wc (cavity frequency), wa (atom frequency), g (coupling strength)

#     Returns:
#         str: The chosen Hamiltonian.
#     """
#     a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
#     sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))
#     sx = qutip.tensor(qutip.sigmax(), qutip.qeye(dims["field"]))

#     if picture == "interaction":
#         hamiltonian = params["g1"] * (a.dag() * sm + a * sm.dag())

#     elif picture == "atom":
#         hamiltonian = 0.5 * abs(params["wc"] - params["wa"]) * a.dag() * a + params[
#             "g1"
#         ] * (a.dag() * sm + a * sm.dag())

#     elif picture == "full":
#         hamiltonian = (
#             params["wc"] * a.dag() * a
#             + params["wa"] * sm.dag() * sm
#             + params["g1"] * (a.dag() * sm + a * sm.dag())
#         )

#     elif picture == "rabi":
#         hamiltonian = (
#             params["wc"] * a.dag() * a
#             + params["wa"] * sm.dag() * sm
#             + params["g1"] * a.dag() * sm
#             + params["g2"] * a * sm.dag()
#             + params["g3"] * a * sm
#             + params["g4"] * a.dag() * sm.dag()
#         )

#     elif picture == "rabi2":
#         hamiltonian = (
#             params["wc"] * a.dag() * a
#             + params["wa"] * sm.dag() * sm
#             + params["g1"] * (a.dag() * sm + a * sm.dag())
#             + params["g2"] * (a * sm + a.dag() * sm.dag())
#         )

#     elif picture == "geral":
#         hamiltonian = (
#             params["wc"] * a.dag() * a
#             + params["wa"] * sm.dag() * sm
#             + params["g1"] * (a.dag() * sm + a * sm.dag())      # jaynes-cummings
#             + params["g2"] * (a * sm + a.dag() * sm.dag())      # rabi
#             + params["g3"] * (a + a.dag())**2 * sx              # two-photon
#             + params["g0"] * sx                                # classic field
#         )

#     else:
#         raise ValueError(
#             "Invalid picture. Choose from 'interaction', 'atom', 'full', 'rabi', 'rabi2', or 'geral'."
#         )

#     return hamiltonian


# def data_jc(
#     params: dict,
#     tfinal: Union[float, int],
#     n_time_steps: int,
#     init_state: str,
#     picture: str = "interaction",
#     dims: dict[str, int] = {"atom": 2, "field": 2},
# ):
#     """
#     Generates data for the Jaynes-Cummings model based on the provided parameters.

#     Args:
#         params: in this order: wc (cavity frequency), wa (atom frequency), g (coupling strength)
#         tfinal: duration of the simulation
#         N: number of time steps for the simulation
#         device (optional): device to be used when running calculations. Defaults to "cpu".
#         state (optional): whether or not states are saved. Defaults to True.
#         picture (optional): picture of the Hamiltonian to be used. Defaults to "interaction".

#     Returns:
#         _type_: _description_
#     """

#     # time vector
#     tlist = np.linspace(0, tfinal, n_time_steps)

#     a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
#     sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))

#     # print(a.shape, sm.shape, sx.shape)

#     # Defining the operators
#     field_op = a.dag() * a
#     atom_op = sm.dag() * sm

#     operators_list = [field_op, atom_op, ]#sx, field_quadrature_op]

#     # Chosen Hamiltonian
#     hamiltonian = chooses_hamiltonian(picture, params, dims)

#     psi0 = choose_init_state(init_state, dims)

#     options = qutip.Options(store_states=True)
#     result = qutip.sesolve(
#         hamiltonian, psi0, tlist, e_ops=operators_list, options=options
#     )

#     # Converting the results to numpy arrays
#     operators_list = np.array([op.full() for op in operators_list], dtype=np.complex64)
#     y_train = np.array(
#         [result.states[i].full().flatten() for i in range(n_time_steps)],
#         dtype=np.complex64,
#     )

#     # correction of the global phase
#     # y_train = y_train * np.exp(1j * params["wc"] * tlist[:, None])

#     expect = np.array(result.expect)

#     # Converting to torch tensors
#     hamiltonian = torch.tensor(hamiltonian.full(), dtype=torch.complex64, device=DEVICE)
#     operators_list = torch.tensor(operators_list, device=DEVICE)
#     y_train = torch.tensor(y_train, device=DEVICE)
#     expect = torch.tensor(expect, device=DEVICE).transpose(0, 1)
#     time = torch.tensor(
#         tlist, dtype=torch.float32, requires_grad=True, device=DEVICE
#     ).reshape(-1, 1)

#     return y_train, expect, hamiltonian, operators_list, time
