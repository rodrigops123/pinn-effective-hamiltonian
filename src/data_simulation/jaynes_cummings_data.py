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


def choose_init_state(init_state: str, dims: dict) -> qutip.Qobj:
    # - estado de fock -> schrodinger
    # - superposição de fock -> schrodinger

    if init_state == "fock_superposition":
        # if dims["field"] == 2:
        #     psi0a = qutip.tensor(
        #         qutip.basis(dims["atom"], 0),
        #         (qutip.basis(dims["field"], 0) + qutip.basis(dims["field"], 1)),
        #     )
        #     psi0 = psi0a.unit()

        # elif dims["field"] == 3:
        psi0a = qutip.tensor(
            qutip.basis(dims["atom"], 1),
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


def chooses_hamiltonian(picture: str, params: dict, dims: dict) -> qutip.Qobj:
    """
    Chooses the Hamiltonian based on the specified picture.

    Args:
        picture: The picture of the Hamiltonian to be used.
        params: in this order: wc (cavity frequency), wa (atom frequency), g (coupling strength)

    Returns:
        str: The chosen Hamiltonian.
    """
    a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))
    if picture == "interaction":
        hamiltonian = params["g"] * (a.dag() * sm + a * sm.dag())

    elif picture == "atom":

        hamiltonian = 0.5 * abs(params["wc"] - params["wa"]) * a.dag() * a + params[
            "g"
        ] * (a.dag() * sm + a * sm.dag())

    elif picture == "full":
        sz = qutip.tensor(qutip.sigmaz(), qutip.qeye(dims["field"]))

        hamiltonian = (
            params["wc"] * a.dag() * a
            + 0.5 * params["wa"] * sz
            + params["g"] * (a.dag() * sm + a * sm.dag())
        )
    else:
        raise ValueError(
            "Picture not recognized. Choose 'interaction', 'atom' or 'full'."
        )

    return hamiltonian


def data_jc(
    params: dict,
    tfinal: Union[float, int],
    n_time_steps: int,
    init_state: str,
    picture: str = "interaction",
    dims: dict[str, int] = {"atom": 2, "field": 2},
):
    """
    Generates data for the Jaynes-Cummings model based on the provided parameters.

    Args:
        params: in this order: wc (cavity frequency), wa (atom frequency), g (coupling strength)
        tfinal: duration of the simulation
        N: number of time steps for the simulation
        device (optional): device to be used when running calculations. Defaults to "cpu".
        state (optional): whether or not states are saved. Defaults to True.
        picture (optional): picture of the Hamiltonian to be used. Defaults to "interaction".

    Returns:
        _type_: _description_
    """

    # time vector
    tlist = np.linspace(0, tfinal, n_time_steps)

    a = qutip.tensor(qutip.qeye(dims["atom"]), qutip.destroy(dims["field"]))
    sm = qutip.tensor(qutip.destroy(dims["atom"]), qutip.qeye(dims["field"]))

    # Defining the operators
    field_op = a.dag() * a
    atom_op = sm.dag() * sm

    operators_list = [field_op, atom_op]

    # Chosen Hamiltonian
    hamiltonian = chooses_hamiltonian(picture, params, dims)

    psi0 = choose_init_state(init_state, dims)

    options = qutip.Options(store_states=True)
    result = qutip.sesolve(
        hamiltonian, psi0, tlist, e_ops=operators_list, options=options
    )

    # Converting the results to numpy arrays
    operators_list = np.array([op.full() for op in operators_list], dtype=np.complex64)
    y_train = np.array(
        [result.states[i].full().flatten() for i in range(n_time_steps)],
        dtype=np.complex64,
    )
    expect = np.array(result.expect)

    # Converting to torch tensors
    hamiltonian = torch.tensor(hamiltonian.full(), dtype=torch.complex64, device=DEVICE)
    operators_list = torch.tensor(operators_list, device=DEVICE)
    y_train = torch.tensor(y_train, device=DEVICE)
    expect = torch.tensor(expect, device=DEVICE).transpose(0, 1)
    time = torch.tensor(
        tlist, dtype=torch.float32, requires_grad=True, device=DEVICE
    ).reshape(-1, 1)

    return y_train, expect, hamiltonian, operators_list, time
