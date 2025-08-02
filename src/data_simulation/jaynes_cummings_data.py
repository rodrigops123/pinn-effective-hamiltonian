import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import qutip
from typing import Union

from config import global_variables


def chooses_hamiltonian(picture: str, params: list[float]) -> qutip.Qobj:
    """
    Chooses the Hamiltonian based on the specified picture.

    Args:
        picture: The picture of the Hamiltonian to be used.
        params: in this order: wc (cavity frequency), wa (atom frequency), g (coupling strength)

    Returns:
        str: The chosen Hamiltonian.
    """
    if picture == "interaction":
        hamiltonian = params[-1] * (
            global_variables.a.dag() * global_variables.sm
            + global_variables.a * global_variables.sm.dag()
        )

    if picture == "atom":
        hamiltonian = abs(
            params[0] - params[1]
        ) / 2 * global_variables.a.dag() * global_variables.a + params[-1] * (
            global_variables.a.dag() * global_variables.sm
            + global_variables.a * global_variables.sm.dag()
        )

    return hamiltonian


def data_jc(
    params: list[float],
    tfinal: Union[float, int],
    n_time_steps: int,
    picture: str = "interaction",
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

    # Initial state (base state)
    psi0a = qutip.tensor(
        (qutip.basis(global_variables.FIELD_DIM, 0) + qutip.basis(global_variables.FIELD_DIM, 1)),
        qutip.basis(global_variables.FIELD_DIM, 0),
    )
    
    # psi0b = qutip.tensor(
    #     qutip.basis(global_variables.FIELD_DIM, 1),
    #     qutip.basis(global_variables.FIELD_DIM, 0),
    # )
    
    psi0 = psi0a.unit()
    
    # psi0 = (psi0a + psi0b).unit()

    # Defining the operators
    field_op = global_variables.a.dag() * global_variables.a
    atom_op = global_variables.sm.dag() * global_variables.sm

    operators_list = [field_op, atom_op]

    # Chosen Hamiltonian
    hamiltonian = chooses_hamiltonian(picture, params)

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
    hamiltonian = torch.tensor(hamiltonian.full(), dtype=torch.complex64)
    operators_list = torch.tensor(operators_list)
    y_train = torch.tensor(y_train)
    expect = torch.tensor(expect).transpose(0, 1)
    time = torch.tensor(tlist, dtype=torch.float32, requires_grad=True).reshape(-1, 1)

    return y_train, expect, hamiltonian, operators_list, time