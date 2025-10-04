import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from tqdm import tqdm

from .loss_functions import (
    loss_ic,
    loss_norm,
    loss_data,
    loss_ode,
    hamiltonian_with_params,
)
from .neural_network import Neural_Net
from src.data_simulation.jaynes_cummings_data import data_jc
from config import global_variables
from utils import SIN


def train_test_split(data, test_size=0.2):
    """
    Splits the dataset into training and testing sets.
    Returns:
        train_data: Training data.
        test_data: Testing data.
    """
    len_data = len(data)
    indices = len_data - int(len_data * test_size)
    train_data = data[:indices]
    test_data = data[indices:]

    return train_data, test_data


def instantiate_model(
    output_dim,
    input_dim=1,
    neural_net_params=global_variables.model_train_params,
    create_parameter=False,
):

    model_real = Neural_Net(
        units=neural_net_params["units"],
        activation=neural_net_params["activation"],
        input=input_dim,
        output=output_dim,
        create_parameter=create_parameter,
    )

    model_imag = Neural_Net(
        units=neural_net_params["units"],
        activation=neural_net_params["activation"],
        input=input_dim,
        output=output_dim,
        create_parameter=False,
    )

    return model_real, model_imag


def train(
    epochs, params, tfinal, n_time_steps, init_state, picture, dims, n_points_loss=None
):

    output_dim = dims["atom"] * dims["field"]

    model_real, model_imag = instantiate_model(output_dim=output_dim)

    optimizer = torch.optim.Adam(
        list(model_real.parameters()) + list(model_imag.parameters()),
        lr=0.001,
        amsgrad=True,
    )

    sim_state, sim_expect, hamiltonian, operator, time = data_jc(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture,
        dims=dims,
    )

    sim_state_train, _ = train_test_split(sim_state, test_size=0.2)
    sim_expect_train, _ = train_test_split(sim_expect, test_size=0.2)
    time_train, _ = train_test_split(time, test_size=0.2)

    loss_dict = {
        "total_loss": [],
        "loss_ic": [],
        "loss_norm": [],
        "loss_data": [],
        "loss_ode": [],
    }

    for _ in tqdm(range(epochs)):

        # Forward pass
        nn_state_real = model_real(time_train)
        nn_state_imag = model_imag(time_train)
        nn_state = nn_state_real + 1j * nn_state_imag

        # Compute losses
        loss_ic_value = loss_ic(nn_state, sim_state_train)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(
            nn_state, operator, sim_expect_train, n_points=n_points_loss
        )
        loss_ode_value = loss_ode(hamiltonian, nn_state, time_train)

        # Total loss
        total_loss = loss_ic_value + loss_norm_value + loss_data_value + loss_ode_value

        # Store losses
        loss_dict["total_loss"].append(total_loss.item())
        loss_dict["loss_ic"].append(loss_ic_value.item())
        loss_dict["loss_norm"].append(loss_norm_value.item())
        loss_dict["loss_data"].append(loss_data_value.item())
        loss_dict["loss_ode"].append(loss_ode_value.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    models_dict = {
        "model_real": model_real,
        "model_imag": model_imag,
    }

    return models_dict, loss_dict


def train_with_parameter(
    epochs, params, tfinal, n_time_steps, init_state, picture, dims, n_points_loss
):

    output_dim = dims["atom"] * dims["field"]

    model_real, model_imag = instantiate_model(
        output_dim=output_dim, create_parameter=True
    )

    optimizer = torch.optim.Adam(
        list(model_real.parameters()) + list(model_imag.parameters()),
        lr=0.001,
        amsgrad=True,
    )

    sim_state, sim_expect, hamiltonian, operator, time = data_jc(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture,
        dims=dims,
    )

    sim_state_train, _ = train_test_split(sim_state, test_size=0.2)
    sim_expect_train, _ = train_test_split(sim_expect, test_size=0.2)
    time_train, _ = train_test_split(time, test_size=0.2)

    loss_dict = {
        "total_loss": [],
        "loss_ic": [],
        "loss_norm": [],
        "loss_data": [],
        "loss_ode": [],
        "learned_param": [],
    }

    for _ in tqdm(range(epochs)):

        # Forward pass
        nn_state_real = model_real(time_train)
        nn_state_imag = model_imag(time_train)
        nn_state = nn_state_real + 1j * nn_state_imag

        coupling_strength = model_real.param

        # Compute the Hamiltonian with the current parameters
        # and coupling strength
        hamiltonian = hamiltonian_with_params(
            picture=picture,
            params=params,
            coupling_strength=coupling_strength,
            dims=dims,
        )

        # Compute losses
        loss_ic_value = loss_ic(nn_state, sim_state_train)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(
            nn_state, operator, sim_expect_train, n_points=n_points_loss
        )
        loss_ode_value = loss_ode(hamiltonian, nn_state, time_train)

        # Total loss
        total_loss = loss_ic_value + loss_norm_value + loss_data_value + loss_ode_value

        # Store losses
        loss_dict["total_loss"].append(total_loss.item())
        loss_dict["loss_ic"].append(loss_ic_value.item())
        loss_dict["loss_norm"].append(loss_norm_value.item())
        loss_dict["loss_data"].append(loss_data_value.item())
        loss_dict["loss_ode"].append(loss_ode_value.item())
        loss_dict["learned_param"].append(coupling_strength.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    models_dict = {
        "model_real": model_real,
        "model_imag": model_imag,
    }

    return models_dict, loss_dict
