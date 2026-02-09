import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .loss_functions import (
    loss_ic,
    loss_norm,
    loss_data,
    loss_ode,
    hamiltonian_with_params,
)
from .neural_network import Neural_Net
from .models import MixFunn
from src.data_simulation.jaynes_cummings_data import data_jc
from config import global_variables


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
    n_paramater=1,
):

    model_real = Neural_Net(
        units=neural_net_params["units"],
        activation=neural_net_params["activation"],
        input=input_dim,
        output=output_dim,
        create_parameter=create_parameter,
        n_paramater=n_paramater,
    ).to(device=global_variables.DEVICE)

    model_imag = Neural_Net(
        units=neural_net_params["units"],
        activation=neural_net_params["activation"],
        input=input_dim,
        output=output_dim,
        create_parameter=False,
    ).to(device=global_variables.DEVICE)

    return model_real, model_imag


def instantiate_model_mixfunn(
    output_dim,
    input_dim=1,
    create_parameter=False,
    n_paramater=1,
):
    model_real = MixFunn(
        input_=input_dim,
        output_=output_dim,
        create_parameter=create_parameter,
        n_paramater=n_paramater,
    ).to(device=global_variables.DEVICE)

    model_imag = MixFunn(
        input_=input_dim,
        output_=output_dim,
        create_parameter=False,
    ).to(device=global_variables.DEVICE)

    return model_real, model_imag


def train(
    epochs, params, tfinal, n_time_steps, init_state, picture, dims, n_points_loss=None
):

    output_dim = dims["atom"] * dims["field"]

    model_real, model_imag = instantiate_model(output_dim=output_dim)

    optimizer = torch.optim.Adam(
        list(model_real.parameters()) + list(model_imag.parameters()),
        lr=global_variables.model_train_params["learning_rate"],
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
    epochs,
    params,
    tfinal,
    n_time_steps,
    init_state,
    picture,
    dims,
    n_points_loss,
    n_parameter,
    is_scaled,
):

    output_dim = dims["atom"] * dims["field"]

    model_real, model_imag = instantiate_model(
        output_dim=output_dim, create_parameter=True, n_paramater=n_parameter
    )

    optimizer = torch.optim.Adam(
        list(model_real.parameters()) + list(model_imag.parameters()),
        lr=global_variables.model_train_params["learning_rate"],
        amsgrad=True,
    )

    # --- KEY: generate training state in the interaction frame (slow envelope)
    sim_state, sim_expect, _, operator, time = data_jc(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture,
        dims=dims,
        state_frame="interaction",
    )

    # sim_state_train, _ = train_test_split(sim_state, test_size=0.2)
    # sim_expect_train, _ = train_test_split(sim_expect, test_size=0.2)
    # time_train, _ = train_test_split(time, test_size=0.2)

    time_scale = 1.0
    if is_scaled:
        tmax = time_train.max().detach()
        time_scale = float(tmax.item())  # d/dτ = tmax * d/dt
        time_train = (time_train.detach() / tmax).requires_grad_(True)

    # Train Hamiltonian picture:
    # If you asked for "rabi2" (lab), we train the interaction-picture ODE instead.
    picture_train = picture
    if picture == "rabi2":
        picture_train = "rabi2_ip"

    # Tiny regularization for g2 only when the DATA is JC-like (true g2 = 0).
    lambda_g2 = 1e-4 if float(params.get("g2", 0.0)) == 0.0 and n_parameter >= 2 else 0.0

    loss_dict = {
        "total_loss": [],
        "loss_ic": [],
        "loss_norm": [],
        "loss_data": [],
        "loss_ode": [],
        "learned_param": [],
    }

    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()

        nn_state_real = model_real(time)
        nn_state_imag = model_imag(time)
        nn_state = nn_state_real + 1j * nn_state_imag

        # Smooth positivity (better gradients than abs)
        coupling_strength = F.softplus(model_real.param)
        # coupling_strength = torch.abs(model_real.param)

        # Hamiltonian (possibly time-dependent)
        H_ = hamiltonian_with_params(
            picture=picture_train,
            params=params,
            coupling_strength=coupling_strength,
            dims=dims,
            tempo=time,
        )

        loss_ic_value = loss_ic(nn_state, sim_state)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(
            nn_state, operator, sim_expect, n_points=n_points_loss
        )
        loss_ode_value = loss_ode(H_, nn_state, time, time_scale=time_scale)

        reg = 0.0
        if lambda_g2 > 0.0:
            reg = lambda_g2 * coupling_strength[1].pow(2)

        total_loss = loss_ic_value + loss_norm_value + loss_data_value + loss_ode_value #+ reg

        total_loss.backward()
        optimizer.step()

        loss_dict["total_loss"].append(float(total_loss.item()))
        loss_dict["loss_ic"].append(float(loss_ic_value.item()))
        loss_dict["loss_norm"].append(float(loss_norm_value.item()))
        loss_dict["loss_data"].append(float(loss_data_value.item()))
        loss_dict["loss_ode"].append(float(loss_ode_value.item()))
        loss_dict["learned_param"].append(coupling_strength.detach().tolist())

    models_dict = {"model_real": model_real, "model_imag": model_imag}
    return models_dict, loss_dict





############################# IMPLEMENTACAO ANTIGA ##########################

# def train_with_parameter(
#     epochs,
#     params,
#     tfinal,
#     n_time_steps,
#     init_state,
#     picture,
#     dims,
#     n_points_loss,
#     n_parameter,
#     is_scaled,
# ):

#     output_dim = dims["atom"] * dims["field"]

#     model_real, model_imag = instantiate_model(
#         output_dim=output_dim, create_parameter=True, n_paramater=n_parameter
#     )

#     optimizer = torch.optim.Adam(
#         list(model_real.parameters()) + list(model_imag.parameters()),
#         lr=global_variables.model_train_params["learning_rate"],
#         amsgrad=True,
#     )

#     # Add learning rate scheduler to monitor loss_ode and adjust learning rate
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",  # Reduce LR when monitored value stops decreasing
#         factor=0.5,  # Multiply LR by this factor when reducing
#         patience=500,  # Number of epochs with no improvement after which LR will be reduced
#         min_lr=1e-6,  # Lower bound on the learning rate
#     )

#     sim_state, sim_expect, hamiltonian, operator, time = data_jc(
#         params=params,
#         tfinal=tfinal,
#         n_time_steps=n_time_steps,
#         init_state=init_state,
#         picture=picture,
#         dims=dims,
#     )

#     sim_state_train, _ = train_test_split(sim_state, test_size=0.2)
#     sim_expect_train, _ = train_test_split(sim_expect, test_size=0.2)
#     time_train, _ = train_test_split(time, test_size=0.2)

#     if is_scaled == True:
#         # scale time train:
#         time_train = (time_train.detach() / time_train.max().detach()).requires_grad_(
#             True
#         )

#     loss_dict = {
#         "total_loss": [],
#         "loss_ic": [],
#         "loss_norm": [],
#         "loss_data": [],
#         "loss_ode": [],
#         "learned_param": [],
#     }

#     for _ in tqdm(range(epochs)):
#         optimizer.zero_grad()

#         # Forward pass
#         nn_state_real = model_real(time_train)
#         nn_state_imag = model_imag(time_train)
#         nn_state = nn_state_real + 1j * nn_state_imag

#         coupling_strength = torch.abs(model_real.param)
#         # print(coupling_strength.shape)

#         # Compute the Hamiltonian with the current parameters
#         # and coupling strength
#         hamiltonian = hamiltonian_with_params(
#             picture=picture,
#             params=params,
#             coupling_strength=coupling_strength,
#             dims=dims,
#         )

#         # Compute losses
#         loss_ic_value = loss_ic(nn_state, sim_state_train)
#         loss_norm_value = loss_norm(nn_state)
#         loss_data_value = loss_data(
#             nn_state, operator, sim_expect_train, n_points=n_points_loss
#         )
#         loss_ode_value = loss_ode(hamiltonian, nn_state, time_train)

#         # Total loss
#         total_loss = loss_ic_value + loss_norm_value + 10*loss_data_value + 10*loss_ode_value
        
#         # Backward pass and optimization
#         total_loss.backward()
#         optimizer.step()

        
#         # print("g1:", coupling_strength[0])
#         # print("g2:", coupling_strength[1].grad())

#         scheduler.step(loss_data_value)  # Adjust learning rate based on data loss
#         scheduler.step(loss_ode_value)  # Adjust learning rate based on ODE loss

#         # Store losses
#         loss_dict["total_loss"].append(total_loss.item())
#         loss_dict["loss_ic"].append(loss_ic_value.item())
#         loss_dict["loss_norm"].append(loss_norm_value.item())
#         loss_dict["loss_data"].append(loss_data_value.item())
#         loss_dict["loss_ode"].append(loss_ode_value.item())
#         loss_dict["learned_param"].append(coupling_strength.tolist())

#     models_dict = {
#         "model_real": model_real,
#         "model_imag": model_imag,
#     }

#     return models_dict, loss_dict


# def train_with_parameter_mixfunn(
#     epochs,
#     params,
#     tfinal,
#     n_time_steps,
#     init_state,
#     picture,
#     dims,
#     n_points_loss,
#     n_parameter,
#     is_scaled,
# ):

#     output_dim = dims["atom"] * dims["field"]

#     model_real, model_imag = instantiate_model_mixfunn(
#         output_dim=output_dim, create_parameter=True, n_paramater=n_parameter
#     )

#     optimizer = torch.optim.Adam(
#         list(model_real.parameters()) + list(model_imag.parameters()),
#         lr=global_variables.model_train_params["learning_rate"],
#         amsgrad=True,
#     )

#     # Add learning rate scheduler to monitor loss_ode and adjust learning rate
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",  # Reduce LR when monitored value stops decreasing
#         factor=0.5,  # Multiply LR by this factor when reducing
#         patience=500,  # Number of epochs with no improvement after which LR will be reduced
#         min_lr=1e-6,  # Lower bound on the learning rate
#     )

#     sim_state, sim_expect, hamiltonian, operator, time = data_jc(
#         params=params,
#         tfinal=tfinal,
#         n_time_steps=n_time_steps,
#         init_state=init_state,
#         picture=picture,
#         dims=dims,
#     )

#     sim_state_train, _ = train_test_split(sim_state, test_size=0.)
#     sim_expect_train, _ = train_test_split(sim_expect, test_size=0.)
#     time_train, _ = train_test_split(time, test_size=0.)

#     if is_scaled == True:
#         # scale time train:
#         time_train = (time_train.detach() / time_train.max().detach()).requires_grad_(
#             True
#         )

#     loss_dict = {
#         "total_loss": [],
#         "loss_ic": [],
#         "loss_norm": [],
#         "loss_data": [],
#         "loss_ode": [],
#         "learned_param": [],
#     }

#     for _ in tqdm(range(epochs)):
#         optimizer.zero_grad()

#         # Forward pass
#         nn_state_real = model_real(time_train)
#         nn_state_imag = model_imag(time_train)
#         nn_state = nn_state_real + 1j * nn_state_imag

#         coupling_strength = torch.abs(model_real.param)

#         # Compute the Hamiltonian with the current parameters
#         # and coupling strength
#         hamiltonian = hamiltonian_with_params(
#             picture=picture,
#             params=params,
#             coupling_strength=coupling_strength,
#             dims=dims,
#         )

#         # Compute losses
#         loss_ic_value = loss_ic(nn_state, sim_state_train)
#         loss_norm_value = loss_norm(nn_state)
#         loss_data_value = loss_data(
#             nn_state, operator, sim_expect_train, n_points=n_points_loss
#         )
#         loss_ode_value = loss_ode(hamiltonian, nn_state, time_train)

#         # Total loss
#         total_loss = loss_ic_value + loss_norm_value + loss_data_value + loss_ode_value

#         # total_loss = loss_ic_value + loss_ode_value

#         # Backward pass and optimization
#         total_loss.backward()
#         optimizer.step()

#         # scheduler.step(loss_data_value)  # Adjust learning rate based on data loss
#         # scheduler.step(loss_ode_value)  # Adjust learning rate based on ODE loss

#         # Store losses
#         loss_dict["total_loss"].append(total_loss.item())
#         loss_dict["loss_ic"].append(loss_ic_value.item())
#         loss_dict["loss_norm"].append(loss_norm_value.item())
#         loss_dict["loss_data"].append(loss_data_value.item())
#         loss_dict["loss_ode"].append(loss_ode_value.item())
#         loss_dict["learned_param"].append(coupling_strength.tolist())

#     models_dict = {
#         "model_real": model_real,
#         "model_imag": model_imag,
#     }

#     return models_dict, loss_dict
