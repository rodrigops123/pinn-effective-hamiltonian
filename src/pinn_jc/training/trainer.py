import torch
from tqdm import tqdm

from pinn_jc import settings
from pinn_jc.data.simulate import simulate
from pinn_jc.models import MLP, ComplexStateNet, Couplings
from pinn_jc.physics import Hamiltonian
from pinn_jc.training.losses import (
    loss_data,
    loss_ic,
    loss_norm,
    loss_ode,
)


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


def data_indices(n_samples, n_points, seed=settings.SEED):
    """Time indices at which expectation values are fitted.

    None or 1 selects the first instant only; otherwise ``n_points`` instants are
    drawn once, with a private generator, and kept fixed for the whole training
    (fixed measurement times). All instants are used when n_points >= n_samples.
    """
    if n_points is None or n_points == 1:
        return torch.tensor([0])
    if n_points >= n_samples:
        return None
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_samples, generator=generator)[:n_points]


def build_models(
    output_dim,
    n_couplings=0,
    positivity="softplus",
    neural_net_params=settings.model_train_params,
):
    """Networks for ψ(t) and, if ``n_couplings`` > 0, the learnable couplings.

    The construction order (real network, couplings, imaginary network) fixes how the
    random initialisation is drawn; keep it so that a seed reproduces earlier runs.
    """

    def network():
        return MLP(
            in_features=1,
            out_features=output_dim,
            hidden_units=neural_net_params["units"],
            activation=neural_net_params["activation"],
        )

    real = network()
    couplings = Couplings(n_couplings, positivity) if n_couplings else None
    imag = network()

    state = ComplexStateNet(real, imag).to(device=settings.DEVICE)
    if couplings is not None:
        couplings = couplings.to(device=settings.DEVICE)
    return state, couplings


def train(
    epochs, params, tfinal, n_time_steps, init_state, picture, dims, n_points_loss=None
):

    output_dim = dims["atom"] * dims["field"]

    state, _ = build_models(output_dim)

    optimizer = torch.optim.Adam(
        state.parameters(),
        lr=settings.model_train_params["learning_rate"],
        amsgrad=True,
    )

    data = simulate(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture,
        dims=dims,
    )
    sim_state, sim_expect = data.states, data.expectations
    operator, time = data.observables, data.time

    sim_state_train, _ = train_test_split(sim_state, test_size=0.2)
    sim_expect_train, _ = train_test_split(sim_expect, test_size=0.2)
    time_train, _ = train_test_split(time, test_size=0.2)
    hamiltonian = Hamiltonian(picture, dims).to_torch(params, time=time_train)
    indices = data_indices(len(time_train), n_points_loss)

    loss_dict = {
        "total_loss": [],
        "loss_ic": [],
        "loss_norm": [],
        "loss_data": [],
        "loss_ode": [],
    }

    for _ in tqdm(range(epochs)):
        nn_state = state(time_train)

        # Compute losses
        loss_ic_value = loss_ic(nn_state, sim_state_train)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(nn_state, operator, sim_expect_train, indices)
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

    return {"state": state, "couplings": None}, loss_dict


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
    positivity="softplus",
):

    output_dim = dims["atom"] * dims["field"]

    state, couplings = build_models(output_dim, n_parameter, positivity)

    optimizer = torch.optim.Adam(
        [*state.parameters(), *couplings.parameters()],
        lr=settings.model_train_params["learning_rate"],
        amsgrad=True,
    )

    # "rabi2" is learned in its interaction picture, which removes the fast carrier
    # phases; the data is simulated in that picture's frame so that it satisfies the
    # ODE being enforced.
    picture_train = "rabi2_ip" if picture == "rabi2" else picture
    hamiltonian = Hamiltonian(picture_train, dims)

    data = simulate(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture_train,
        dims=dims,
    )
    sim_state, sim_expect = data.states, data.expectations
    operator, time = data.observables, data.time
    indices = data_indices(len(time), n_points_loss)

    # The network sees τ = t / T; the ODE becomes dψ/dτ = -i T H(t) ψ, where the
    # Hamiltonian keeps the physical time.
    time_scale = 1.0
    network_time = time
    if is_scaled:
        tmax = time.max().detach()
        time_scale = float(tmax.item())
        network_time = (time.detach() / tmax).requires_grad_(True)

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

        nn_state = state(network_time)
        coupling_strength = couplings()

        # Hamiltonian (possibly time-dependent)
        H_ = hamiltonian.to_torch(
            params, couplings=coupling_strength, time=time.detach()
        )

        loss_ic_value = loss_ic(nn_state, sim_state)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(nn_state, operator, sim_expect, indices)
        loss_ode_value = loss_ode(H_, nn_state, network_time, time_scale=time_scale)

        total_loss = loss_ic_value + loss_norm_value + loss_data_value + loss_ode_value

        total_loss.backward()
        optimizer.step()

        loss_dict["total_loss"].append(float(total_loss.item()))
        loss_dict["loss_ic"].append(float(loss_ic_value.item()))
        loss_dict["loss_norm"].append(float(loss_norm_value.item()))
        loss_dict["loss_data"].append(float(loss_data_value.item()))
        loss_dict["loss_ode"].append(float(loss_ode_value.item()))
        loss_dict["learned_param"].append(coupling_strength.detach().tolist())

    return {"state": state, "couplings": couplings}, loss_dict
