"""PINN training: learn ψ(t) and, optionally, the Hamiltonian couplings."""

import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from pinn_jc import settings
from pinn_jc.data.simulate import simulate
from pinn_jc.models import MLP, ComplexStateNet, Couplings, Sine
from pinn_jc.physics import Hamiltonian
from pinn_jc.reproducibility import set_seed
from pinn_jc.training.config import ExperimentConfig, ModelConfig
from pinn_jc.training.history import TrainingHistory, TrainingResult
from pinn_jc.training.losses import loss_data, loss_ic, loss_norm, loss_ode

logger = logging.getLogger(__name__)

ACTIVATIONS = {"sine": Sine, "tanh": nn.Tanh}


def data_indices(n_samples, n_points, seed=42):
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


def build_models(output_dim, n_couplings=0, positivity="softplus", model=None):
    """Networks for ψ(t) and, if ``n_couplings`` > 0, the learnable couplings.

    The construction order (real network, couplings, imaginary network) fixes how the
    random initialisation is drawn; keep it so that a seed reproduces earlier runs.
    """
    model = model or ModelConfig()
    activation = ACTIVATIONS[model.activation]()

    def network():
        return MLP(
            in_features=1,
            out_features=output_dim,
            hidden_units=model.hidden_units,
            activation=activation,
            dropout=model.dropout,
        )

    real = network()
    couplings = Couplings(n_couplings, positivity) if n_couplings else None
    imag = network()

    state = ComplexStateNet(real, imag).to(device=settings.DEVICE)
    if couplings is not None:
        couplings = couplings.to(device=settings.DEVICE)
    return state, couplings


def _validation_metrics(state, network_time, states, expectations, observables):
    """Mean fidelity |<ψ|ψ_nn>|² / ||ψ_nn||² and expectation-value MAE."""
    was_training = state.training
    state.eval()
    with torch.no_grad():
        psi = state(network_time)
    state.train(was_training)
    overlap = torch.sum(psi.conj() * states, dim=1)
    fidelity = overlap.abs() ** 2 / (psi.abs() ** 2).sum(dim=1)
    predicted = torch.einsum("ni,kij,nj->nk", psi.conj(), observables, psi).real
    return fidelity.mean().item(), (predicted - expectations).abs().mean().item()


def train(config: ExperimentConfig) -> TrainingResult:
    physics, data_config, training = config.physics, config.data, config.training
    if training.seed is not None:
        set_seed(training.seed)

    state, couplings = build_models(
        physics.hilbert_dim, physics.n_couplings, physics.positivity, config.model
    )
    parameters = list(state.parameters())
    if couplings is not None:
        parameters += list(couplings.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=training.learning_rate, amsgrad=training.amsgrad
    )

    # States come in the frame of `physics.picture`, so they satisfy its ODE.
    data = simulate(
        params=physics.params,
        tfinal=data_config.tfinal,
        n_time_steps=data_config.n_time_steps,
        init_state=physics.init_state,
        picture=physics.picture,
        dims=physics.dims,
        init_state_options=physics.init_state_options,
    )
    n_train = data_config.n_train
    time = data.time
    time_train = time if n_train == len(time) else time[:n_train]
    sim_state, sim_expect = data.states[:n_train], data.expectations[:n_train]
    indices = data_indices(n_train, data_config.n_data_points, data_config.data_seed)

    # The network sees τ = t / T; the ODE becomes dψ/dτ = -i T H(t) ψ, where the
    # Hamiltonian keeps the physical time.
    time_scale, network_time = 1.0, time_train
    validation_time = time[n_train:].detach()
    if data_config.scale_time:
        tmax = time.max().detach()
        time_scale = float(tmax.item())
        network_time = (time_train.detach() / tmax).requires_grad_(True)
        validation_time = validation_time / tmax

    hamiltonian = Hamiltonian(physics.picture, physics.dims)
    fixed_hamiltonian = None
    if couplings is None:
        fixed_hamiltonian = hamiltonian.to_torch(
            physics.params, time=time_train.detach()
        )

    weights = training.loss_weights
    history = TrainingHistory()
    for epoch in tqdm(range(training.epochs), disable=not training.progress_bar):
        optimizer.zero_grad()

        nn_state = state(network_time)
        if couplings is None:
            H_ = fixed_hamiltonian
        else:
            coupling_strength = couplings()
            H_ = hamiltonian.to_torch(
                physics.params, couplings=coupling_strength, time=time_train.detach()
            )

        loss_ic_value = loss_ic(nn_state, sim_state)
        loss_norm_value = loss_norm(nn_state)
        loss_data_value = loss_data(nn_state, data.observables, sim_expect, indices)
        loss_ode_value = loss_ode(H_, nn_state, network_time, time_scale=time_scale)

        total_loss = (
            weights.ic * loss_ic_value
            + weights.norm * loss_norm_value
            + weights.data * loss_data_value
            + weights.ode * loss_ode_value
        )

        total_loss.backward()
        optimizer.step()

        history.total_loss.append(float(total_loss.item()))
        history.loss_ic.append(float(loss_ic_value.item()))
        history.loss_norm.append(float(loss_norm_value.item()))
        history.loss_data.append(float(loss_data_value.item()))
        history.loss_ode.append(float(loss_ode_value.item()))
        if couplings is not None:
            history.couplings.append(coupling_strength.detach().tolist())

        if (epoch + 1) % training.log_every == 0 or epoch + 1 == training.epochs:
            _log_epoch(epoch, training.epochs, history)
            if n_train < len(time):
                fidelity, expect_mae = _validation_metrics(
                    state,
                    validation_time,
                    data.states[n_train:],
                    data.expectations[n_train:],
                    data.observables,
                )
                history.validation_epochs.append(epoch + 1)
                history.validation_fidelity.append(fidelity)
                history.validation_expect_mae.append(expect_mae)
                logger.info(
                    "  validation: fidelity %.4f, <O> MAE %.3e", fidelity, expect_mae
                )

    return TrainingResult(config, state, couplings, history)


def _log_epoch(epoch, epochs, history):
    couplings = f", couplings {history.couplings[-1]}" if history.couplings else ""
    logger.info(
        "epoch %d/%d: total %.3e (ic %.3e, norm %.3e, data %.3e, ode %.3e)%s",
        epoch + 1,
        epochs,
        history.total_loss[-1],
        history.loss_ic[-1],
        history.loss_norm[-1],
        history.loss_data[-1],
        history.loss_ode[-1],
        couplings,
    )
