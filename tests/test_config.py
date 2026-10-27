import pytest

from pinn_jc.training.config import (
    DataConfig,
    ExperimentConfig,
    LossWeights,
    ModelConfig,
    PhysicsConfig,
    TrainingConfig,
)

CONFIG = {
    "physics": {
        "picture": "rabi2_ip",
        "params": {"wc": 1.0, "wa": 1.0, "g1": 1.0, "g2": 1.0},
        "init_state": "coherent",
        "dims": {"atom": 2, "field": 4},
        "init_state_options": {"alpha": 0.5},
    },
    "data": {"tfinal": 6.28, "n_time_steps": 400, "validation_fraction": 0.25},
    "model": {"hidden_units": [64, 64], "activation": "tanh"},
    "training": {"epochs": 100, "seed": 3, "loss_weights": {"ode": 10.0}},
}


def test_config_round_trips_through_plain_dicts():
    config = ExperimentConfig.from_dict(CONFIG)

    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert config.model.hidden_units == (64, 64)
    assert config.training.loss_weights == LossWeights(ode=10.0)
    assert config.to_dict()["model"]["hidden_units"] == [64, 64]


def test_derived_quantities():
    config = ExperimentConfig.from_dict(CONFIG)

    assert config.physics.hilbert_dim == 8
    assert config.physics.n_couplings == 2
    assert config.data.n_train == 300


def test_fixed_couplings_have_no_learnable_parameters():
    physics = PhysicsConfig(
        "atom", {"wc": 1, "wa": 1, "g1": 1}, "fock", learn_couplings=False
    )

    assert physics.n_couplings == 0


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: PhysicsConfig("lab", {"wc": 1, "wa": 1}, "fock"), "unknown picture"),
        (
            lambda: PhysicsConfig("rabi2", {"wc": 1, "wa": 1, "g1": 1}, "fock"),
            "miss \\['g2'\\]",
        ),
        (
            lambda: PhysicsConfig("atom", {"wc": 1, "wa": 1, "g1": 1}, "thermal"),
            "unknown initial state",
        ),
        (
            lambda: PhysicsConfig(
                "atom", {"wc": 1, "wa": 1, "g1": 1}, "fock", positivity="exp"
            ),
            "positivity",
        ),
        (lambda: DataConfig(tfinal=0, n_time_steps=10), "tfinal"),
        (
            lambda: DataConfig(tfinal=1, n_time_steps=10, validation_fraction=1.0),
            "validation_fraction",
        ),
        (lambda: ModelConfig(hidden_units=()), "hidden_units"),
        (lambda: ModelConfig(activation="relu"), "activation"),
        (lambda: LossWeights(ode=-1), "non-negative"),
        (lambda: TrainingConfig(epochs=10, learning_rate=0), "learning_rate"),
        (
            lambda: ExperimentConfig.from_dict({**CONFIG, "optimizer": {}}),
            "unknown config sections",
        ),
    ],
)
def test_invalid_configs_are_rejected(build, message):
    with pytest.raises(ValueError, match=message):
        build()
