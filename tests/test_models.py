import pytest
import torch

from pinn_jc.io.legacy import LegacyNeuralNet, as_models
from pinn_jc.models import MLP, ComplexStateNet, Couplings, Sine
from pinn_jc.training.config import ModelConfig
from pinn_jc.training.trainer import build_models

TIME = torch.linspace(0, 5, 17).reshape(-1, 1)


def test_mlp_maps_times_to_state_components():
    mlp = MLP(in_features=1, out_features=6, hidden_units=[8, 8, 8])

    assert mlp(TIME).shape == (17, 6)
    assert len(mlp.hidden_layers) == 3


def test_mlp_rejects_mismatched_activations():
    with pytest.raises(ValueError, match="2 activations for 3 hidden layers"):
        MLP(1, 4, [8, 8, 8], activation=[Sine(), Sine()])


def test_mlp_dropout_is_active_only_in_training():
    mlp = MLP(1, 4, [32], dropout=0.5)

    mlp.eval()
    torch.testing.assert_close(mlp(TIME), mlp(TIME))
    mlp.train()
    assert not torch.equal(mlp(TIME), mlp(TIME))


def test_complex_state_combines_real_and_imaginary_networks():
    real, imag = MLP(1, 4, [8]), MLP(1, 4, [8])

    psi = ComplexStateNet(real, imag)(TIME)

    assert psi.is_complex()
    torch.testing.assert_close(psi.real, real(TIME))
    torch.testing.assert_close(psi.imag, imag(TIME))


@pytest.mark.parametrize("positivity", ["softplus", "abs"])
def test_couplings_are_non_negative(positivity):
    couplings = Couplings(3, positivity)
    with torch.no_grad():
        couplings.raw.copy_(torch.tensor([-2.0, 0.0, 1.5]))

    values = couplings()

    assert values.shape == (3,)
    assert (values >= 0).all()
    expected = (
        torch.nn.functional.softplus(couplings.raw)
        if positivity == "softplus"
        else couplings.raw.abs()
    )
    torch.testing.assert_close(values, expected)


@pytest.mark.parametrize("positivity", ["softplus", "abs"])
def test_couplings_start_at_the_given_initial_values(positivity):
    couplings = Couplings(3, positivity, initial=[None, 0.05, 2.0])

    values = couplings()

    torch.testing.assert_close(values[1:], torch.tensor([0.05, 2.0]))


def test_fixed_couplings_stay_fixed_and_get_no_gradient():
    couplings = Couplings(2, fixed=[None, 0.3])

    couplings().sum().backward()

    assert couplings()[1].item() == pytest.approx(0.3)
    assert couplings.raw.grad[1] == 0
    assert couplings.raw.grad[0] != 0
    assert set(couplings.state_dict()) == {"raw"}


def test_coupling_options_do_not_change_the_random_stream():
    torch.manual_seed(0)
    Couplings(2)
    expected_next = torch.rand(1)
    torch.manual_seed(0)

    Couplings(2, initial=[0.1, None], fixed=[None, 1.0])

    assert torch.equal(torch.rand(1), expected_next)


def test_couplings_reject_unknown_positivity():
    with pytest.raises(ValueError, match="positivity must be"):
        Couplings(2, "relu")


def test_input_scale_multiplies_the_input_and_adds_no_parameters():
    torch.manual_seed(0)
    plain = build_models(4, model=ModelConfig(hidden_units=(16,)))[0]
    torch.manual_seed(0)
    scaled = build_models(4, model=ModelConfig(hidden_units=(16,), input_scale=3.0))[0]

    torch.testing.assert_close(scaled(TIME), plain(3.0 * TIME))
    assert scaled.state_dict().keys() == plain.state_dict().keys()


def test_build_models_reproduces_the_legacy_initialisation():
    """Same seed, same weights and couplings as the legacy network construction."""
    torch.manual_seed(7)
    legacy_real = LegacyNeuralNet(
        [16, 16], Sine(), output=4, create_parameter=True, n_paramater=2
    )
    legacy_imag = LegacyNeuralNet([16, 16], Sine(), output=4)

    torch.manual_seed(7)
    state, couplings = build_models(
        4, n_couplings=2, model=ModelConfig(hidden_units=(16, 16))
    )

    torch.testing.assert_close(couplings.raw, legacy_real.param, rtol=0, atol=0)
    torch.testing.assert_close(state.real(TIME), legacy_real(TIME), rtol=0, atol=0)
    torch.testing.assert_close(state.imag(TIME), legacy_imag(TIME), rtol=0, atol=0)


def test_legacy_models_convert_exactly_without_touching_the_rng():
    legacy = {
        "model_real": LegacyNeuralNet(
            [16, 16], Sine(), output=4, create_parameter=True, n_paramater=2
        ),
        "model_imag": LegacyNeuralNet([16, 16], Sine(), output=4),
    }
    torch.manual_seed(3)
    expected_next = torch.rand(1)
    torch.manual_seed(3)

    models = as_models(legacy)

    assert torch.equal(torch.rand(1), expected_next)
    torch.testing.assert_close(
        models["state"](TIME),
        legacy["model_real"](TIME) + 1j * legacy["model_imag"](TIME),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(models["couplings"].raw, legacy["model_real"].param)


def test_current_models_pass_through_unchanged():
    state, couplings = build_models(4, n_couplings=1)
    models = {"state": state, "couplings": couplings}

    converted = as_models(models)

    assert converted["state"] is state
    assert converted["couplings"] is couplings
