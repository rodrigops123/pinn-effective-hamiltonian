from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pinn_jc.data.simulate import simulate  # noqa: E402
from pinn_jc.evaluation import evaluate, evaluate_run  # noqa: E402
from pinn_jc.experiments import load_experiment, run_experiment  # noqa: E402
from pinn_jc.io.runs import load_run  # noqa: E402
from pinn_jc.physics import Frame, coupling_label  # noqa: E402
from pinn_jc.training.config import PhysicsConfig  # noqa: E402
from pinn_jc.training.history import TrainingHistory  # noqa: E402
from pinn_jc.visualization import plots  # noqa: E402

RABI = PhysicsConfig(
    "rabi2_ip",
    {"wc": 1.0, "wa": 1.2, "g1": 0.8, "g2": 0.5},
    "fock",
    {"atom": 2, "field": 3},
)
TFINAL, N = 3.0, 60
CONFIGS = Path(__file__).parents[1] / "configs"


class FixedState(torch.nn.Module):
    """Stand-in for a trained network: returns given states, records its input."""

    def __init__(self, states):
        super().__init__()
        self.states = states
        self.inputs = None

    def forward(self, time):
        self.inputs = time
        return self.states


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_exact_states_in_the_training_frame_score_perfectly():
    exact = simulate(RABI.params, TFINAL, N, "fock", "rabi2_ip", RABI.dims).states

    evaluation = evaluate({"state": FixedState(exact)}, RABI, TFINAL, N)

    assert evaluation.frame is Frame.INTERACTION
    np.testing.assert_allclose(evaluation.fidelity, 1.0, atol=1e-5)
    np.testing.assert_allclose(
        evaluation.predicted_expectations, evaluation.exact_expectations, atol=1e-5
    )


def test_states_in_another_frame_are_not_mistaken_for_correct():
    """The old plots compared states in different frames; evaluation must not."""
    lab = simulate(RABI.params, TFINAL, N, "fock", "rabi2", RABI.dims).states

    evaluation = evaluate({"state": FixedState(lab)}, RABI, TFINAL, N)

    assert evaluation.fidelity.min() < 0.9
    # populations do not depend on the frame, which is why the bug went unnoticed
    np.testing.assert_allclose(
        evaluation.predicted_expectations, evaluation.exact_expectations, atol=1e-5
    )


def test_scaled_time_is_fed_to_the_network():
    exact = simulate(RABI.params, TFINAL, N, "fock", "rabi2_ip", RABI.dims).states
    model = FixedState(exact)

    evaluate({"state": model}, RABI, TFINAL, N, scale_time=True)

    assert model.inputs.max().item() == pytest.approx(1.0)


def test_labels_and_dimensionless_time():
    exact = simulate(RABI.params, TFINAL, N, "fock", "rabi2_ip", RABI.dims).states
    evaluation = evaluate({"state": FixedState(exact)}, RABI, TFINAL, N)

    assert evaluation.basis_labels[:2] == [r"$|g,0\rangle$", r"$|g,1\rangle$"]
    assert evaluation.basis_labels[3] == r"$|e,0\rangle$"
    time, label = evaluation.time_unit
    assert label == r"$g_1 t$"
    assert time[-1] == pytest.approx(0.8 * TFINAL)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    experiment = load_experiment(CONFIGS / "rabi2_g_equals_omega_fock.yaml")
    data = replace(experiment.config.data, validation_fraction=0.25)
    experiment = replace(experiment, config=replace(experiment.config, data=data))
    out = run_experiment(
        experiment.with_overrides(epochs=4, seeds=[1]),
        tmp_path_factory.mktemp("runs"),
        allow_dirty=True,
        progress_bar=False,
    )
    return load_run(out / "seed_1")


def test_evaluate_run_marks_the_validation_window(saved_run):
    evaluation = evaluate_run(saved_run, n_time_steps=50)

    data = saved_run.config.data
    assert evaluation.train_end == pytest.approx(
        data.tfinal * (data.n_train - 1) / (data.n_time_steps - 1)
    )
    assert evaluation.time.shape == (50,)


def test_every_figure_renders(saved_run, tmp_path):
    evaluation = evaluate_run(saved_run, n_time_steps=40)
    picture = saved_run.config.physics.picture

    figures = {
        "expectations": plots.plot_expectations(evaluation),
        "fidelity": plots.plot_fidelity(evaluation),
        "states": plots.plot_state_components(evaluation),
        "losses": plots.plot_losses(saved_run.history),
        "couplings": plots.plot_couplings(
            saved_run.history, picture, saved_run.config.physics.params, inset_last=2
        ),
        "positivity": plots.plot_positivity_transforms(),
    }

    assert all(isinstance(fig, Figure) for fig in figures.values())
    assert len(figures["states"].axes) == 8  # 6 panels + 2 colour bars
    saved = plots.save_figure(
        figures["fidelity"], tmp_path / "sub" / "fidelity", ["png", "pdf"]
    )
    assert [p.name for p in saved] == ["fidelity.png", "fidelity.pdf"]
    assert all(p.stat().st_size > 0 for p in saved)


def test_history_plots_accept_every_history_format():
    history = TrainingHistory(
        total_loss=[3.0, 2.0],
        loss_ic=[1.0, 0.5],
        loss_norm=[1.0, 0.5],
        loss_data=[0.5, 0.5],
        loss_ode=[0.5, 0.5],
        couplings=[[0.1], [0.2]],
    )
    legacy = history.to_legacy_dict()

    for source in (history, history.to_arrays(), legacy):
        plots.plot_losses(source)
        figure = plots.plot_couplings(source, "atom", {"g1": 1.0})
        labels = [line.get_label() for line in figure.axes[0].get_lines()]
        assert coupling_label("atom", "g1") in labels


def test_couplings_must_match_the_picture():
    history = {"couplings": np.zeros((5, 1))}

    with pytest.raises(ValueError, match="has 2"):
        plots.plot_couplings(history, "rabi2_ip")
