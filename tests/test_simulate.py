import numpy as np
import pytest
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.physics import Frame

DIMS = {"atom": 2, "field": 3}


@pytest.mark.parametrize(
    "picture, params, simulated_picture, frame",
    [
        ("full", {"wc": 1.0, "wa": 1.0, "g1": 0.5}, "interaction", Frame.LAB),
        ("full", {"wc": 1.2, "wa": 1.0, "g1": 0.5}, "full", Frame.LAB),
        ("rabi2", {"wc": 1.0, "wa": 1.0, "g1": 0.5, "g2": 0.3}, "rabi2", Frame.LAB),
        (
            "rabi2_ip",
            {"wc": 1.0, "wa": 1.2, "g1": 0.5, "g2": 0.3},
            "rabi2",
            Frame.INTERACTION,
        ),
        (
            "rabi2_ip",
            {"wc": 1.0, "wa": 1.0, "g1": 0.5, "g2": 0.0},
            "interaction",
            Frame.INTERACTION,
        ),
        ("atom", {"wc": 1.0, "wa": 1.0, "g1": 0.5}, "atom", Frame.ROTATING),
        # Bug G: "rabi" with g2 = 0 was integrated as "interaction", dropping g3 and g4.
        (
            "rabi",
            {"wc": 1.0, "wa": 1.0, "g1": 0.5, "g2": 0.0, "g3": 0.2, "g4": 0.2},
            "rabi",
            Frame.LAB,
        ),
    ],
)
def test_states_come_in_the_frame_of_the_requested_picture(
    picture, params, simulated_picture, frame
):
    data = simulate(params, 1.0, 5, "fock", picture, DIMS)

    assert data.picture == picture
    assert data.simulated_picture == simulated_picture
    assert data.frame is frame


def test_requested_frame_overrides_the_picture_frame():
    data = simulate(
        {"wc": 1.2, "wa": 1.0, "g1": 0.5}, 1.0, 5, "fock", "full", DIMS, "rotating"
    )

    assert data.frame is Frame.ROTATING


def test_expectations_match_the_returned_states_in_every_frame():
    params = {"wc": 1.2, "wa": 1.0, "g1": 0.5}
    for frame in Frame:
        data = simulate(params, 2.0, 20, "coherent", "full", DIMS, frame)

        from_states = torch.einsum(
            "ni,kij,nj->nk", data.states.conj(), data.observables, data.states
        ).real

        np.testing.assert_allclose(
            from_states.numpy(), data.expectations.numpy(), atol=1e-5
        )


def test_detuned_interaction_picture_is_rejected():
    with pytest.raises(ValueError, match="needs wc == wa"):
        simulate({"wc": 1.2, "wa": 1.0, "g1": 0.5}, 1.0, 5, "fock", "interaction", DIMS)


def test_unknown_frame_is_rejected():
    with pytest.raises(ValueError, match="'heisenberg' is not a valid Frame"):
        simulate({"wc": 1.0, "wa": 1.0, "g1": 0.5}, 1.0, 5, "fock", frame="heisenberg")
