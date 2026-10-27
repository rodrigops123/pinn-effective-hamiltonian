import numpy as np
import pytest
import torch

from pinn_jc.data.simulate import simulate
from pinn_jc.physics import Frame

DIMS = {"atom": 2, "field": 3}


@pytest.mark.parametrize(
    "picture, params, simulated_picture, frame",
    [
        ("full", {"wc": 1.0, "wa": 1.0, "g1": 0.5}, "interaction", Frame.INTERACTION),
        ("full", {"wc": 1.2, "wa": 1.0, "g1": 0.5}, "full", Frame.LAB),
        ("rabi2", {"wc": 1.0, "wa": 1.0, "g1": 0.5, "g2": 0.3}, "rabi2", Frame.LAB),
        ("atom", {"wc": 1.0, "wa": 1.0, "g1": 0.5}, "atom", Frame.ROTATING),
    ],
)
def test_simulation_records_the_picture_actually_integrated(
    picture, params, simulated_picture, frame
):
    data = simulate(params, 1.0, 5, "fock", picture, DIMS, state_frame="interaction")

    assert data.picture == picture
    assert data.simulated_picture == simulated_picture
    assert data.simulation_frame is frame
    assert data.rotated_to_interaction_frame


def test_expectations_match_the_returned_states():
    data = simulate(
        {"wc": 1.2, "wa": 1.0, "g1": 0.5}, 2.0, 20, "coherent", "full", DIMS
    )

    from_states = torch.einsum(
        "ni,kij,nj->nk", data.states.conj(), data.observables, data.states
    ).real

    np.testing.assert_allclose(
        from_states.numpy(), data.expectations.numpy(), atol=1e-5
    )


def test_unknown_state_frame_is_rejected():
    with pytest.raises(ValueError, match="state_frame must be"):
        simulate(
            {"wc": 1.0, "wa": 1.0, "g1": 0.5}, 1.0, 5, "fock", state_frame="rotating"
        )
