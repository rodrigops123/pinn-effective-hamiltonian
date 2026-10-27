import numpy as np
import pytest

from pinn_jc.physics import INITIAL_STATES, initial_state


@pytest.mark.parametrize("name", sorted(INITIAL_STATES))
def test_initial_states_are_normalized_kets_of_the_right_size(name):
    dims = {"atom": 2, "field": 5}
    state = initial_state(name, dims)

    assert state.isket
    assert state.shape == (10, 1)
    assert np.isclose(state.norm(), 1.0)


def test_fock_state_is_excited_atom_and_empty_cavity():
    field = 4
    amplitudes = initial_state("fock", {"atom": 2, "field": field}).full().ravel()

    expected = np.zeros(2 * field)
    expected[field] = 1.0  # |e,0> comes right after the |g,n> block
    np.testing.assert_allclose(amplitudes, expected)


def test_fock_superposition_spreads_over_three_photon_numbers():
    amplitudes = (
        initial_state("fock_superposition", {"atom": 2, "field": 4}).full().ravel()
    )

    np.testing.assert_allclose(np.abs(amplitudes[:3]) ** 2, [1 / 3] * 3)
    np.testing.assert_allclose(amplitudes[3:], 0)


def test_state_options_are_forwarded():
    vacuum = (
        initial_state("coherent", {"atom": 2, "field": 3}, alpha=0.0).full().ravel()
    )

    np.testing.assert_allclose(vacuum, [1, 0, 0, 0, 0, 0])


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError, match="unknown initial state 'thermal'"):
        initial_state("thermal", {"atom": 2, "field": 3})


def test_too_small_field_is_rejected():
    with pytest.raises(ValueError, match="needs a field dimension >= 3, got 2"):
        initial_state("fock_superposition", {"atom": 2, "field": 2})


def test_single_level_atom_is_rejected():
    with pytest.raises(ValueError, match="at least two atomic levels"):
        initial_state("fock", {"atom": 1, "field": 3})
