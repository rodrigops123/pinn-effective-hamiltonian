import numpy as np
import pytest
import qutip
import torch

from pinn_jc.physics import PICTURES, Hamiltonian, get_picture

DIMS = {"atom": 2, "field": 3}
PARAMS = {"wc": 1.3, "wa": 1.0, "g0": 0.2, "g1": 0.7, "g2": 0.4, "g3": 0.1, "g4": 0.05}
STATIC_PICTURES = sorted(name for name, p in PICTURES.items() if not p.time_dependent)
TIME = torch.linspace(0, 3, 11).reshape(-1, 1)


def _to_torch(qobj):
    return torch.tensor(qobj.full(), dtype=torch.complex64)


@pytest.mark.parametrize("name", sorted(PICTURES))
def test_couplings_are_the_named_non_frequency_coefficients(name):
    picture = PICTURES[name]
    named = {t.coefficient for t in picture.terms if isinstance(t.coefficient, str)}
    assert named - {"wc", "wa"} == set(picture.couplings)


@pytest.mark.parametrize("name", STATIC_PICTURES)
def test_static_hamiltonian_is_hermitian_and_torch_matches_qutip(name):
    # "rabi" gives each process its own coupling; it is Hermitian only for g1 = g2, g3 = g4.
    params = PARAMS | (
        {"g2": PARAMS["g1"], "g4": PARAMS["g3"]} if name == "rabi" else {}
    )
    hamiltonian = Hamiltonian(name, DIMS)
    qobj = hamiltonian.to_qobj(params)

    assert qobj.isherm
    torch.testing.assert_close(hamiltonian.to_torch(params), _to_torch(qobj))


@pytest.mark.parametrize("name", STATIC_PICTURES)
def test_learned_couplings_override_params(name):
    picture = get_picture(name)
    learned = {g: 0.1 * (i + 1) for i, g in enumerate(picture.couplings)}
    hamiltonian = Hamiltonian(name, DIMS)

    from_tensor = hamiltonian.to_torch(
        PARAMS, couplings=torch.tensor(list(learned.values()))
    )

    torch.testing.assert_close(
        from_tensor, _to_torch(hamiltonian.to_qobj(PARAMS | learned))
    )


def test_interaction_picture_rabi_is_hermitian_at_all_times():
    hamiltonian = Hamiltonian("rabi2_ip", DIMS).to_torch(PARAMS, time=TIME)

    assert hamiltonian.shape == (len(TIME), 6, 6)
    torch.testing.assert_close(hamiltonian, hamiltonian.mH)


def test_interaction_picture_rabi_is_the_rotated_lab_coupling():
    """H_I(t) = exp(+i H0 t) (H_rabi2 - H0) exp(-i H0 t), computed independently."""
    free = PARAMS["wc"] * qutip.tensor(qutip.qeye(2), qutip.num(3)) + PARAMS[
        "wa"
    ] * qutip.tensor(qutip.num(2), qutip.qeye(3))
    coupling = Hamiltonian("rabi2", DIMS).to_qobj(PARAMS) - free

    expected = np.stack(
        [
            ((1j * free * t).expm() * coupling * (-1j * free * t).expm()).full()
            for t in TIME.ravel().tolist()
        ]
    )
    actual = Hamiltonian("rabi2_ip", DIMS).to_torch(PARAMS, time=TIME)

    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-5)


def test_unknown_picture_is_rejected():
    with pytest.raises(ValueError, match="unknown picture 'lab'"):
        Hamiltonian("lab", DIMS)


def test_wrong_number_of_couplings_is_rejected():
    with pytest.raises(ValueError, match="has 2 coupling"):
        Hamiltonian("rabi2", DIMS).to_torch(PARAMS, couplings=torch.tensor([0.7]))


def test_missing_parameter_is_rejected():
    with pytest.raises(ValueError, match="missing Hamiltonian parameter 'g2'"):
        Hamiltonian("rabi2", DIMS).to_qobj({"wc": 1.0, "wa": 1.0, "g1": 0.5})


def test_time_dependent_picture_needs_time_and_has_no_qobj():
    hamiltonian = Hamiltonian("rabi2_ip", DIMS)
    with pytest.raises(ValueError, match="needs the time grid"):
        hamiltonian.to_torch(PARAMS)
    with pytest.raises(NotImplementedError):
        hamiltonian.to_qobj(PARAMS)
