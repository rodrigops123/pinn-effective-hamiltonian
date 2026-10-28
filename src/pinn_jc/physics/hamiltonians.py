"""Hamiltonian registry: every picture is defined once, as a sum of terms.

A term is ``coefficient * operator * exp(+i frequency t)``. The same definition builds
the QuTiP operator used to simulate data and the torch tensor used in the ODE loss,
so the two can no longer drift apart.

Coefficients and frequencies are either a parameter name (looked up in ``params``,
or in the learnable ``couplings`` tensor when the name is one of the picture's
couplings) or a function of ``params``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import qutip
import torch

from pinn_jc import settings
from pinn_jc.physics.frames import Frame
from pinn_jc.physics.operators import operator

Parameters = Mapping[str, float]
Coefficient = str | Callable[[Parameters], float]


def detuning(params: Parameters) -> float:
    """Δ = ωc - ωa."""
    return params["wc"] - params["wa"]


def frequency_sum(params: Parameters) -> float:
    """Σ = ωc + ωa."""
    return params["wc"] + params["wa"]


def _negative(
    frequency: Callable[[Parameters], float],
) -> Callable[[Parameters], float]:
    return lambda params: -frequency(params)


@dataclass(frozen=True)
class Term:
    coefficient: Coefficient
    operator: str
    frequency: Callable[[Parameters], float] | None = None


@dataclass(frozen=True)
class Picture:
    name: str
    description: str
    frame: Frame
    terms: tuple[Term, ...]
    couplings: tuple[str, ...]  # learnable coefficients, in coupling-tensor order

    @property
    def time_dependent(self) -> bool:
        return any(term.frequency is not None for term in self.terms)


_FREE = (Term("wc", "photon_number"), Term("wa", "atom_excitation"))
_JC = (Term("g1", "jc_emission"), Term("g1", "jc_absorption"))
_COUNTER_ROTATING = (Term("g2", "cr_lowering"), Term("g2", "cr_raising"))

PICTURES = {
    picture.name: picture
    for picture in (
        Picture(
            "interaction",
            "Resonant Jaynes-Cummings in the interaction picture: g1 (a†σ- + aσ+)",
            Frame.INTERACTION,
            _JC,
            couplings=("g1",),
        ),
        Picture(
            "atom",
            "Jaynes-Cummings rotating at ωa for both subsystems: Δ a†a + g1 (a†σ- + aσ+)",
            Frame.ROTATING,
            (Term(detuning, "photon_number"), *_JC),
            couplings=("g1",),
        ),
        Picture(
            "full",
            "Lab-frame Jaynes-Cummings: ωc a†a + ωa σ+σ- + g1 (a†σ- + aσ+)",
            Frame.LAB,
            (*_FREE, *_JC),
            couplings=("g1",),
        ),
        Picture(
            "rabi",
            "Lab-frame Rabi with one coupling per process: "
            "g1 a†σ- + g2 aσ+ + g3 aσ- + g4 a†σ+. "
            "Hermitian only when g1 = g2 and g3 = g4",
            Frame.LAB,
            (
                *_FREE,
                Term("g1", "jc_emission"),
                Term("g2", "jc_absorption"),
                Term("g3", "cr_lowering"),
                Term("g4", "cr_raising"),
            ),
            couplings=("g1", "g2", "g3", "g4"),
        ),
        Picture(
            "rabi2",
            "Lab-frame Rabi: ωc a†a + ωa σ+σ- + g1 (a†σ- + aσ+) + g2 (aσ- + a†σ+)",
            Frame.LAB,
            (*_FREE, *_JC, *_COUNTER_ROTATING),
            couplings=("g1", "g2"),
        ),
        Picture(
            "rabi2_ip",
            "Rabi in the interaction picture w.r.t. H0 = ωc a†a + ωa σ+σ-: "
            "g1 (a†σ- e^{+iΔt} + aσ+ e^{-iΔt}) + g2 (aσ- e^{-iΣt} + a†σ+ e^{+iΣt})",
            Frame.INTERACTION,
            (
                Term("g1", "jc_emission", detuning),
                Term("g1", "jc_absorption", _negative(detuning)),
                Term("g2", "cr_lowering", _negative(frequency_sum)),
                Term("g2", "cr_raising", frequency_sum),
            ),
            couplings=("g1", "g2"),
        ),
        Picture(
            "geral",
            "Lab-frame Rabi plus two-photon and classical-drive terms: "
            "... + g3 (a + a†)² σx + g0 σx",
            Frame.LAB,
            (
                *_FREE,
                *_JC,
                *_COUNTER_ROTATING,
                Term("g3", "two_photon"),
                Term("g0", "atom_drive"),
            ),
            couplings=("g1", "g2", "g3", "g0"),
        ),
    )
}


def get_picture(name: str) -> Picture:
    if name not in PICTURES:
        raise ValueError(f"unknown picture {name!r}; choose from {sorted(PICTURES)}")
    return PICTURES[name]


# Physical process of a coupling, named by the set of operators it multiplies.
_PROCESSES = {
    frozenset({"jc_emission", "jc_absorption"}): "Jaynes-Cummings",
    frozenset({"cr_lowering", "cr_raising"}): "counter-rotating",
    frozenset({"two_photon"}): "two-photon",
    frozenset({"atom_drive"}): "classical drive",
}
_OPERATOR_SYMBOLS = {
    "jc_emission": r"a^\dagger\sigma_-",
    "jc_absorption": r"a\sigma_+",
    "cr_lowering": r"a\sigma_-",
    "cr_raising": r"a^\dagger\sigma_+",
}


def coupling_label(picture: str, coupling: str) -> str:
    """Plot label of a learnable coupling, e.g. ``$g_1$ (Jaynes-Cummings)``."""
    picture = get_picture(picture)
    if coupling not in picture.couplings:
        raise ValueError(f"picture {picture.name!r} has no coupling {coupling!r}")
    operators = frozenset(
        t.operator for t in picture.terms if t.coefficient == coupling
    )
    symbol = f"$g_{{{coupling[1:]}}}$"
    if operators in _PROCESSES:
        return f"{symbol} ({_PROCESSES[operators]})"
    return (
        f"{symbol} (${' + '.join(_OPERATOR_SYMBOLS[op] for op in sorted(operators))}$)"
    )


class Hamiltonian:
    """A picture's Hamiltonian for given dimensions, with operators built once."""

    def __init__(self, picture: str, dims: Mapping[str, int], device=settings.DEVICE):
        self.picture = get_picture(picture)
        self.dims = dict(dims)
        self.device = device
        names = {term.operator for term in self.picture.terms}
        self._qobj_operators = {name: operator(name, dims) for name in names}
        self._torch_operators = {
            name: torch.tensor(op.full(), dtype=torch.complex64, device=device)
            for name, op in self._qobj_operators.items()
        }

    @property
    def time_dependent(self) -> bool:
        return self.picture.time_dependent

    def to_qobj(self, params: Parameters) -> qutip.Qobj:
        """QuTiP operator with every coefficient taken from ``params``."""
        if self.time_dependent:
            raise NotImplementedError(
                f"picture {self.picture.name!r} is time dependent; use to_torch"
            )
        terms = [
            self._coefficient(term.coefficient, params, {})
            * self._qobj_operators[term.operator]
            for term in self.picture.terms
        ]
        return sum(terms[1:], terms[0])

    def to_torch(
        self,
        params: Parameters,
        couplings: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Torch Hamiltonian, shape (D, D), or (N, D, D) for time-dependent pictures.

        ``couplings`` holds the learnable coefficients in ``picture.couplings`` order
        and overrides ``params`` for those names; ``time`` has shape (N, 1).
        """
        learned = self._learned_couplings(couplings)
        if self.time_dependent:
            if time is None:
                raise ValueError(f"picture {self.picture.name!r} needs the time grid")
            time = time.to(dtype=torch.float32, device=self.device).view(-1, 1, 1)

        hamiltonian = None
        for term in self.picture.terms:
            value = (
                self._coefficient(term.coefficient, params, learned)
                * (self._torch_operators[term.operator])
            )
            if term.frequency is not None:
                value = (
                    torch.exp(1j * term.frequency(params) * time).to(torch.complex64)
                    * value
                )
            hamiltonian = value if hamiltonian is None else hamiltonian + value
        return hamiltonian

    def _learned_couplings(self, couplings: torch.Tensor | None) -> dict:
        if couplings is None:
            return {}
        couplings = couplings.reshape(-1)
        expected = len(self.picture.couplings)
        if couplings.numel() != expected:
            raise ValueError(
                f"picture {self.picture.name!r} has {expected} coupling(s) "
                f"{self.picture.couplings}, got a tensor with {couplings.numel()}"
            )
        return dict(zip(self.picture.couplings, couplings, strict=True))

    @staticmethod
    def _coefficient(coefficient: Coefficient, params: Parameters, learned: Mapping):
        if callable(coefficient):
            return coefficient(params)
        if coefficient in learned:
            return learned[coefficient]
        if coefficient not in params:
            raise ValueError(f"missing Hamiltonian parameter {coefficient!r}")
        return params[coefficient]
