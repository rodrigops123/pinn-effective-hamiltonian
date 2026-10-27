from dataclasses import dataclass, field, fields

import numpy as np

from pinn_jc.models import ComplexStateNet, Couplings
from pinn_jc.training.config import ExperimentConfig


@dataclass
class TrainingHistory:
    """Per-epoch losses (unweighted terms; ``total_loss`` is the weighted sum) and the
    learned couplings, plus validation metrics every ``log_every`` epochs."""

    total_loss: list[float] = field(default_factory=list)
    loss_ic: list[float] = field(default_factory=list)
    loss_norm: list[float] = field(default_factory=list)
    loss_data: list[float] = field(default_factory=list)
    loss_ode: list[float] = field(default_factory=list)
    couplings: list[list[float]] = field(default_factory=list)
    validation_epochs: list[int] = field(default_factory=list)
    validation_fidelity: list[float] = field(default_factory=list)
    validation_expect_mae: list[float] = field(default_factory=list)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}

    def to_legacy_dict(self) -> dict[str, list]:
        """The ``loss_dict`` returned by the pre-Phase-5 trainers."""
        legacy = {
            "total_loss": self.total_loss,
            "loss_ic": self.loss_ic,
            "loss_norm": self.loss_norm,
            "loss_data": self.loss_data,
            "loss_ode": self.loss_ode,
        }
        if self.couplings:
            legacy["learned_param"] = self.couplings
        return legacy


@dataclass
class TrainingResult:
    config: ExperimentConfig
    state: ComplexStateNet
    couplings: Couplings | None
    history: TrainingHistory

    @property
    def models(self) -> dict:
        """``{"state", "couplings"}``, the format the plotting functions accept."""
        return {"state": self.state, "couplings": self.couplings}
