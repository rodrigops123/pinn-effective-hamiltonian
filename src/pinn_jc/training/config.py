"""Experiment configuration: everything a training run depends on, validated up front.

Configs are frozen dataclasses that round-trip through plain dicts (``to_dict`` /
``from_dict``), so they can be stored next to results and loaded from YAML.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from pinn_jc.physics import INITIAL_STATES, get_picture
from pinn_jc.physics.basis import TRUNCATIONS, basis_indices

ACTIVATIONS = ("sine", "tanh")
LR_SCHEDULES = ("constant", "cosine", "exponential")
ODE_REDUCTIONS = ("sum", "mean")


@dataclass(frozen=True)
class PhysicsConfig:
    """The system that generates the data and the picture the PINN learns in.

    ``params`` holds the true parameters used to simulate the data: ``wc``, ``wa`` and
    the picture's couplings. With ``learn_couplings`` the couplings are learned (their
    true values are only used for the data); otherwise they are fixed to ``params``.
    ``truncation`` selects the basis the PINN works in (``physics.basis``).
    ``fixed_couplings`` keeps the named couplings at the given values while the others
    are learned; ``coupling_init`` starts the named learned couplings at the given
    values instead of a random draw. ``coupling_draw`` is ``Couplings``' ``draw``:
    "softplus" starts abs couplings from the same values as softplus ones.
    """

    picture: str
    params: dict[str, float]
    init_state: str
    dims: dict[str, int] = field(default_factory=lambda: {"atom": 2, "field": 2})
    init_state_options: dict[str, float] = field(default_factory=dict)
    learn_couplings: bool = True
    positivity: str = "softplus"
    truncation: str = "field"
    fixed_couplings: dict[str, float] = field(default_factory=dict)
    coupling_init: dict[str, float] = field(default_factory=dict)
    coupling_draw: str = "positivity"

    def __post_init__(self):
        picture = get_picture(self.picture)
        missing = {"wc", "wa", *picture.couplings} - set(self.params)
        if missing:
            raise ValueError(
                f"params for picture {self.picture!r} miss {sorted(missing)}"
            )
        if self.init_state not in INITIAL_STATES:
            raise ValueError(
                f"unknown initial state {self.init_state!r}; "
                f"choose from {sorted(INITIAL_STATES)}"
            )
        if set(self.dims) != {"atom", "field"} or min(self.dims.values()) < 1:
            raise ValueError(f"dims must be {{'atom': n, 'field': m}}, got {self.dims}")
        if self.positivity not in ("softplus", "abs"):
            raise ValueError(
                f"positivity must be 'softplus' or 'abs', got {self.positivity!r}"
            )
        if self.truncation not in TRUNCATIONS:
            raise ValueError(
                f"truncation must be one of {TRUNCATIONS}, got {self.truncation!r}"
            )
        for name, values in (
            ("fixed_couplings", self.fixed_couplings),
            ("coupling_init", self.coupling_init),
        ):
            unknown = set(values) - set(picture.couplings)
            if unknown:
                raise ValueError(
                    f"{name} names {sorted(unknown)}, not couplings of "
                    f"{self.picture!r} {list(picture.couplings)}"
                )
            if values and not self.learn_couplings:
                raise ValueError(f"{name} needs learn_couplings")
            if any(value < 0 for value in values.values()):
                raise ValueError(f"{name} must be non-negative, got {values}")
        if self.coupling_draw not in ("positivity", "softplus"):
            raise ValueError(
                "coupling_draw must be 'positivity' or 'softplus', "
                f"got {self.coupling_draw!r}"
            )
        if set(self.fixed_couplings) & set(self.coupling_init):
            raise ValueError("a coupling cannot be both fixed and initialised")
        if self.positivity == "softplus" and 0 in self.coupling_init.values():
            raise ValueError("softplus couplings cannot start at 0")

    @property
    def hilbert_dim(self) -> int:
        """Dimension D of the (truncated) atom-cavity Hilbert space."""
        return len(basis_indices(self.dims, self.truncation))

    @property
    def n_couplings(self) -> int:
        """Number of learned couplings (0 when they are fixed)."""
        return len(get_picture(self.picture).couplings) if self.learn_couplings else 0

    def coupling_overrides(self) -> tuple[list, list]:
        """Return the initial and fixed value of each learned coupling (None: not set)."""
        names = get_picture(self.picture).couplings if self.learn_couplings else ()
        return (
            [self.coupling_init.get(name) for name in names],
            [self.fixed_couplings.get(name) for name in names],
        )


@dataclass(frozen=True)
class DataConfig:
    """Time grid, data points and the held-out validation window.

    The last ``validation_fraction`` of the time grid is held out: no loss term sees
    it, so it measures extrapolation in time. ``n_data_points`` follows
    ``trainer.data_indices`` (None or 1: first instant only). With ``scale_time`` the
    network sees τ = t / tfinal instead of t.
    """

    tfinal: float
    n_time_steps: int
    n_data_points: int | None = None
    validation_fraction: float = 0.0
    scale_time: bool = False
    data_seed: int = 42

    def __post_init__(self):
        if self.tfinal <= 0:
            raise ValueError(f"tfinal must be positive, got {self.tfinal}")
        if self.n_time_steps < 2:
            raise ValueError(
                f"n_time_steps must be at least 2, got {self.n_time_steps}"
            )
        if not 0 <= self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )
        if self.n_data_points is not None and self.n_data_points < 1:
            raise ValueError(
                f"n_data_points must be None or >= 1, got {self.n_data_points}"
            )

    @property
    def n_train(self) -> int:
        """Number of leading time points used for training."""
        return self.n_time_steps - int(self.n_time_steps * self.validation_fraction)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the real- and imaginary-part networks.

    With ``hard_initial_condition`` the state is ψ(t) = ψ0 + (t - t0) net(t), which
    satisfies the initial condition exactly instead of through ``loss_ic``. Both
    networks see ``input_scale`` × time (``MLP``).
    """

    hidden_units: tuple[int, ...] = (100, 100, 100)
    activation: str = "sine"
    dropout: float = 0.0
    hard_initial_condition: bool = False
    input_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "hidden_units", tuple(self.hidden_units))
        if not self.hidden_units or min(self.hidden_units) < 1:
            raise ValueError(
                f"hidden_units must be positive widths, got {self.hidden_units}"
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {self.activation!r}"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.input_scale <= 0:
            raise ValueError(f"input_scale must be positive, got {self.input_scale}")


@dataclass(frozen=True)
class LossWeights:
    """Weights of the loss terms in the total loss."""

    ic: float = 1.0
    norm: float = 1.0
    data: float = 1.0
    ode: float = 1.0

    def __post_init__(self):
        if min(self.ic, self.norm, self.data, self.ode) < 0:
            raise ValueError(f"loss weights must be non-negative, got {self}")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings. ``seed`` (if set) seeds torch, numpy and random first.

    ``lr_schedule`` decays the learning rate from ``learning_rate`` to
    ``final_learning_rate`` over the epochs ("cosine" or "exponential"; "constant"
    keeps it fixed). With ``window_epochs`` > 0 the losses see a growing time window:
    the first ``window_start`` fraction of the training grid at epoch 0, widening
    linearly to the whole grid at epoch ``window_epochs``. ``ode_reduction`` combines
    the per-component ODE residuals: "sum" over the D components, or "mean", which
    keeps the weight of the ODE loss relative to the others independent of D.
    """

    epochs: int
    learning_rate: float = 0.01
    amsgrad: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)
    seed: int | None = None
    log_every: int = 1000
    progress_bar: bool = True
    lr_schedule: str = "constant"
    final_learning_rate: float = 1e-4
    window_start: float = 1.0
    window_epochs: int = 0
    ode_reduction: str = "sum"

    def __post_init__(self):
        if isinstance(self.loss_weights, dict):
            object.__setattr__(self, "loss_weights", LossWeights(**self.loss_weights))
        if self.epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}"
            )
        if self.final_learning_rate <= 0:
            raise ValueError(
                f"final_learning_rate must be positive, got {self.final_learning_rate}"
            )
        if not 0 < self.window_start <= 1:
            raise ValueError(f"window_start must be in (0, 1], got {self.window_start}")
        if self.ode_reduction not in ODE_REDUCTIONS:
            raise ValueError(
                f"ode_reduction must be one of {ODE_REDUCTIONS}, "
                f"got {self.ode_reduction!r}"
            )
        if self.window_epochs < 0:
            raise ValueError(
                f"window_epochs must be non-negative, got {self.window_epochs}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete specification of one training run."""

    physics: PhysicsConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain, YAML-serializable dicts and lists."""
        config = asdict(self)
        config["model"]["hidden_units"] = list(self.model.hidden_units)
        return config

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ExperimentConfig":
        """Build and validate a config from plain dicts (inverse of ``to_dict``)."""
        unknown = set(config) - {"physics", "data", "training", "model"}
        if unknown:
            raise ValueError(f"unknown config sections {sorted(unknown)}")
        return cls(
            physics=PhysicsConfig(**config["physics"]),
            data=DataConfig(**config["data"]),
            training=TrainingConfig(**config["training"]),
            model=ModelConfig(**config.get("model", {})),
        )
