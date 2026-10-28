"""Figures. Every function returns a matplotlib Figure and never shows or saves it.

Inputs are precomputed: an ``Evaluation`` (``pinn_jc.evaluation``) for the state and
expectation-value figures, a training history for the loss and coupling figures.
Call ``use_style()`` once for the publication style, and ``save_figure`` to export.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pinn_jc.evaluation import Evaluation
from pinn_jc.physics import coupling_label, get_picture

STYLE = ("science", "retro", "grid")
LEGEND = {
    "fontsize": 6,
    "framealpha": 0.9,
    "facecolor": "lightgray",
    "edgecolor": "gray",
}
LOSS_LABELS = {
    "total_loss": r"$L$",
    "loss_ic": r"$L_\mathrm{ic}$",
    "loss_norm": r"$L_\mathrm{norm}$",
    "loss_data": r"$L_\mathrm{data}$",
    "loss_ode": r"$L_\mathrm{eq}$",
}


def use_style(usetex: bool = True) -> None:
    """Apply the SciencePlots style used for the thesis figures (needs LaTeX if usetex)."""
    import scienceplots  # noqa: F401  (registers the "science" styles)

    plt.style.use(list(STYLE))
    plt.rcParams["text.usetex"] = usetex


def save_figure(
    fig: Figure,
    path: Path | str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save ``fig`` as ``path`` with each extension in ``formats``; returns the paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = [path.with_suffix(f".{fmt}") for fmt in formats]
    for target in paths:
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return paths


def _history_arrays(history) -> dict[str, np.ndarray]:
    """Accept a TrainingHistory, a loaded history dict, or a legacy loss_dict."""
    if hasattr(history, "to_arrays"):
        history = history.to_arrays()
    arrays = {key: np.asarray(value) for key, value in history.items()}
    if "learned_param" in arrays and "couplings" not in arrays:
        arrays["couplings"] = arrays.pop("learned_param")
    return arrays


def _mark_validation(ax, evaluation: Evaluation) -> None:
    if evaluation.train_end is not None:
        scale = evaluation.time_unit[0][-1] / evaluation.time[-1]
        ax.axvline(
            evaluation.train_end * scale, color="k", linestyle=":", linewidth=0.8
        )


def plot_expectations(evaluation: Evaluation) -> Figure:
    """Predicted (solid) and exact (dashed) ⟨a†a⟩, ⟨σ+σ-⟩, and their absolute error.

    A dotted line marks the end of the training window, if one was held out.
    """
    time, time_label = evaluation.time_unit
    fig, (ax, ax_error) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, dpi=300
    )
    colors = ("tab:blue", "tab:orange")
    for k, label in enumerate(evaluation.observable_labels):
        predicted = evaluation.predicted_expectations[:, k]
        exact = evaluation.exact_expectations[:, k]
        ax.plot(time, predicted, color=colors[k], label=rf"${label}_\mathrm{{NN}}$")
        ax.plot(
            time,
            exact,
            color=colors[k],
            linestyle="--",
            label=rf"${label}_\mathrm{{exact}}$",
        )
        ax_error.plot(
            time, np.abs(predicted - exact), color=colors[k], label=rf"${label}$"
        )
    for axis in (ax, ax_error):
        _mark_validation(axis, evaluation)
        axis.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **LEGEND
        )
    ax.set_ylabel("Populations")
    ax_error.set_ylabel("Error")
    ax_error.set_xlabel(time_label)
    return fig


def plot_fidelity(evaluation: Evaluation) -> Figure:
    time, time_label = evaluation.time_unit
    fig, ax = plt.subplots(dpi=300)
    ax.plot(time, evaluation.fidelity)
    _mark_validation(ax, evaluation)
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$\mathcal{F}(|\tilde\psi(t)\rangle, |\psi(t)\rangle)$")
    return fig


def _strips(
    columns: np.ndarray, major_breaks: Sequence[int], width=10, gap=3, major_gap=8
):
    """Lay out the (T, D) components as vertical strips separated by NaN gaps.

    Returns the image (T, width * D + gaps), the x centre of each strip, and the x
    positions of the major gaps (between atomic manifolds).
    """
    blocks, centres, breaks, x = [], [], [], 0
    n_time, n_dim = columns.shape
    for i in range(n_dim):
        blocks.append(np.repeat(columns[:, i : i + 1], width, axis=1))
        centres.append(x + width / 2)
        x += width
        if i < n_dim - 1:
            this_gap = major_gap if i + 1 in major_breaks else gap
            if i + 1 in major_breaks:
                breaks.append(x + this_gap / 2)
            blocks.append(np.full((n_time, this_gap), np.nan))
            x += this_gap
    return np.hstack(blocks), np.array(centres), breaks


def plot_state_components(evaluation: Evaluation) -> Figure:
    """Real and imaginary parts of every basis component over time.

    Columns: network prediction, exact solution, absolute error; rows: real, imaginary.
    The prediction and the exact solution share one colour scale, the errors another.
    """
    time, time_label = evaluation.time_unit
    predicted, exact = evaluation.predicted_state, evaluation.exact_state
    parts = {"R": (predicted.real, exact.real), "I": (predicted.imag, exact.imag)}
    panels = {
        (row, col): matrix
        for row, (pred, ex) in enumerate(parts.values())
        for col, matrix in enumerate((pred, ex, np.abs(ex - pred)))
    }
    amplitude = max(np.abs(panels[(r, c)]).max() for r in range(2) for c in range(2))
    amplitude_norm = TwoSlopeNorm(vcenter=0.0, vmin=-amplitude, vmax=amplitude)
    error_norm = Normalize(
        vmin=0.0, vmax=max(panels[(0, 2)].max(), panels[(1, 2)].max())
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    major_breaks = (
        [evaluation.physics.dims["field"]]
        if evaluation.physics.dims["atom"] == 2
        else []
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 6), sharex=True, sharey=True, dpi=300, layout="constrained"
    )
    for (row, col), matrix in panels.items():
        image, centres, breaks = _strips(matrix, major_breaks)
        axes[row, col].imshow(
            image,
            cmap=cmap,
            norm=error_norm if col == 2 else amplitude_norm,
            extent=[0, image.shape[1], time[0], time[-1]],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )
        for x in breaks:
            axes[row, col].axvline(x, color="k", linewidth=0.6, alpha=0.25)

    part_names = list(parts)
    for row in range(2):
        p = part_names[row]
        axes[row, 0].set_title(rf"$|\tilde\psi_{p}(t)\rangle$ (network)")
        axes[row, 1].set_title(rf"$|\psi_{p}(t)\rangle$ (exact)")
        axes[row, 2].set_title(
            rf"$\left||\psi_{p}\rangle - |\tilde\psi_{p}\rangle\right|$"
        )
        axes[row, 0].set_ylabel(time_label)

    labels = evaluation.basis_labels
    step = 1 if len(labels) <= 12 else max(1, len(labels) // 8)
    for ax in axes[1]:
        ax.set_xticks(centres[::step], labels[::step], rotation=90)
        ax.set_xticks(centres, minor=True)
        ax.set_xlabel(r"Basis component $|s,n\rangle$")

    fig.colorbar(
        plt.cm.ScalarMappable(norm=amplitude_norm, cmap=cmap),
        ax=axes[:, :2].ravel().tolist(),
        shrink=0.9,
        label="Component value",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=error_norm, cmap=cmap),
        ax=axes[:, 2].tolist(),
        shrink=0.9,
        label="Absolute error",
    )
    return fig


def plot_losses(history, every: int = 1) -> Figure:
    """Loss terms per epoch (log scale), plotting one epoch in ``every``."""
    arrays = _history_arrays(history)
    fig, ax = plt.subplots(dpi=300)
    for key, label in LOSS_LABELS.items():
        if key in arrays and arrays[key].size:
            epochs = np.arange(1, arrays[key].size + 1)
            ax.plot(epochs[::every], arrays[key][::every], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", **LEGEND)
    return fig


def plot_couplings(
    history,
    picture: str,
    true_values: Mapping[str, float] | None = None,
    every: int = 1,
    inset_last: int | None = 1000,
) -> Figure:
    """Learned couplings per epoch, with their true values (dashed, same colour).

    ``true_values`` maps coupling names to values (e.g. the config's ``params``);
    the inset zooms on the last ``inset_last`` epochs, without subsampling.
    """
    couplings = _history_arrays(history)["couplings"]
    names = get_picture(picture).couplings
    if couplings.ndim != 2 or couplings.shape[1] != len(names):
        raise ValueError(
            f"history has couplings of shape {couplings.shape}; picture {picture!r} "
            f"has {len(names)}: {names}"
        )
    epochs = np.arange(1, len(couplings) + 1)
    fig, ax = plt.subplots(dpi=300)
    lines = [
        ax.plot(
            epochs[::every], couplings[::every, k], label=coupling_label(picture, name)
        )[0]
        for k, name in enumerate(names)
    ]

    def true_lines(axis):
        for line, name in zip(lines, names, strict=True):
            if true_values is not None and name in true_values:
                axis.axhline(
                    true_values[name],
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=1,
                )

    true_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coupling")
    ax.legend(loc="lower right", **LEGEND)

    if inset_last and len(couplings) > inset_last:
        inset = inset_axes(
            ax,
            width="40%",
            height="40%",
            bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
            bbox_transform=ax.transAxes,
            loc="center right",
        )
        for k, line in enumerate(lines):
            inset.plot(
                epochs[-inset_last:], couplings[-inset_last:, k], color=line.get_color()
            )
        true_lines(inset)
        inset.tick_params(labelsize=6)
        inset.set_title(f"last {inset_last} epochs", fontsize=6)
    return fig


def plot_positivity_transforms(span: float = 10.0) -> Figure:
    """softplus(θ) and |θ| with their derivatives: the two ways couplings are kept ≥ 0."""
    theta = np.linspace(-span, span, 1001)
    fig, (ax_softplus, ax_abs) = plt.subplots(
        1, 2, figsize=(7, 2.6), sharey=True, dpi=300
    )
    ax_softplus.plot(theta, np.log1p(np.exp(theta)), label=r"softplus$(\theta)$")
    ax_softplus.plot(theta, 1 / (1 + np.exp(-theta)), label=r"softplus$'(\theta)$")
    ax_abs.plot(theta, np.abs(theta), label=r"$|\theta|$")
    ax_abs.plot(theta, np.sign(theta), label=r"$|\theta|'$")
    for ax in (ax_softplus, ax_abs):
        ax.set_xlabel(r"$\theta$")
        ax.legend(**LEGEND)
    return fig
