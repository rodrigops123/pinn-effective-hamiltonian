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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import median_filter

from pinn_jc.evaluation import Evaluation
from pinn_jc.physics import coupling_label, get_picture
from pinn_jc.sweep import SweepPoint

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
EXACT = {"color": "0.75", "linewidth": 2.5}  # exact solution, drawn under predictions
# true coupling values: dark and dotted, unlike the dashed grey grid
TRUE_VALUE = {"color": "0.15", "linestyle": ":", "linewidth": 1.0, "zorder": 1}


def use_style(usetex: bool = True) -> None:
    """Apply the publication style (SciencePlots; needs LaTeX if ``usetex``)."""
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


def _epochs_in_thousands(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
    ax.set_xlabel(r"Epoch ($\times 10^3$)")


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running median over ``window`` points (edges repeat the end values)."""
    if window <= 1:
        return values
    return median_filter(values, size=2 * (window // 2) + 1, mode="nearest")


def _mark_validation(ax, evaluation: Evaluation) -> None:
    if evaluation.train_end is not None:
        scale = evaluation.time_unit[0][-1] / evaluation.time[-1]
        ax.axvline(
            evaluation.train_end * scale, color="k", linestyle=":", linewidth=0.8
        )


def plot_expectations(evaluation: Evaluation) -> Figure:
    """Predicted ⟨a†a⟩, ⟨σ+σ-⟩ over the exact ones (thick grey), and the absolute error.

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
        ax.plot(time, exact, **EXACT)
        ax.plot(time, predicted, color=colors[k], label=rf"${label}$")
        ax_error.plot(time, np.abs(predicted - exact), color=colors[k])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **EXACT))
    labels.append("exact")
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        borderaxespad=0.0,
        **LEGEND,
    )
    for axis in (ax, ax_error):
        _mark_validation(axis, evaluation)
    ax.set_ylabel("Expectation value")
    ax_error.set_ylabel("Abs. error")
    ax_error.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax_error.set_xlabel(time_label)
    return fig


def plot_fidelity(evaluation: Evaluation) -> Figure:
    """Infidelity 1 - F between the predicted and the exact state (log scale)."""
    time, time_label = evaluation.time_unit
    fig, ax = plt.subplots(dpi=300)
    infidelity = np.clip(1.0 - evaluation.fidelity, 1e-16, None)
    ax.plot(time, infidelity)
    _mark_validation(ax, evaluation)
    ax.set_yscale("log")
    # whole decades, so at least two labelled ticks
    low, high = np.log10(infidelity.min()), np.log10(infidelity.max())
    ax.set_ylim(10 ** np.floor(low), 10 ** max(np.ceil(high), np.floor(low) + 1))
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$1 - \mathcal{F}(|\tilde\psi(t)\rangle, |\psi(t)\rangle)$")
    return fig


def plot_populations(evaluation: Evaluation, threshold: float = 1e-4) -> Figure:
    """Basis-state populations |⟨s,n|ψ(t)⟩|², predicted over exact (log scale).

    Populations do not depend on the global phase nor on the frame (the pictures
    differ by diagonal phases), unlike the components themselves. Only the states
    whose exact population exceeds ``threshold`` at some time are shown. The axis
    starts a decade below ``threshold`` (at 1e-5 if ``threshold`` <= 0); predictions
    below it are drawn just above the bottom edge, so a state the model leaves empty
    stays visible.
    """
    floor = threshold / 10 if threshold > 0 else 1e-5
    time, time_label = evaluation.time_unit
    predicted = np.abs(evaluation.predicted_state) ** 2
    predicted /= predicted.sum(axis=1, keepdims=True)
    exact = np.abs(evaluation.exact_state) ** 2
    shown = np.flatnonzero(exact.max(axis=0) > threshold)
    fig, ax = plt.subplots(dpi=300)
    for i in shown:
        ax.plot(time, exact[:, i], **EXACT)
    colors = [c for c in plt.get_cmap("tab10").colors if len(set(c)) > 1]  # no grey
    for k, i in enumerate(shown):
        ax.plot(
            time,
            np.clip(predicted[:, i], 1.5 * floor, None),
            color=colors[k % len(colors)],
            label=evaluation.basis_labels[i],
        )
    _mark_validation(ax, evaluation)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **EXACT))
    labels.append("exact")
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        borderaxespad=0.0,
        **LEGEND,
    )
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.5)
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"Population $|\langle s,n|\psi(t)\rangle|^2$")
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
    """Real and imaginary parts of every basis component over time (diagnostic).

    Components carry the global phase, so their error can be large while the
    fidelity is close to 1; ``plot_populations`` is the phase-free alternative.

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
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("white")

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
            cmap=error_cmap if col == 2 else cmap,
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
        plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap),
        ax=axes[:, 2].tolist(),
        shrink=0.9,
        label="Absolute error",
    )
    return fig


def plot_losses(history, every: int = 1, smooth: int | None = None) -> Figure:
    """Loss terms per epoch (log scale): raw (faint) and a running median (solid).

    ``smooth`` is the median window in epochs (default: 1 % of the epochs);
    one epoch in ``every`` is plotted.
    """
    arrays = _history_arrays(history)
    fig, ax = plt.subplots(dpi=300)
    for key, label in LOSS_LABELS.items():
        if key in arrays and arrays[key].size:
            values = arrays[key]
            window = smooth if smooth is not None else values.size // 100
            epochs = np.arange(1, values.size + 1)
            (line,) = ax.plot(
                epochs[::every], values[::every], alpha=0.2, linewidth=0.5
            )
            ax.plot(
                epochs[::every],
                _smooth(values, window)[::every],
                color=line.get_color(),
                label=label,
            )
    ax.set_yscale("log")
    _epochs_in_thousands(ax)
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
    """Learned couplings per epoch, with their true values (dark dotted).

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

    targets = sorted(
        {float(true_values[n]) for n in names if true_values and n in true_values}
    )

    def true_lines(axis):
        for value in targets:
            axis.axhline(value, **TRUE_VALUE)

    true_lines(ax)
    _epochs_in_thousands(ax)
    ax.set_ylabel("Coupling")
    handles, labels = ax.get_legend_handles_labels()
    if targets:
        handles.append(Line2D([], [], **TRUE_VALUE))
        labels.append("true value")
    ax.legend(handles, labels, loc="lower right", **LEGEND)

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
        inset.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
        inset.ticklabel_format(axis="y", useOffset=False)
        inset.text(
            0.5,
            1.04,
            f"last {inset_last} epochs",
            transform=inset.transAxes,
            ha="center",
            va="bottom",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    return fig


def plot_coupling_trajectories(
    histories: Mapping[int, object],
    picture: str,
    true_values: Mapping[str, float] | None = None,
    selected: int | None = None,
    every: int = 1,
) -> Figure:
    """Learned couplings per epoch for every seed of an experiment, one panel each.

    ``histories`` maps seeds to training histories (any format ``plot_couplings``
    accepts). The ``selected`` seed (e.g. the one with the lowest final loss) is
    drawn bold and the others faint, so seeds that end in different minima stay
    visible as separate lines rather than being averaged into a band.
    """
    names = get_picture(picture).couplings
    couplings = {}
    for seed, history in histories.items():
        values = _history_arrays(history)["couplings"]
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError(
                f"seed {seed} has couplings of shape {values.shape}; picture "
                f"{picture!r} has {len(names)}: {names}"
            )
        couplings[seed] = values
    if selected is not None and selected not in couplings:
        raise ValueError(f"selected seed {selected} is not among {list(couplings)}")

    fig, axes = plt.subplots(
        len(names),
        1,
        sharex=True,
        squeeze=False,
        figsize=(3.5, 1.4 * len(names) + 0.6),
        dpi=300,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (ax, name) in enumerate(zip(axes[:, 0], names, strict=True)):
        color = colors[k % len(colors)]
        for seed, values in couplings.items():
            epochs = np.arange(1, len(values) + 1)[::every]
            bold = seed == selected
            ax.plot(
                epochs,
                values[::every, k],
                color=color,
                alpha=1.0 if bold else 0.35,
                linewidth=1.2 if bold else 0.6,
                zorder=3 if bold else 2,
            )
        if true_values is not None and name in true_values:
            ax.axhline(float(true_values[name]), **TRUE_VALUE)
        ax.set_ylabel(coupling_label(picture, name), fontsize=7)

    handles = [
        Line2D([], [], color="0.2", linewidth=0.6, alpha=0.35),
        Line2D([], [], **TRUE_VALUE),
    ]
    labels = [f"other seeds ({len(couplings) - (selected is not None)})", "true value"]
    if selected is not None:
        handles.insert(0, Line2D([], [], color="0.2", linewidth=1.2))
        labels.insert(0, f"seed {selected}")
    axes[0, 0].legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(labels),
        borderaxespad=0.0,
        **LEGEND,
    )
    _epochs_in_thousands(axes[-1, 0])
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


SWEEP_PANELS = {"rabi": r"Rabi data ($g_2 = g$)", "jc": r"JC data ($g_2 = 0$)"}


def plot_g2_sweep(points: Sequence[SweepPoint]) -> Figure:
    """Learned g2 against g/ω, one panel per data type (Rabi, JC).

    Filled markers are the seeds' selected starts, open grey ones the starts they
    beat; the dashed line is the true g2 and the shaded band the g2 the data cannot
    tell from it (``pinn_jc.sweep.resolution_band``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, dpi=300)
    for ax, (data, title) in zip(axes, SWEEP_PANELS.items(), strict=True):
        panel = sorted(
            (p for p in points if p.data == data), key=lambda p: p.g_over_omega
        )
        if not panel:
            ax.set_visible(False)
            continue
        x = np.array([p.g_over_omega for p in panel])
        ax.fill_between(
            x,
            [p.band[0] for p in panel],
            [p.band[1] for p in panel],
            color="lightgray",
            alpha=0.6,
            linewidth=0,
            label="unresolved by the data",
        )
        ax.plot(x, [p.g2_true for p in panel], "k--", linewidth=0.8, label="true")
        for k, p in enumerate(panel):
            ax.plot(
                np.full(len(p.rejected), p.g_over_omega),
                p.rejected,
                "o",
                markersize=3,
                markerfacecolor="none",
                markeredgecolor="gray",
                label="rejected start" if k == 0 else None,
            )
            ax.plot(
                np.full(len(p.selected), p.g_over_omega),
                p.selected,
                "o",
                markersize=3,
                color="C0",
                alpha=0.7,
                label="selected start" if k == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$g/\omega$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel(r"learned $g_2$")
    axes[-1].legend(loc="upper right", **LEGEND)
    return fig


def plot_g2_basins(
    panels: Mapping[str, Mapping[float, Sequence[float]]],
    g2_true: float = 1.0,
    full_above: float = 0.5,
) -> Figure:
    """Learned g2 against its initial value, one panel per data set.

    ``panels`` maps a panel title to {initial g2: learned g2 of every seed}. The
    initial values sit at equal spacing (labelled with their values) and seeds are
    spread slightly in x, so that equal outcomes stay countable; the number of seeds
    ending above ``full_above`` (the full model) is written over each start. The
    dashed line is the true g2.
    """
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.5 * len(panels), 2.8), sharey=True, dpi=300
    )
    axes = np.atleast_1d(axes)
    for ax, (title, starts) in zip(axes, panels.items(), strict=True):
        inits = sorted(starts)
        for k, init in enumerate(inits):
            learned = np.asarray(starts[init], dtype=float)
            ax.plot(
                k + np.linspace(-0.2, 0.2, len(learned)),
                learned,
                "o",
                markersize=2.5,
                color="C0",
                alpha=0.7,
            )
            full = int((learned > full_above).sum())
            ax.annotate(
                f"{full}/{len(learned)}",
                (k, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.axhline(g2_true, color="k", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(inits)), [f"{init:g}" for init in inits])
        ax.set_xlim(-0.6, len(inits) - 0.4)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlabel(r"initial $g_2$")
        ax.set_title(title, fontsize=8, pad=12)
    axes[0].set_ylabel(r"learned $g_2$")
    return fig
