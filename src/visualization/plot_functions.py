import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scienceplots

plt.style.use(["science", "retro", "grid"])
import matplotlib.gridspec as gridspec
import numpy as np
from src.data_simulation.jaynes_cummings_data import data_jc
from src.model.train_and_eval import train_test_split


def prep_plot_input(
    params, tfinal, n_time_steps, init_state, picture, dims, plot_input
):

    sim_state, sim_expect, _, operators_list, time = data_jc(
        params=params,
        tfinal=tfinal,
        n_time_steps=n_time_steps,
        init_state=init_state,
        picture=picture,
        dims=dims,
    )

    time = torch.linspace(0, tfinal, n_time_steps)

    time = time.reshape(-1, 1)

    if plot_input == "expected":
        return sim_expect, time, operators_list

    else:
        return (
            sim_state,
            time,
        )


def set_plot_params_expected_values():

    fig = plt.figure(dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax0.grid(linestyle="--")
    ax0.set_ylabel(r"Populations")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.set_xlabel(r"\(gt\)")
    ax1.set_ylabel("Error")
    ax1.grid(linestyle="--", alpha=0.6)

    return ax0, ax1


def set_labels_and_colors_expected_values():
    labels = [
        [
            r"\(\langle \hat{a}^\dagger \hat{a} \rangle_{NN}\)",
            r"\(\langle \hat{a}^\dagger \hat{a} \rangle_{sim}\)",
        ],
        [
            r"\(\langle \hat{\sigma}_+ \hat{\sigma}_- \rangle_{NN}\)",
            r"\(\langle \hat{\sigma}_+ \hat{\sigma}_- \rangle_{sim}\)",
        ],
    ]

    labels_error = [
        r"error\((\hat{a}^\dagger \hat{a})\)",
        r"error\((\hat{\sigma}_+ \hat{\sigma}_-)\)",
    ]

    colors = ["blue", "orange"]
    colors_error = ["blue", "orange"]

    return labels, labels_error, colors, colors_error


def plot_expected_values(
    models_dict,
    tfinal,
    n_time_steps,
    init_state,
    params,
    picture,
    dims,
    is_scaled,
    plot_input="expected",
):
    """
    Plots the expected values of the model and optionally compares them with simulation data.

    Args:
        model_real: Real part of the model's output.
        model_imag: Imaginary part of the model's output.
        time: Time points for the x-axis.
        operator: Operator used to compute expected values.
    """

    sim_expect, time, operators_list = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    if is_scaled:
        time = time / time.max()

    ax0, ax1 = set_plot_params_expected_values()
    labels, labels_error, colors, colors_error = set_labels_and_colors_expected_values()

    for i, operator in enumerate(operators_list):
        nn_state_train = models_dict["model_real"](time) + 1j * models_dict[
            "model_imag"
        ](time)

        expected_values_train = torch.einsum(
            "ni,ij,nj->n", nn_state_train.conj(), operator, nn_state_train
        ).real

        error = np.abs((expected_values_train - sim_expect[:, i]).detach().numpy())

        ax0.plot(
            time.numpy().squeeze(),
            expected_values_train.detach().numpy(),
            label=labels[i][0],
            color=colors[i],
        )

        ax0.plot(
            time.numpy().squeeze(),
            sim_expect[:, i].detach().numpy(),
            label=labels[i][1],
            color=colors[i],
            linestyle="--",
        )

        ax0.legend(
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),  # push legend outside
            borderaxespad=0.0,
            framealpha=0.9,
            facecolor="lightgray",
            edgecolor="gray",
        )

        ax1.plot(
            time.numpy().squeeze(),
            error,
            label=labels_error[i],
            color=colors_error[i],
        )
        ax1.legend(
            fontsize=6,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),  # push legend outside
            borderaxespad=0.0,
            framealpha=0.9,
            facecolor="lightgray",
            edgecolor="gray",
        )

    plt.show()


def set_plot_params_states():
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=True, dpi=300)
    axs[0, 0].set_title(r"\(|\tilde{\psi}_R (t)\rangle\)", fontsize=18)
    axs[0, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[0, 0].yaxis.set_ticklabels([])

    axs[1, 0].set_title(r"\(|\psi_R(t)\rangle\)", fontsize=18)
    axs[1, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[1, 0].yaxis.set_ticklabels([])

    axs[2, 0].set_title(
        r"abs(\(|\psi_R(t)\rangle - |\tilde{\psi}_R(t)\rangle\))", fontsize=18
    )
    axs[2, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[2, 0].yaxis.set_ticklabels([])

    axs[0, 1].set_title(r"\(|\tilde{\psi}_I(t)\rangle\)", fontsize=18)
    axs[0, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[0, 1].yaxis.set_ticklabels([])

    axs[1, 1].set_title(r"\(|\psi_I(t)\rangle\)", fontsize=18)
    axs[1, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[1, 1].yaxis.set_ticklabels([])

    axs[2, 1].set_title(
        r"abs(\(|\psi_I(t)\rangle - |\tilde{\psi}_I(t)\rangle\))", fontsize=18
    )
    axs[2, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[2, 1].yaxis.set_ticklabels([])

    axs[2, 0].set_xlabel(r"\(gt\)", fontsize=16)
    axs[2, 1].set_xlabel(r"\(gt\)", fontsize=16)

    axs[2, 0].xaxis.set_tick_params(labelsize=14)
    axs[2, 1].xaxis.set_tick_params(labelsize=14)

    return fig, axs


# def plot_states(
#     models_dict,
#     params,
#     tfinal,
#     n_time_steps,
#     init_state,
#     picture,
#     dims,
#     is_scaled,
#     plot_input="state",
# ):

#     sim_state, time = prep_plot_input(
#         params, tfinal, n_time_steps, init_state, picture, dims, plot_input
#     )

#     if is_scaled:
#         time = time / time.max()

#     fig, axs = set_plot_params_states()

#     # ============================================================
#     # NEW: render each vector component as a "strip", with REAL gaps
#     # ============================================================
#     STRIP_PX = 6  # thickness of each strip
#     GAP_PX = 2  # small gap between strips
#     MAJOR_GAP_PX = 4  # big gap between groups (|g,n> block and |e,n> block)

#     def to_separated_strips(
#         mat_2d, strip_px, gap_px, major_breaks=None, major_gap_px=None
#     ):
#         """
#         mat_2d: shape (n_dim, n_time)
#         Inserts NaN rows to create visible whitespace gaps between strips,
#         and inserts bigger NaN gaps after indices in major_breaks.

#         major_breaks: list of indices k meaning "insert major gap BEFORE component k"
#                       Example: major_breaks=[N] inserts big gap between components N-1 and N
#         """
#         mat_2d = np.asarray(mat_2d)
#         n_dim, n_time = mat_2d.shape

#         if major_breaks is None:
#             major_breaks = []
#         if major_gap_px is None:
#             major_gap_px = gap_px

#         # Build output by walking down y and placing strips
#         rows = []
#         for i in range(n_dim):
#             # add strip (repeat 1 row into strip_px rows)
#             strip = np.repeat(mat_2d[i : i + 1, :], repeats=strip_px, axis=0)
#             rows.append(strip)

#             # add gap (except after last component)
#             if i != n_dim - 1:
#                 next_idx = i + 1
#                 this_gap = major_gap_px if next_idx in major_breaks else gap_px
#                 rows.append(np.full((this_gap, n_time), np.nan, dtype=float))

#         out = np.vstack(rows)
#         return out

#     # colormap that shows NaNs as white gaps
#     cmap_strips = plt.get_cmap("magma").copy()
#     cmap_strips.set_bad(color="white")

#     def infer_major_breaks(n_dim, dims):
#         """
#         Automatic "big group separation" guess:
#         If dims looks like atom=2 and field=N and n_dim == 2*N,
#         we assume ordering is:
#             [|g,0>, |g,1>, ..., |g,N-1>, |e,0>, |e,1>, ..., |e,N-1>]
#         so we insert a major gap before index N.
#         """
#         if isinstance(dims, dict) and ("atom" in dims) and ("field" in dims):
#             if dims["atom"] == 2 and n_dim == dims["atom"] * dims["field"]:
#                 return [dims["field"]]
#         return []

#     def imshow_strips(
#         ax, mat_dim_time, t0, tf, strip_px, gap_px, major_breaks, major_gap_px
#     ):
#         """
#         mat_dim_time: (n_dim, n_time)
#         """
#         data = to_separated_strips(
#             mat_dim_time,
#             strip_px=strip_px,
#             gap_px=gap_px,
#             major_breaks=major_breaks,
#             major_gap_px=major_gap_px,
#         )

#         extent_sep = [t0, tf, 0, data.shape[0]]

#         im = ax.imshow(
#             data,
#             cmap=cmap_strips,
#             extent=extent_sep,
#             aspect="auto",
#             interpolation="nearest",
#         )
#         return im, extent_sep

#     nn_state_real = models_dict["model_real"](time)
#     nn_state_imag = models_dict["model_imag"](time)

#     n_dim = nn_state_real.shape[1]

#     t0, tf = time[0].item(), time[-1].item()

#     major_breaks = infer_major_breaks(n_dim, dims)

#     # REAL PART
#     im, extent_sep = imshow_strips(
#         axs[0, 0],
#         nn_state_real.detach().numpy().T,
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )
#     cbar = fig.colorbar(im, ax=axs[0, 0], orientation="vertical", pad=0.15)
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Component magnitude", fontsize=12)

#     im, _ = imshow_strips(
#         axs[1, 0],
#         sim_state.real.T.detach().numpy(),
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )

#     cbar = fig.colorbar(im, ax=axs[1, 0], orientation="vertical", pad=0.1)
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Component magnitude", fontsize=12)

#     im, _ = imshow_strips(
#         axs[2, 0],
#         abs(sim_state.real - nn_state_real).T.detach().numpy(),
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )
#     cbar = fig.colorbar(im, ax=axs[2, 0], orientation="vertical", pad=0.1)
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Error Magnitude", fontsize=12)

#     # IMAGINARY PART
#     im, _ = imshow_strips(
#         axs[0, 1],
#         nn_state_imag.detach().numpy().T,
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )
#     cbar = fig.colorbar(im, ax=axs[0, 1], orientation="vertical", pad=0.1)
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Component magnitude", fontsize=12)

#     im, _ = imshow_strips(
#         axs[1, 1],
#         sim_state.imag.T.detach().numpy(),
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )
#     cbar = fig.colorbar(im, ax=axs[1, 1], orientation="vertical", pad=0.1)
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Component magnitude", fontsize=12)

#     im, _ = imshow_strips(
#         axs[2, 1],
#         abs(sim_state.imag - nn_state_imag).T.detach().numpy(),
#         t0,
#         tf,
#         STRIP_PX,
#         GAP_PX,
#         major_breaks,
#         MAJOR_GAP_PX,
#     )

#     cbar = fig.colorbar(
#         im, ax=axs[2, 1], orientation="vertical", pad=0.1
#     )
#     cbar.ax.yaxis.set_tick_params(labelsize=12)
#     cbar.ax.set_title("Error Magnitude", fontsize=12)


#     axs[0, 0].set_ylabel("Vector element index", fontsize=16)
#     axs[1, 0].set_ylabel("Vector element index", fontsize=16)
#     axs[2, 0].set_ylabel("Vector element index", fontsize=16)
#     axs[0, 1].set_ylabel("Vector element index", fontsize=16)
#     axs[1, 1].set_ylabel("Vector element index", fontsize=16)
#     axs[2, 1].set_ylabel("Vector element index", fontsize=16)

#     # Keep your x ticks logic (unchanged)
#     for ax in axs.flat:
#         ax.set_xticks(np.linspace(t0, tf, num=5))
#         ax.set_xticklabels(np.round(np.linspace(t0, tf, num=5), 2))

#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize


def plot_states(
    models_dict,
    params,
    tfinal,
    n_time_steps,
    init_state,
    picture,
    dims,
    is_scaled,
    plot_input="state",
):

    sim_state, time = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    if is_scaled:
        time = time / time.max()

    # >>> Make figure taller (key for “strip readability”)
    # If set_plot_params_states() already creates fig/axs, change its figsize there.
    fig, axs = set_plot_params_states()  # ideally: figsize=(16, 9) or (16, 10)

    # ============================================================
    # STRIPS: thicker + clearer gaps
    # ============================================================
    STRIP_PX = 10
    GAP_PX = 3
    MAJOR_GAP_PX = 8

    def infer_major_breaks(n_dim, dims):
        if isinstance(dims, dict) and ("atom" in dims) and ("field" in dims):
            if dims["atom"] == 2 and n_dim == dims["atom"] * dims["field"]:
                return [dims["field"]]  # split between |g,n> and |e,n>
        return []

    def build_basis_labels(n_dim, dims):
        if isinstance(dims, dict) and dims.get("atom", None) == 2 and "field" in dims:
            N = dims["field"]
            if n_dim == 2 * N:
                return [rf"$|g,{n}\rangle$" for n in range(N)] + [
                    rf"$|e,{n}\rangle$" for n in range(N)
                ]
        return [f"{i}" for i in range(n_dim)]

    def to_separated_strips_with_centers(
        mat_2d, strip_px, gap_px, major_breaks=None, major_gap_px=None
    ):
        """
        Returns:
          data (with NaN gaps),
          centers: y-coordinate (data units) at the center of each component strip,
          breaks_y: y-coordinates (data units) for group boundary markers (optional).
        """
        mat_2d = np.asarray(mat_2d)
        n_dim, n_time = mat_2d.shape

        if major_breaks is None:
            major_breaks = []
        if major_gap_px is None:
            major_gap_px = gap_px

        rows = []
        centers = []
        breaks_y = []

        y = 0
        for i in range(n_dim):
            strip = np.repeat(mat_2d[i : i + 1, :], repeats=strip_px, axis=0)
            rows.append(strip)

            centers.append(y + strip_px / 2.0)
            y += strip_px

            if i != n_dim - 1:
                next_idx = i + 1
                this_gap = major_gap_px if next_idx in major_breaks else gap_px
                if next_idx in major_breaks:
                    breaks_y.append(y + this_gap / 2.0)  # middle of the major gap
                rows.append(np.full((this_gap, n_time), np.nan, dtype=float))
                y += this_gap

        out = np.vstack(rows)
        return out, np.array(centers), np.array(breaks_y), out.shape[0]

    def imshow_strips(
        ax,
        mat_dim_time,
        t0,
        tf,
        strip_px,
        gap_px,
        major_breaks,
        major_gap_px,
        cmap,
        norm,
    ):
        data, centers, breaks_y, H = to_separated_strips_with_centers(
            mat_dim_time,
            strip_px=strip_px,
            gap_px=gap_px,
            major_breaks=major_breaks,
            major_gap_px=major_gap_px,
        )

        im = ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            extent=[t0, tf, 0, H],
            aspect="auto",
            interpolation="nearest",
            origin="lower",  # <-- makes y increase upward (more intuitive with ticks)
        )
        return im, centers, breaks_y, H

    # Colormaps: diverging for signed amplitudes, sequential for abs error
    cmap_amp = plt.get_cmap("magma").copy()
    cmap_err = plt.get_cmap("magma").copy()
    cmap_amp.set_bad(color="white")
    cmap_err.set_bad(color="white")

    nn_state_real = models_dict["model_real"](time)
    nn_state_imag = models_dict["model_imag"](time)

    n_dim = nn_state_real.shape[1]
    t0, tf = time[0].item(), time[-1].item()

    major_breaks = infer_major_breaks(n_dim, dims)
    basis_labels = build_basis_labels(n_dim, dims)

    # ---------- Shared color scaling (so “top 2 rows” really share one colorbar)
    nnR = nn_state_real.detach().cpu().numpy()
    nnI = nn_state_imag.detach().cpu().numpy()
    simR = sim_state.real.detach().cpu().numpy()
    simI = sim_state.imag.detach().cpu().numpy()

    amp_max = np.max(np.abs(np.concatenate([nnR, nnI, simR, simI], axis=None)))
    amp_norm = TwoSlopeNorm(vcenter=0.0, vmin=-amp_max, vmax=amp_max)

    errI = np.abs(simI - nnI)
    errR = np.abs(simR - nnR)
    err_max = np.max(np.concatenate([errR, errI], axis=None))
    err_norm = Normalize(vmin=0.0, vmax=err_max)

    # ---------- Plot all panels
    im00, centers, breaks_y, H = imshow_strips(
        axs[0, 0],
        nnR.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_amp,
        norm=amp_norm,
    )
    im01, _, _, _ = imshow_strips(
        axs[0, 1],
        nnI.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_amp,
        norm=amp_norm,
    )

    im10, _, _, _ = imshow_strips(
        axs[1, 0],
        simR.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_amp,
        norm=amp_norm,
    )
    im11, _, _, _ = imshow_strips(
        axs[1, 1],
        simI.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_amp,
        norm=amp_norm,
    )

    im20, _, _, _ = imshow_strips(
        axs[2, 0],
        errR.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_err,
        norm=err_norm,
    )
    im21, _, _, _ = imshow_strips(
        axs[2, 1],
        errI.T,
        t0,
        tf,
        STRIP_PX,
        GAP_PX,
        major_breaks,
        MAJOR_GAP_PX,
        cmap=cmap_err,
        norm=err_norm,
    )

    # ---------- Y ticks: centers of strips
    # If many components, label sparsely but keep minor ticks for every strip
    if n_dim <= 12:
        major_idx = list(range(n_dim))
    else:
        step = max(1, n_dim // 8)
        major_idx = list(range(0, n_dim, step))

    major_ticks = [centers[i] for i in major_idx]
    major_labels = [basis_labels[i] for i in major_idx]

    for r in range(3):
        axs[r, 0].set_yticks(major_ticks)
        axs[r, 0].set_yticklabels(major_labels, fontsize=11)
        axs[r, 0].set_yticks(centers, minor=True)
        axs[r, 0].tick_params(axis="y", which="minor", length=2)

        # Right column: keep tick marks but hide labels (less clutter)
        axs[r, 1].set_yticks(major_ticks)
        axs[r, 1].set_yticklabels([])
        axs[r, 1].set_yticks(centers, minor=True)
        axs[r, 1].tick_params(axis="y", which="minor", length=2)

        # Optional: mark the group boundary (if any)
        for yb in breaks_y:
            axs[r, 0].axhline(yb, color="k", lw=0.6, alpha=0.25)
            axs[r, 1].axhline(yb, color="k", lw=0.6, alpha=0.25)

    # ---------- Axis labels (only once, clean)
    for r in range(3):
        axs[r, 0].set_ylabel(r"Basis component $|s,n\rangle$", fontsize=14)

    # X ticks (keep yours)
    for ax in axs.flat:
        ax.set_xticks(np.linspace(t0, tf, num=5))
        ax.set_xticklabels(np.round(np.linspace(t0, tf, num=5), 2))

    # ---------- Shared colorbars (2 total)
    amp_sm = plt.cm.ScalarMappable(norm=amp_norm, cmap=cmap_amp)
    err_sm = plt.cm.ScalarMappable(norm=err_norm, cmap=cmap_err)

    cbar_amp = fig.colorbar(
        amp_sm,
        ax=[axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]],
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar_amp.set_label("Component value (signed)", fontsize=12)

    cbar_err = fig.colorbar(
        err_sm,
        ax=[axs[2, 0], axs[2, 1]],
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    cbar_err.set_label("Absolute error", fontsize=12)

    # If you use constrained_layout in set_plot_params_states(), prefer fig.tight_layout OFF.
    # plt.tight_layout()
    plt.show()


def plot_loss_functions(loss_dict: dict, skip_param: int):
    """
    Plots the loss functions during training.

    Args:
        loss_dict: Dictionary containing the loss functions.
    """

    labels = [
        r"\(L\)",
        r"\(L_{\mathrm{ic}}\)",
        r"\(L_{\mathrm{norm}}\)",
        r"\(L_{\mathrm{data}}\)",
        r"\(L_{\mathrm{eq}}\)",
    ]

    epochs = len(loss_dict["total_loss"])
    skip_epochs = int(epochs // skip_param)

    plt.figure(dpi=300)
    i = 0
    for key, value in loss_dict.items():
        if "loss" in key:
            plt.plot(value[0:-1:skip_epochs], label=labels[i])
            i += 1
    plt.yscale("log")
    plt.xlabel(r"Epochs (\(\times\)100)")
    plt.ylabel("Loss")
    plt.grid(alpha=0.4)
    plt.legend(
        loc="upper right",
        framealpha=0.9,
        facecolor="lightgray",
        edgecolor="gray",
        fontsize=6,
    )
    plt.tight_layout()
    plt.show()


def plot_learned_param(
    loss_dict: dict, skip_param: int, true_param: float, picture: str
):
    """
    Plots the learned parameters during training.

    Args:
        loss_dict: Dictionary containing the loss functions.
    """

    epochs = len(loss_dict["total_loss"])
    skip_epochs = int(epochs // skip_param)

    learned_params = np.array(loss_dict["learned_param"])  # (raw_epochs, n_params)

    # -----------------------------
    # Main (downsampled) plot data
    # -----------------------------
    y_ds = learned_params[0:-1:skip_epochs, :]
    x_ds = np.arange(len(y_ds))
    true_param_line_g1 = [true_param] * len(x_ds)
    true_param_line_g2 = [0] * len(x_ds)

    plt.figure(dpi=300)
    ax = plt.gca()

    if picture == "atom":
        labels = [r"Learned Parameter"]
    elif picture == "rabi":
        labels = [r"\(g_1\)", r"\(g_2\)", r"\(g_3\)", r"\(g_4\)"]
    elif picture == "rabi2":
        labels = [r"\(g_1\) (Jaynes-Cummings)", r"\(g_2\) (Counter-Rotating)"]
    elif picture == "geral":
        labels = [
            r"\(g_1\) (Jaynes-Cummings)",
            r"\(g_2\) (Rabi)",
            r"\(g_3\) (Two-Photon)",
            r"\(g_0\) (Classical Field)",
        ]

    for param in range(y_ds.shape[1]):
        ax.plot(x_ds, y_ds[:, param], label=labels[param])

    true_line_g1 = ax.plot(
        x_ds, true_param_line_g1, label=r"True Parameter", linestyle="--"
    )
    # true_line_g2 = ax.plot(x_ds, true_param_line_g2, label="True Counter-Rotating Parameter", linestyle="--")

    ax.set_xlabel(r"Epochs (\(\times\)100)")
    ax.set_ylabel("Parameter Value")
    ax.legend(
        fontsize=6,
        facecolor="lightgray",
        edgecolor="gray",
        framealpha=0.9,
        # bbox_to_anchor=(1.02, 1.0),
        bbox_to_anchor=(0.5, 0.25),  # vertical center
    )
    ax.grid(alpha=0.4)

    # ==================================================
    # INSET: last 100 *RAW* epochs (no downsampling)
    # ==================================================
    N_LAST = 1000
    raw_epochs = learned_params.shape[0]

    if raw_epochs >= 5:
        n_last = min(N_LAST, raw_epochs)

        x_raw = np.arange(raw_epochs - n_last, raw_epochs)
        # x_raw = np.round(x_raw / 100)  # convert to "epochs x100" scale
        # x_raw = x_raw.astype(int)
        # x_raw = np.arange(n_last)  # local inset scale
        y_raw = learned_params[-n_last:, :]  # RAW values

        axins = inset_axes(
            ax,
            width="40%",
            height="50%",
            # loc="upper right",
            bbox_to_anchor=(0.01, 0.05, 0.9, 0.9),
            bbox_transform=ax.transAxes,
            borderpad=1.1,
        )

        for param in range(y_raw.shape[1]):
            axins.plot(x_raw, y_raw[:, param])

        axins.axhline(true_param, linestyle="--", linewidth=1)
        axins.axhline(
            true_param, linestyle="--", linewidth=1, color=true_line_g1[0].get_color()
        )
        # axins.axhline(
        #     0, linestyle="--", linewidth=1, color=true_line_g2[0].get_color()
        # )

        # tight zoom around convergence
        ymin = min(y_raw.min(), true_param) - 5e-1 / 5
        ymax = max(y_raw.max(), true_param) + 5e-2
        axins.set_xlim(x_raw[0], x_raw[-1])
        # axins.set_ylim(ymin, ymax)
        axins.set_xlabel(r"Epochs", fontsize=6)
        axins.set_ylabel(r"Parameter Value", fontsize=6)
        axins.set_xticks(
            [x_raw[0], x_raw[0] + (raw_epochs - x_raw[0]) // 2, raw_epochs]
        )

        axins.grid(alpha=0.3)
        axins.tick_params(labelsize=6)

    plt.show()


def plot_fidelity(
    models_dict,
    params,
    tfinal,
    n_time_steps,
    init_state,
    picture,
    dims,
    is_scaled,
    plot_input="state",
):

    sim_state, time = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    if is_scaled:
        time = time / time.max()

    nn_state_real = models_dict["model_real"](time)
    nn_state_imag = models_dict["model_imag"](time)

    nn_state = nn_state_real + 1j * nn_state_imag

    nn_state_conj = nn_state.conj()

    inner_product = torch.sum(nn_state_conj * sim_state, dim=1)

    fidelity = torch.abs(inner_product) ** 2

    plt.figure(dpi=300)
    plt.plot(time.detach().numpy(), fidelity.detach().numpy(), label="Fidelity")
    plt.xlabel(r"\(gt\)")
    plt.ylabel(r"\(\mathcal{F}(|\tilde{\psi}(t)\rangle, |\psi(t)\rangle)\)")
    plt.grid(alpha=0.4)
    plt.show()
