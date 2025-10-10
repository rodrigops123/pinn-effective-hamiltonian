import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "text.latex.preamble": r"\usepackage{physics}",
    }
)

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

    if plot_input == "expected":
        sim_expect_train, sim_expect_test = train_test_split(sim_expect, test_size=0.2)
        time_train, time_test = train_test_split(time, test_size=0.2)

        return (
            sim_expect_train,
            sim_expect_test,
            time_train.reshape(-1, 1),
            time_test.reshape(-1, 1),
            operators_list,
        )

    else:
        sim_state_train, sim_state_test = train_test_split(sim_state, test_size=0.2)
        time_train, time_test = train_test_split(time, test_size=0.2)

        return (
            sim_state_train,
            sim_state_test,
            time_train.reshape(-1, 1),
            time_test.reshape(-1, 1),
        )


def set_plot_params_expected_values():

    fig = plt.figure(figsize=(10, 8))
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
            r"\(\ev{a^\dagger a}_{NN}\)",
            r"\(\ev{a^\dagger a}_{sim}\)",
        ],
        [
            r"\(\ev{ \sigma_+ \sigma_-}_{NN}\)",
            r"\(\ev{ \sigma_+ \sigma_-}_{sim}\)",
        ],
    ]

    labels_error = [
        r"error\(\left(a^\dagger a\right)\)",
        r"error\(\left(\sigma_+ \sigma_-\right)\)",
    ]

    colors = ["blue", "orange"]
    colors_error = ["green", "red"]

    return labels, labels_error, colors, colors_error


def plot_expected_values(
    models_dict,
    tfinal,
    n_time_steps,
    init_state,
    params,
    picture,
    dims,
    train_or_test,
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
    (
        sim_expect_train,
        sim_expect_test,
        time_train,
        time_test,
        operators_list,
    ) = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    ax0, ax1 = set_plot_params_expected_values()
    labels, labels_error, colors, colors_error = set_labels_and_colors_expected_values()

    if train_or_test == "train":
        for i, operator in enumerate(operators_list):
            nn_state_train = models_dict["model_real"](time_train) + 1j * models_dict[
                "model_imag"
            ](time_train)

            expected_values_train = torch.einsum(
                "ni,ij,nj->n", nn_state_train.conj(), operator, nn_state_train
            ).real

            error = (
                expected_values_train - sim_expect_train[:, i]
            ).detach().numpy() / sim_expect_train[:, i].detach().numpy()

            ax0.plot(
                time_train.numpy().squeeze(),
                expected_values_train.detach().numpy(),
                label=labels[i][0],
                color=colors[i],
            )

            ax0.plot(
                time_train.numpy().squeeze(),
                sim_expect_train[:, i].detach().numpy(),
                label=labels[i][1],
                color=colors[i],
                linestyle="--",
            )
            ax0.legend()

            ax1.plot(
                time_train.numpy().squeeze(),
                error,
                label=labels_error[i],
                color=colors_error[i],
            )
            ax1.legend()

            plt.tight_layout()

        plt.show()

    else:
        for i, operator in enumerate(operators_list):
            nn_state_test = models_dict["model_real"](time_test) + 1j * models_dict[
                "model_imag"
            ](time_test)

            expected_values_test = torch.einsum(
                "ni,ij,nj->n", nn_state_test.conj(), operator, nn_state_test
            ).real

            error = (
                expected_values_test - sim_expect_test[:, i]
            ).detach().numpy() / sim_expect_test[:, i].detach().numpy()

            ax0.plot(
                time_test.numpy().squeeze(),
                expected_values_test.detach().numpy(),
                label=labels[i][0],
                color=colors[i],
            )

            ax0.plot(
                time_test.numpy().squeeze(),
                sim_expect_test[:, i].detach().numpy(),
                label=labels[i][1],
                color=colors[i],
                linestyle="--",
            )
            ax0.legend()

            ax1.plot(
                time_test.numpy().squeeze(),
                error,
                label=labels_error[i],
                color=colors_error[i],
            )
            ax1.legend()

            plt.tight_layout()

        plt.show()


def set_plot_params_states():
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 4), sharex=True, dpi=300)
    axs[0, 0].set_title(r"$\ket{\tilde{\psi}_R (t)}$")
    axs[0, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[0, 0].yaxis.set_ticklabels([])

    axs[1, 0].set_title(r"$\ket{\psi_R(t)}$")
    axs[1, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[1, 0].yaxis.set_ticklabels([])

    axs[2, 0].set_title(r"$\abs{\ket{\psi_R(t)} - \ket{\tilde{\psi}_R(t)} }$")
    axs[2, 0].yaxis.set_ticks([])  # Remove y-ticks
    axs[2, 0].yaxis.set_ticklabels([])

    axs[0, 1].set_title(r"$\ket{\tilde{\psi}_I(t)}$")
    axs[0, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[0, 1].yaxis.set_ticklabels([])

    axs[1, 1].set_title(r"$\ket{\psi_I(t)}$")
    axs[1, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[1, 1].yaxis.set_ticklabels([])

    axs[2, 1].set_title(r"$\abs*{ \ket{\psi_I(t)} - \ket{\tilde{\psi}_I(t)} }$")
    axs[2, 1].yaxis.set_ticks([])  # Remove y-ticks
    axs[2, 1].yaxis.set_ticklabels([])

    axs[2, 0].set_xlabel(r"\(gt\)")
    axs[2, 1].set_xlabel(r"\(gt\)")

    return fig, axs


def plot_states(
    models_dict,
    params,
    tfinal,
    n_time_steps,
    init_state,
    picture,
    dims,
    train_or_test,
    plot_input="state",
):

    sim_state_train, sim_state_test, time_train, time_test = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    fig, axs = set_plot_params_states()

    if train_or_test == "train":
        nn_state_train_real = models_dict["model_real"](time_train)
        nn_state_train_imag = models_dict["model_imag"](time_train)

        extent = [
            time_train[0].item(),
            time_train[-1].item(),
            0,
            nn_state_train_real.shape[1],
        ]

        # REAL PART
        im = axs[0, 0].imshow(
            nn_state_train_real.detach().numpy().T,
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[0, 0], orientation="vertical")

        im = axs[1, 0].imshow(
            sim_state_train.real.T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[1, 0], orientation="vertical")

        im = axs[2, 0].imshow(
            abs(sim_state_train.real - nn_state_train_real).T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[2, 0], orientation="vertical")

        # IMAGINARY PART
        im = axs[0, 1].imshow(
            nn_state_train_imag.detach().numpy().T,
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[0, 1], orientation="vertical")

        im = axs[1, 1].imshow(
            sim_state_train.imag.T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[1, 1], orientation="vertical")

        im = axs[2, 1].imshow(
            abs(sim_state_train.imag - nn_state_train_imag).T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )

        fig.colorbar(im, ax=axs[2, 1], orientation="vertical")

        for ax in axs.flat:
            ax.set_xticks(
                np.linspace(time_train[0].item(), time_train[-1].item(), num=5)
            )
            ax.set_xticklabels(
                np.round(
                    np.linspace(time_train[0].item(), time_train[-1].item(), num=5), 2
                )
            )

        plt.tight_layout()
        plt.show()

    else:
        nn_state_test_real = models_dict["model_real"](time_test)
        nn_state_test_imag = models_dict["model_imag"](time_test)

        extent = [
            time_test[0].item(),
            time_test[-1].item(),
            0,
            nn_state_test_real.shape[1],
        ]

        # REAL PART
        im = axs[0, 0].imshow(
            nn_state_test_real.detach().numpy().T,
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[0, 0], orientation="vertical")

        im = axs[1, 0].imshow(
            sim_state_test.real.T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[1, 0], orientation="vertical")

        im = axs[2, 0].imshow(
            abs(sim_state_test.real - nn_state_test_real).T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[2, 0], orientation="vertical")

        # IMAGINARY PART
        im = axs[0, 1].imshow(
            nn_state_test_imag.detach().numpy().T,
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[0, 1], orientation="vertical")

        im = axs[1, 1].imshow(
            sim_state_test.imag.T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[1, 1], orientation="vertical")

        im = axs[2, 1].imshow(
            abs(sim_state_test.imag - nn_state_test_imag).T.detach().numpy(),
            cmap="magma",
            extent=extent,
            aspect="auto",
        )
        fig.colorbar(im, ax=axs[2, 1], orientation="vertical")

        for ax in axs.flat:
            ax.set_xticks(np.linspace(time_test[0].item(), time_test[-1].item(), num=5))
            ax.set_xticklabels(
                np.round(
                    np.linspace(time_test[0].item(), time_test[-1].item(), num=5), 2
                )
            )

        plt.tight_layout()
        plt.show()


def plot_loss_functions(loss_dict: dict, skip_param: int):
    """
    Plots the loss functions during training.

    Args:
        loss_dict: Dictionary containing the loss functions.
    """

    epochs = len(loss_dict["total_loss"])
    skip_epochs = int(epochs // skip_param)

    plt.figure(figsize=(10, 6))
    for key, value in loss_dict.items():
        if "loss" in key:
            plt.plot(value[0:-1:skip_epochs], label=key)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Functions During Training")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.6)
    plt.show()


def plot_learned_param(loss_dict: dict, skip_param: int):
    """
    Plots the learned parameters during training.

    Args:
        loss_dict: Dictionary containing the loss functions.
    """

    epochs = len(loss_dict["total_loss"])
    skip_epochs = int(epochs // skip_param)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_dict["learned_param"][0:-1:skip_epochs], label="Learned Parameter")
    plt.xlabel("Epochs")
    plt.ylabel("Parameter Value")
    plt.title("Learned Parameter During Training")
    plt.legend()
    plt.grid()
    plt.show()


def plot_fidelity(
    models_dict,
    params,
    tfinal,
    n_time_steps,
    init_state,
    picture,
    dims,
    train_or_test,
    plot_input="state",
):

    sim_state_train, sim_state_test, time_train, time_test = prep_plot_input(
        params, tfinal, n_time_steps, init_state, picture, dims, plot_input
    )

    if train_or_test == "train":
        nn_state_train_real = models_dict["model_real"](time_train)
        nn_state_train_imag = models_dict["model_imag"](time_train)
        
        nn_state_train = nn_state_train_real + 1j * nn_state_train_imag

        nn_state_train_conj = nn_state_train.conj()
        
        inner_product = torch.sum(nn_state_train_conj * sim_state_train, dim=1)
        
        fidelity = torch.abs(inner_product) ** 2

        plt.figure(figsize=(10, 6))
        plt.plot(time_train.detach().numpy(), fidelity.detach().numpy(), label="Fidelity")
        plt.xlabel(r"\(gt\)")
        plt.ylabel(r"\(\mathcal{F}(\tilde{\psi}(t), \psi(t))\)")
        plt.title("Fidelity During Training")
        plt.grid(linestyle="--", alpha=0.6)
        plt.show()