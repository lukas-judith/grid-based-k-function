import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_relative_error_comparison(experiment_dict):
    """
    Plots the mean and std of relative errors of grid-based K-function
    and the (original) K-function computed on points extracted from the image.

    Args:
        experiment_dict (dict): Dictionary containing simulation results.
    """
    # unpack simulation results from dictionary
    d = experiment_dict["d"]
    n = experiment_dict["n"]
    n_iter = experiment_dict["n_iter"]
    range_of_t = experiment_dict["range_of_t"]
    err_grid_mean = experiment_dict["err_grid_mean"]
    err_grid_std = experiment_dict["err_grid_std"]
    err_extr_mean = experiment_dict["err_extr_mean"]
    err_extr_std = experiment_dict["err_extr_std"]

    fig, ax = plt.subplots(1, figsize=(8, 3.5))
    ax.plot(range_of_t, err_grid_mean, label="Error, grid-based", color="blue")
    ax.fill_between(
        range_of_t,
        err_grid_mean - err_grid_std,
        err_grid_mean + err_grid_std,
        color="blue",
        alpha=0.2,
    )
    ax.plot(range_of_t, err_extr_mean, label="Error, extracted", color="orange")
    ax.fill_between(
        range_of_t,
        err_extr_mean - err_extr_std,
        err_extr_mean + err_extr_std,
        color="orange",
        alpha=0.2,
    )
    x_min, x_max = ax.get_xlim()
    ax.hlines(0.01, x_min, x_max, color="red", linestyle="--", label="1% error margin")
    ax.set_yscale("log")
    ax.legend(loc="upper right", prop={"size": 12})
    ax.set_xlabel("K-function parameter t", size=12)
    ax.set_ylabel("Relative Error", size=12)
    ax.set_title(
        f"Error Comparison ({n_iter} iterations, {d}x{d} Images, {n} Points each)",
        size=14,
    )
    plt.tight_layout()
    plt.show()


def plot_K_diff_comparison(experiment_dict):
    """
    Plots a simulated image containing n points as well as the different
    K_diff-functions computed from the image.

    Args:
        experiment_dict (dict): Dictionary containing simulation results.
    """
    # unpack simulation results from dictionary
    d = experiment_dict["d"]
    n = experiment_dict["n"]
    grid = experiment_dict["grid"]
    extr_y = experiment_dict["extr_y"]
    extr_x = experiment_dict["extr_x"]
    range_of_t = experiment_dict["range_of_t"]
    K_vals_csr = experiment_dict["K_vals_csr"]
    K_vals_orig = experiment_dict["K_vals_orig"]
    K_vals_extr = experiment_dict["K_vals_extr"]
    K_vals_ours = experiment_dict["K_vals_ours"]

    img_name = f"{d}x{d} Image, {n} Points"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax[0].imshow(grid)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")
    ax[0].scatter(extr_y, extr_x, s=3, c="red", marker="x", label="Extracted Points")
    ax[0].legend(loc="upper right")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(img_name, size=14)
    ax[1].plot(range_of_t, K_vals_orig, color="red", label="Original")
    ax[1].plot(
        range_of_t, K_vals_ours, color="blue", label="Grid-based", linestyle="--"
    )
    ax[1].plot(
        range_of_t, K_vals_extr, color="orange", label="Extracted", linestyle="--"
    )
    ax[1].legend(loc="upper right")
    ax[1].set_title("$K_{diff}$-functions", size=14)
    ax[1].set_xlabel("K-function parameter t", size=12)
    ax[1].grid()
    plt.show()
