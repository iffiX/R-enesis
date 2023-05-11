import os
import pickle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from torch._C._return_types import sort

from experiments.patch.config import dimension_size
from renesis.env_model.patch import normalize

np.set_printoptions(precision=4, threshold=10000, suppress=True)


def plot_error_graph(axes: Axes, title: str, tick: np.ndarray, data: np.ndarray):
    min_value = np.min(data, axis=1)
    max_value = np.max(data, axis=1)
    axes.plot(tick, min_value, color="skyblue")
    axes.plot(tick, max_value, color="orange")
    axes.fill_between(tick, min_value, max_value, color="royalblue")
    axes.set_title(title)


def plot_episode_change_before_after_training(
    fig: Figure,
    axes: Axes,
    fig_title: str,
    y_label: str,
    all_data: np.ndarray,
    dimension_size: int,
    normalize_mode: str = "clip",
):
    fig.suptitle(fig_title)
    # all_data shape: [epochs, batch_size, steps]
    sorted_start_and_end_data = [
        normalize(np.sort(all_data[0], axis=-1), mode=normalize_mode) * dimension_size,
        normalize(np.sort(all_data[-1], axis=-1), mode=normalize_mode) * dimension_size,
    ]
    y_range = [
        np.min(sorted_start_and_end_data),
        np.max(sorted_start_and_end_data),
    ]
    for idx, sorted_episode_data in enumerate(sorted_start_and_end_data):
        sorted_episode_data_means_across_batch = np.mean(sorted_episode_data, axis=0)

        ci_results = [
            sc.stats.bootstrap(
                (sorted_episode_data[:, action_idx],), np.mean, confidence_level=0.99
            )
            for action_idx in range(sorted_episode_data.shape[1])
        ]
        ci_min, ci_max = zip(
            *[
                (r.confidence_interval.low, r.confidence_interval.high)
                for r in ci_results
            ]
        )
        y_errors = np.abs(
            np.array((ci_min, ci_max)) - sorted_episode_data_means_across_batch
        )  # shape [2, 10]

        axes[0][idx].set_ylim(0, dimension_size)
        axes[0][idx].axhline(y=0, color="grey", linestyle="dashed")
        axes[0][idx].bar(
            range(len(sorted_episode_data_means_across_batch)),
            sorted_episode_data_means_across_batch,
            yerr=y_errors,
        )

        axes[1][idx].set_ylim(0, dimension_size)
        axes[1][idx].axhline(y=0, color="grey", linestyle="dashed")
        axes[1][idx].scatter(
            np.array(
                [list(range(sorted_episode_data.shape[1]))]
                * sorted_episode_data.shape[0]
            ).flatten(),
            sorted_episode_data.flatten(),
            alpha=0.4,
        )
        axes[1][idx].set_xlabel(
            f"{'Before training' if idx == 0 else 'After training'}, "
            f"sorted low to high"
        )
        if idx == 0:
            axes[0][idx].set_ylabel(y_label)
            axes[1][idx].set_ylabel(y_label)


if __name__ == "__main__":
    path = "/home/mlw0504/ray_results/CustomPPO_2023-05-10_15-48-01/CustomPPO_VoxcraftSingleRewardTestPatchEnvironment_f41e7_00000_0_2023-05-10_15-48-01/data"
    # path = input("Enter path to ray experiment data directory: ")
    data_files = {}
    for file in os.listdir(path):
        if file.startswith("data"):
            data_files[int(file.split("_")[2])] = file

    all_iter = np.array(sorted(list(data_files.keys())))
    all_data = []
    for iter_num in all_iter:
        with open(os.path.join(path, data_files[iter_num]), "rb") as file:
            all_data.append(pickle.load(file))

    # shape: [epoch, batch_size, steps, action_dists]
    action_dists = np.array(
        [[data[idx]["step_dists"] for idx in range(len(data))] for data in all_data]
    )
    actions = np.array(
        [[data[idx]["steps"] for idx in range(len(data))] for data in all_data]
    )

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    plot_episode_change_before_after_training(
        fig,
        axs,
        "Z means of action distributions change",
        "Z means in voxels",
        action_dists[:, :, :, 2],
        dimension_size=dimension_size,
    )
    plt.savefig("z_means_of_action_distributions.pdf", bbox_inches="tight")
    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    plot_episode_change_before_after_training(
        fig,
        axs,
        "Z stds of action distributions change",
        "Z stds in voxels",
        np.exp(action_dists[:, :, :, 2 + action_dists.shape[-1] // 2]) / 4,
        dimension_size=dimension_size,
        normalize_mode="none",
    )
    plt.savefig("z_stds_of_action_distributions.pdf", bbox_inches="tight")
    plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    plot_episode_change_before_after_training(
        fig,
        axs,
        "Z means of actions change",
        "Z means in voxels",
        actions[:, :, :, 2],
        dimension_size=dimension_size,
    )
    plt.savefig("z_means_of_actions.pdf", bbox_inches="tight")
    plt.show()
