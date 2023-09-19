import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from renesis.utils.robot import get_robot_voxels_from_voxels
from experiments.navigator.trial import TrialRecord
from experiments.navigator.functions.multi.draw_reward_curves import (
    generate_rewards_for_trial,
    smooth,
)
from experiments.navigator.functions.multi.draw_robot_metrics_curves import (
    generate_robot_metrics_for_trial,
    _show_robot_metrics_keys,
)


def draw_voxcraft_reward(records: List[TrialRecord], truncated_epochs: List[int], ax):
    records_rewards = []

    for record in records:
        records_rewards.append(generate_rewards_for_trial(record))

    mean = np.zeros(len(truncated_epochs))
    shift = np.zeros(len(truncated_epochs))
    # Combine same epoch results from multiple trials, and compute mean & std
    for epoch in truncated_epochs:
        epoch_rewards = []
        for record_rewards in records_rewards:
            epoch_rewards += record_rewards[epoch]
        mean[epoch - 1] = np.mean(epoch_rewards)
        shift[epoch - 1] = np.std(epoch_rewards) * 2.576 / np.sqrt(len(epoch_rewards))

    ax.fill_between(
        truncated_epochs,
        mean - shift,
        mean + shift,
        color="lightgrey",
    )
    ax.plot(
        truncated_epochs,
        smooth(mean),
        color="grey",
    )
    ax.set_ylabel("Displacement\n(voxel length)", fontsize=14)
    ax.set_ylim(0, np.max(mean + shift) * 1.1)
    ax.set_xticks([])
    # ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.text(
        0.05,
        0.95,
        "A",
        transform=ax.transAxes,
        color="black",
        fontsize=15,
        verticalalignment="top",
    )


def draw_voxcraft_robot_metrics(
    records: List[TrialRecord], truncated_epochs: List[int], axs
):
    for row, col, show_key, ref_label, y_label in (
        (0, 1, "volume", "B", "Volume\n(number of voxels)"),
        (
            0,
            2,
            "surface_voxels_to_total_volume_ratio",
            "C",
            "Surface voxels\nto volume",
        ),
        (0, 3, "passive_material_ratio", "D", "Passive material\nratio"),
        (
            1,
            0,
            "largest_connected_component_ratio",
            "E",
            "Largest connected\ncomponent ratio",
        ),
        (1, 1, "section_num", "F", "Substructures"),
        (1, 2, "reflection_symmetry", "G", "Reflection\nsymmetry"),
        (1, 3, "gzip_compressed_ratio", "H", "Gzip compressed\nratio"),
    ):
        ax = axs[row][col]
        metric_curves = np.zeros([len(records), len(truncated_epochs)])
        for record_idx, record in enumerate(records):
            metrics = generate_robot_metrics_for_trial(record)
            metric_curves[record_idx] = metrics[show_key][: metric_curves.shape[1]]
            # plt.plot(truncated_epochs, reward_curves[record_idx, :])
        std = np.std(metric_curves, axis=0)
        mean = np.mean(metric_curves, axis=0)
        shift = std * 2.576 / np.sqrt(len(records))
        ax.fill_between(
            truncated_epochs,
            mean - shift,
            mean + shift,
            color="lightgrey",
        )
        ax.plot(
            truncated_epochs,
            smooth(mean),
            color="grey",
        )
        ax.text(
            0.05,
            0.95,
            ref_label,
            transform=ax.transAxes,
            color="black",
            fontsize=15,
            verticalalignment="top",
        )
        if show_key == "surface_voxels_to_total_volume_ratio":
            ax.set_ylim(0.8, 1)
        else:
            ax.set_ylim(0, np.max(mean) * 1.1)
        ax.set_ylabel(y_label, fontsize=14)
        if row == 0:
            ax.set_xticks([])
        if row != 0:
            ax.set_xlabel("Epoch", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)


def draw_voxcraft_task_result(
    records: List[TrialRecord],
):
    fig, axs = plt.subplots(2, 4, figsize=(12, 4), layout="constrained")
    plt.subplots_adjust(hspace=0.0, bottom=0, top=1)
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    print(f"show epoch num: {truncated_epochs[-1]}")
    draw_voxcraft_reward(records, truncated_epochs, axs[0][0])
    draw_voxcraft_robot_metrics(records, truncated_epochs, axs)
    fig.align_ylabels()

    fig.savefig(
        f"generated_data/voxcraft.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.show()
