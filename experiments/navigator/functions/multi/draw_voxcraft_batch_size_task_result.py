import os
import tqdm
import pickle
import numpy as np
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from renesis.utils.robot import get_robot_voxels_from_voxels
from experiments.navigator.trial import TrialRecord
from experiments.navigator.utils import get_cache_directory
from experiments.navigator.functions.multi.draw_reward_curves import (
    generate_rewards_for_trial,
    smooth,
)


def generate_top_reward_metrics_for_trial(record: TrialRecord):
    metrics = {}
    cache_path = os.path.join(
        get_cache_directory("top_reward_cache"),
        record.trial_dir.replace("/", "#") + ".cache",
    )
    if os.path.exists(cache_path):
        with open(os.path.join(cache_path), "rb") as cache_file:
            return pickle.load(cache_file)
    else:
        pool = Pool()
        epoch_data_file_paths = [
            os.path.join(record.data_dir, record.epoch_files[epoch].data_file_name)
            for epoch in record.epochs
        ]
        results = list(
            tqdm.tqdm(
                pool.imap(compute_top_reward_metrics_for_epoch, epoch_data_file_paths),
                total=len(epoch_data_file_paths),
            )
        )
        for epoch, result in zip(record.epochs, results):
            metrics[epoch] = result
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def compute_top_reward_metrics_for_epoch(epoch_data_file_path):
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        rewards = sorted([d["reward"] for d in data], reverse=True)
        rewards = rewards[: int(len(rewards) / 10)]
        return (
            np.max(rewards),
            np.min(rewards),
            np.mean(rewards),
            np.std(rewards),
            len(rewards),
        )


def draw_voxcraft_batch_size_top_reward(
    records: List[TrialRecord], labels: List[str], truncated_epochs: List[int], ax
):
    for record, label, color in zip(
        records, labels, [["skyblue", "steelblue", 1], ["lightgrey", "grey", 0.5]]
    ):
        metrics = generate_top_reward_metrics_for_trial(record)
        mean = np.zeros([len(truncated_epochs)])
        shift = np.zeros([len(truncated_epochs)])
        for epoch in truncated_epochs:
            # mean
            mean[epoch - 1] = metrics[epoch][2]
            shift[epoch - 1] = metrics[epoch][3] * 2.576 / np.sqrt(metrics[epoch][4])
        ax.fill_between(
            truncated_epochs,
            mean - shift,
            mean + shift,
            color=color[0],
            alpha=color[2],
        )
        ax.plot(truncated_epochs, smooth(mean), color=color[1], label=label)
    ax.legend()
    ax.set_ylabel("Displacement\n(voxel length) of top 10% robots", fontsize=14)
    ax.set_ylim(0, np.max(mean + shift) * 1.1)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
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


def draw_voxcraft_batch_size_mean_reward(
    records: List[TrialRecord], labels: List[str], truncated_epochs: List[int], ax
):
    for record, label, color in zip(
        records, labels, [["skyblue", "steelblue", 1], ["lightgrey", "grey", 0.5]]
    ):
        metrics = generate_rewards_for_trial(record)
        mean = np.zeros([len(truncated_epochs)])
        shift = np.zeros([len(truncated_epochs)])
        for epoch in truncated_epochs:
            # mean
            mean[epoch - 1] = np.mean(metrics[epoch])
            shift[epoch - 1] = (
                np.std(metrics[epoch]) * 2.576 / np.sqrt(len(metrics[epoch][4]))
            )
        ax.fill_between(
            truncated_epochs, mean - shift, mean + shift, color=color[0], alpha=color[2]
        )
        ax.plot(truncated_epochs, smooth(mean), color=color[1], label=label)
    ax.legend()
    ax.set_ylabel("Displacement\n(voxel length)", fontsize=14)
    ax.set_ylim(0, np.max(mean + shift) * 1.1)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.text(
        0.05,
        0.95,
        "B",
        transform=ax.transAxes,
        color="black",
        fontsize=15,
        verticalalignment="top",
    )


def draw_voxcraft_batch_size_task_result(
    records: List[TrialRecord],
):
    assert len(records) == 3
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    plt.subplots_adjust(hspace=0.0, bottom=0, top=1)
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    draw_voxcraft_batch_size_top_reward(
        records[:2],
        ["BS=20480, train with top 10%", "BS=2048, train with all"],
        truncated_epochs,
        axs[0],
    )
    draw_voxcraft_batch_size_mean_reward(
        records[1:],
        ["BS=2048, train with all", "BS=128, train with all"],
        truncated_epochs,
        axs[1],
    )
    fig.savefig(
        f"data/generated_data/voxcraft_bs.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.show()
