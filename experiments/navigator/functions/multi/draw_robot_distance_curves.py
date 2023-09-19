import os
import tqdm
import pickle
import numpy as np
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from typing import List
from experiments.navigator.trial import TrialRecord
from experiments.navigator.utils import get_cache_directory
from experiments.navigator.functions.multi.draw_reward_curves import smooth
from renesis.utils.robot import get_robot_voxels_from_voxels


def generate_robot_distance_metrics_for_trial(record: TrialRecord):
    metrics = {}
    cache_path = os.path.join(
        get_cache_directory("robot_distance_cache"),
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
                pool.imap(
                    generate_robot_distance_metrics_for_epoch, epoch_data_file_paths
                ),
                total=len(epoch_data_file_paths),
            )
        )
        for epoch, result in zip(record.epochs, results):
            metrics[epoch] = result
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def generate_robot_distance_metrics_for_epoch(epoch_data_file_path):
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        robots_voxels = np.array(
            [
                get_robot_voxels_from_voxels(d["voxels"])[0].reshape(-1)
                for d in data
                if len(d["steps"]) > 0
            ]
        )
        pairwise_l0_distance = np.linalg.norm(
            robots_voxels[:, None, :] - robots_voxels[None, :, :], ord=0, axis=-1
        )
        return (
            pairwise_l0_distance,
            np.mean(pairwise_l0_distance),
            np.std(pairwise_l0_distance),
            len(data),
        )


def draw_separate_robot_distance_curves(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    robot_distance_curves = np.zeros([len(records), len(truncated_epochs), 3])
    print(f"show epoch num: {truncated_epochs[-1]}")
    legend_labels = []
    if input("Add legend labels? [y/n]").lower() == "y":
        for i in range(len(records)):
            legend_labels.append(input(f"Label for record {i}:"))
    current_curve_max = -np.inf
    for record_idx, record in enumerate(records):
        metrics = generate_robot_distance_metrics_for_trial(record)
        for epoch in truncated_epochs:
            # mean
            robot_distance_curves[record_idx, epoch - 1] = metrics[epoch][1:]
        std = robot_distance_curves[record_idx, :, 1]
        mean = robot_distance_curves[record_idx, :, 0]
        shift = std * 2.576 / np.sqrt(robot_distance_curves[record_idx, :, 2])
        plt.fill_between(
            truncated_epochs,
            mean - shift,
            mean + shift,
            color=f"skyblue",
        )
        plt.plot(
            truncated_epochs,
            smooth(mean),
            color=f"steelblue",
            label=legend_labels[record_idx]
            if legend_labels
            else f"record {record_idx}",
        )
        current_curve_max = max(np.max(mean + shift), current_curve_max)

    plt.ylim(0, np.max(robot_distance_curves) * 1.1)
    plt.ylabel("Average L0 distance")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Average pairwise distances")
    plt.show()
