import os
import tqdm
import pickle
import numpy as np
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from typing import List
from experiments.navigator.trial import TrialRecord
from experiments.navigator.utils import get_cache_directory


def generate_reward_metrics_for_trial(record: TrialRecord):
    metrics = {}
    cache_path = os.path.join(
        get_cache_directory("reward_cache"),
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
                pool.imap(compute_reward_metrics_for_epoch, epoch_data_file_paths),
                total=len(epoch_data_file_paths),
            )
        )
        for epoch, result in zip(record.epochs, results):
            metrics[epoch] = result
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def compute_reward_metrics_for_epoch(epoch_data_file_path):
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        rewards = np.array([d["reward"] for d in data])
        return (
            np.max(rewards),
            np.min(rewards),
            np.mean(rewards),
            np.std(rewards),
            len(rewards),
        )


def smooth(scalars: np.array, window_size: int = 5) -> np.array:
    smoothed = list()
    for idx in range(len(scalars)):
        min_idx = max(0, idx - window_size // 2)
        max_idx = min(len(scalars), idx + window_size // 2) + 1
        samples = scalars[min_idx:max_idx]
        smoothed.append(np.mean(samples))
    return np.array(smoothed)


def draw_reward_curve(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    reward_curves = np.zeros([len(records), len(truncated_epochs)])
    print(f"show epoch num: {truncated_epochs[-1]}")
    for record_idx, record in enumerate(records):
        metrics = generate_reward_metrics_for_trial(record)
        for epoch in truncated_epochs:
            # mean
            reward_curves[record_idx, epoch - 1] = metrics[epoch][2]
    std = np.std(reward_curves, axis=0)
    mean = np.mean(reward_curves, axis=0)
    shift = std * 2.576 / np.sqrt(len(records))
    plt.fill_between(
        truncated_epochs,
        mean - shift,
        mean + shift,
        color="skyblue",
    )
    plt.plot(
        truncated_epochs,
        smooth(mean),
        color="steelblue",
    )

    plt.ylim(0, np.max(mean + shift) * 1.1)
    plt.ylabel("Travel distance in voxels")
    # plt.ylabel("Volume")
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()


def draw_separate_reward_curves(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    reward_curves = np.zeros([len(records), len(truncated_epochs)])
    print(f"show epoch num: {truncated_epochs[-1]}")
    labels = [
        "log std bias 1->0",
        "log std bias 1->0.25",
        "log std bias 1->0.5",
        "no bias",
    ]
    for record_idx, record in enumerate(records):
        metrics = generate_reward_metrics_for_trial(record)
        for epoch in truncated_epochs:
            # mean
            reward_curves[record_idx, epoch - 1] = metrics[epoch][2]
        plt.plot(
            truncated_epochs,
            smooth(reward_curves[record_idx, :]),
            label=labels[record_idx],  # label=f"Record {record_idx}"
        )

    plt.ylim(0, np.max(reward_curves) * 1.1)
    plt.ylabel("Travel distance in voxels")
    # plt.ylabel("Volume")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()


def draw_separate_reward_curves_with_batch_std(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    reward_curves = np.zeros([len(records), len(truncated_epochs), 3])
    print(f"show epoch num: {truncated_epochs[-1]}")
    labels = [
        "log std bias 1->0",
        "log std bias 1->0.25",
        "log std bias 1->0.5",
        "no bias",
    ]
    current_curve_max = -np.inf
    for record_idx, record in enumerate(records):
        metrics = generate_reward_metrics_for_trial(record)
        for epoch in truncated_epochs:
            # mean
            reward_curves[record_idx, epoch - 1] = metrics[epoch][2:]
        std = reward_curves[record_idx, :, 1]
        mean = reward_curves[record_idx, :, 0]
        shift = std * 2.576 / np.sqrt(reward_curves[record_idx, :, 2])
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
            label=labels[record_idx],
        )
        current_curve_max = max(np.max(mean + shift), current_curve_max)

    plt.ylim(0, current_curve_max * 1.1)
    plt.ylabel("Travel distance in voxels")
    # plt.ylabel("Volume")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()
