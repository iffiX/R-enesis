import os
import tqdm
import pickle
import numpy as np
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from typing import List
from experiments.navigator.trial import TrialRecord
from experiments.navigator.utils import get_cache_directory


def generate_rewards_for_trial(record: TrialRecord):
    # shape: {epoch_num: [batch_size]}
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
                pool.imap(generate_rewards_for_epoch, epoch_data_file_paths),
                total=len(epoch_data_file_paths),
            )
        )
        for epoch, result in zip(record.epochs, results):
            metrics[epoch] = result
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def generate_rewards_for_epoch(epoch_data_file_path):
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        # shape: [batch_size]
        rewards = [d["reward"] for d in data if len(d["steps"]) > 0]
        return rewards


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
    records_rewards = []
    print(f"show epoch num: {truncated_epochs[-1]}")
    y_label = input('Y label? [default="Travel distance in voxels"]')
    if not y_label:
        y_label = "Travel distance in voxels"

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
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()


def draw_separate_reward_curves(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    print(f"show epoch num: {truncated_epochs[-1]}")
    labels = [f"record {i}" for i in range(len(records))]
    y_label = input('Y label? [default="Travel distance in voxels"]')
    if not y_label:
        y_label = "Travel distance in voxels"
    if input("Customize legend labels? [y/n] ").lower() == "y":
        for i in range(len(records)):
            labels[i] = input(f"Legend label for record {i}: ")
    current_curve_max = -np.inf
    for record_idx, record in enumerate(records):
        record_rewards = generate_rewards_for_trial(record)
        mean = np.zeros(len(truncated_epochs))
        shift = np.zeros(len(truncated_epochs))
        for epoch in truncated_epochs:
            epoch_rewards = record_rewards[epoch]
            mean[epoch - 1] = np.mean(epoch_rewards)
            shift[epoch - 1] = (
                np.std(epoch_rewards) * 2.576 / np.sqrt(len(epoch_rewards))
            )
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
    plt.ylabel(y_label)
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()
