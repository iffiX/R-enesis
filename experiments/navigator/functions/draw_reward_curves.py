import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from experiments.navigator.trial import TrialRecord

_cahce_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), os.path.pardir, "cache"
)


def generate_reward_metrics_for_trial(record: TrialRecord):
    metrics = {}
    cache_path = os.path.join(_cahce_dir, record.trial_dir.replace("/", "#") + ".cache")
    if os.path.exists(cache_path):
        with open(os.path.join(cache_path), "rb") as cache_file:
            return pickle.load(cache_file)
    else:
        for epoch, epoch_files in tqdm.tqdm(
            record.epoch_files.items(), total=len(record.epoch_files)
        ):
            with open(
                os.path.join(record.data_dir, epoch_files.data_file_name), "rb"
            ) as file:
                data = pickle.load(file)
                rewards = np.array([d["reward"] for d in data])
                metrics[epoch] = (np.max(rewards), np.min(rewards), np.mean(rewards))
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def smooth(scalars: np.array, weight: float) -> np.array:
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return np.array(smoothed)


def draw_reward_curves(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    reward_curves = np.zeros([len(records), len(truncated_epochs)])
    print(f"show epoch num: {truncated_epochs[-1]}")
    for record_idx, record in enumerate(records):
        metrics = generate_reward_metrics_for_trial(record)
        for epoch in truncated_epochs:
            # mean
            reward_curves[record_idx, epoch - 1] = metrics[epoch][2]
        # plt.plot(truncated_epochs, reward_curves[record_idx, :])
    std = np.std(reward_curves, axis=0)
    mean = np.mean(reward_curves, axis=0)
    shift = std * 1.96 / np.sqrt(len(records))
    plt.fill_between(
        truncated_epochs,
        mean - shift,
        mean + shift,
        color="skyblue",
    )
    plt.plot(
        truncated_epochs,
        smooth(mean, 0.9),
        color="steelblue",
    )

    plt.ylim(0, np.max(mean + shift) * 1.1)
    plt.ylabel("Travel distance in voxels")
    plt.xlabel("Epoch")
    plt.title("Rewards")
    plt.show()
