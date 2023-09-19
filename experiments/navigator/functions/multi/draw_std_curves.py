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


def generate_std_metrics_for_trial(record: TrialRecord):
    metrics = {}
    cache_path = os.path.join(
        get_cache_directory("std_cache"),
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
                pool.imap(compute_std_metrics_for_epoch, epoch_data_file_paths),
                total=len(epoch_data_file_paths),
            )
        )
        for epoch, result in zip(record.epochs, results):
            metrics[epoch] = result
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def compute_std_metrics_for_epoch(epoch_data_file_path):
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        # shape [batch_size, T, (3 + material_num (k)) * 2]
        all_step_dists = np.array(
            [d["step_dists"] for d in data if len(d["steps"]) > 0]
        )
        # return shape: [T, 3 + material_num (k)]
        return np.mean(all_step_dists[:, :, all_step_dists.shape[-1] // 2 :], axis=0)


def draw_std_curves(records: List[TrialRecord]):
    # Note: since std curves evaluates the average std value output by network
    # we do not combine results from multiple trials epoch-wise since each model
    # trained in each trial may be very different, we first average std per trial,
    # then average across trials
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    std_curves = []
    print(f"show epoch num: {truncated_epochs[-1]}")
    for record_idx, record in enumerate(records):
        metrics = generate_std_metrics_for_trial(record)
        std_curves.append([])
        for epoch in truncated_epochs:
            # mean
            std_curves[-1].append(metrics[epoch])
    # std_curves shape: [trial_num, epoch_num, T, 3 + material_num(k)]
    std_curves = np.array(std_curves)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # std_curves_for_T shape: [epoch_num, T]
    std_curves_for_T = np.mean(np.exp(std_curves[:, :, :, :3]) / 4 * 20, axis=(0, 3))
    for i in range(std_curves.shape[2]):
        axs[0].plot(
            truncated_epochs,
            smooth(std_curves_for_T[:, i]),
            color=[
                0,
                (std_curves.shape[2] - i) / std_curves.shape[2],
                (i + 1) / std_curves.shape[2],
                1,
            ],
            label=f"step {i}",
        )
    axs[0].set_ylabel("Std in voxels")
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Std for each step")
    # axs[0].legend()

    # std_curves_for_xyz shape: [epoch_num, 3]
    std_curves_for_xyz = np.mean(np.exp(std_curves[:, :, :, :3]) / 4 * 20, axis=(0, 2))
    for i in range(3):
        axs[1].plot(
            truncated_epochs,
            smooth(std_curves_for_xyz[:, i]),
            color=f"C{i}",
            label=["x", "y", "z"][i],
        )

    axs[1].set_ylabel("Std in voxels")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Std for x,y,z")
    axs[1].legend()
    fig.show()
