import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from multiprocessing.pool import Pool
from renesis.utils.robot import get_robot_voxels_from_voxels
from renesis.utils.metrics import (
    get_volume,
    get_surface_voxels,
    get_surface_area,
    get_section_num,
    get_reflection_symmetry,
    get_gzip_compressed_ratio,
    get_passive_material_ratio,
)
from experiments.navigator.trial import TrialRecord
from experiments.navigator.utils import get_cache_directory
from experiments.navigator.functions.multi.draw_reward_curves import smooth

_robot_metrics_keys = [
    "volume",
    "surface_area",
    "surface_voxels",
    "surface_area_to_total_volume_ratio",
    "surface_voxels_to_total_volume_ratio",
    "section_num",
    "reflection_symmetry",
    "gzip_compressed_ratio",
    "passive_material_ratio",
    "largest_connected_component_ratio",
]

_show_robot_metrics_keys = [
    "volume",
    "surface_area",
    "surface_voxels",
    "surface_area_to_total_volume_ratio",
    "surface_voxels_to_total_volume_ratio",
    "section_num",
    "reflection_symmetry",
    "gzip_compressed_ratio",
    "passive_material_ratio",
    "largest_connected_component_ratio",
]


def generate_robot_metrics_for_trial(record: TrialRecord):
    cache_path = os.path.join(
        get_cache_directory("robot_metrics_cache"),
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
                pool.imap(compute_metrics_for_epoch, epoch_data_file_paths),
                total=len(epoch_data_file_paths),
            )
        )
        metrics = {
            key: [result[key] for result in results] for key in _robot_metrics_keys
        }
        with open(os.path.join(cache_path), "wb") as cache_file:
            pickle.dump(metrics, cache_file)
        return metrics


def compute_metrics_for_epoch(epoch_data_file_path):
    metrics = {}
    with open(
        epoch_data_file_path,
        "rb",
    ) as file:
        data = pickle.load(file)
        results = np.array(list(map(compute_metrics_for_robot, data))).transpose()
        for key, result in zip(_robot_metrics_keys, results):
            value = np.nanmean(result)
            metrics[key] = value if not np.isnan(value) else 0
    return metrics


def compute_metrics_for_robot(robot_data):
    robot_voxels, _ = get_robot_voxels_from_voxels(robot_data["voxels"])
    volume = get_volume(robot_voxels)
    surface_area = get_surface_area(robot_voxels)
    surface_voxels = get_surface_voxels(robot_voxels)
    return (
        volume,
        surface_area,
        surface_voxels,
        surface_area / volume,
        surface_voxels / volume,
        get_section_num(robot_voxels),
        get_reflection_symmetry(robot_voxels),
        get_gzip_compressed_ratio(robot_voxels),
        get_passive_material_ratio(robot_voxels),
        get_volume(robot_voxels) / get_volume(robot_data["voxels"]),
    )


def draw_robot_metric_curves(records: List[TrialRecord]):
    truncated_epochs = list(range(1, min(record.epochs[-1] for record in records) + 1))
    row_size = (len(_show_robot_metrics_keys) + 2) // 3
    col_size = 3
    fig, axs = plt.subplots(row_size, col_size, figsize=(4 * col_size, 4 * row_size))
    for row in range(row_size):
        for col in range(col_size):
            idx = row * col_size + col
            if idx < len(_show_robot_metrics_keys):
                key = _show_robot_metrics_keys[idx]
                metric_curves = np.zeros([len(records), len(truncated_epochs)])
                print(f"show epoch num: {truncated_epochs[-1]}")
                for record_idx, record in enumerate(records):
                    metrics = generate_robot_metrics_for_trial(record)
                    metric_curves[record_idx] = metrics[key][: metric_curves.shape[1]]
                    # plt.plot(truncated_epochs, reward_curves[record_idx, :])
                std = np.std(metric_curves, axis=0)
                mean = np.mean(metric_curves, axis=0)
                shift = std * 2.576 / np.sqrt(len(records))
                axs[row][col].fill_between(
                    truncated_epochs,
                    mean - shift,
                    mean + shift,
                    color="lightgrey",
                )
                axs[row][col].plot(
                    truncated_epochs,
                    smooth(mean),
                    color="grey",
                )

                axs[row][col].set_ylim(0, np.max(mean) * 1.1)
                if key != "surface_voxels_to_total_volume_ratio":
                    axs[row][col].set_ylabel(key.replace("_", " ").capitalize())
                else:
                    axs[row][col].set_ylabel("Surface voxels to volume ratio")
                axs[row][col].set_xlabel("Epoch")
    fig.show()
