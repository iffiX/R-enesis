import os
import pickle
import pprint
import cc3d
import numpy as np
import matplotlib.pyplot as plt
from renesis.utils.robot import get_robot_voxels_from_voxels
from renesis.utils.metrics import (
    get_volume,
    get_surface_voxels,
    get_surface_area,
    get_section_num,
    get_reflection_symmetry,
)
from experiments.navigator.trial import TrialRecord


def compute_robot_metrics(trial_record: TrialRecord, show_epoch: int = -1):
    if show_epoch not in trial_record.epochs:
        if show_epoch > 0:
            print(f"Required epoch {show_epoch} not found")
        print("Use epoch with max reward")
        show_epoch = trial_record.max_reward_epoch
        print(f"Show epoch {show_epoch}")

    with open(
        os.path.join(
            trial_record.data_dir, trial_record.epoch_files[show_epoch].data_file_name
        ),
        "rb",
    ) as file:
        data = pickle.load(file)
        data = sorted(data, key=lambda d: d["reward"], reverse=True)

        robot_voxels, _ = get_robot_voxels_from_voxels(data[0]["voxels"])

        metrics = {}
        metrics["volume"] = get_volume(robot_voxels)
        metrics["surface_area"] = get_surface_area(robot_voxels)
        metrics["surface_voxels"] = get_surface_voxels(robot_voxels)
        metrics["surface_area_to_total_volume_ratio"] = (
            metrics["surface_area"] / metrics["volume"]
        )
        metrics["surface_voxels_to_total_volume_ratio"] = (
            metrics["surface_voxels"] / metrics["volume"]
        )
        metrics["section_num"] = get_section_num(robot_voxels)
        metrics["reflection_symmetry"] = get_reflection_symmetry(robot_voxels)
        pprint.pprint(metrics)

        colors = np.empty(robot_voxels.shape, dtype=object)
        colors[robot_voxels == 1] = "blue"
        colors[robot_voxels == 2] = "green"
        colors[robot_voxels == 3] = "red"
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.voxels(robot_voxels != 0, facecolors=colors)
        ax.axis("equal")
        fig.show()
