import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from renesis.utils.robot import get_robot_voxels_from_voxels
from experiments.navigator.trial import TrialRecord
from experiments.navigator.functions.multi.draw_reward_curves import (
    generate_reward_metrics_for_trial,
    smooth,
)

divide_colors = ["orangered", "orange", "gold", "green", "skyblue", "darkviolet"]
divide_points = [0, 0.1, 0.2, 0.4, 0.6, 1]


def draw_volume_task_result(
    records: List[TrialRecord],
):
    fig, axd = plt.subplot_mosaic(
        [
            ["reward", "0", "1", "2"],
            ["reward", "3", "4", "5"],
        ],
        figsize=(12.5, 5),
        gridspec_kw={"width_ratios": [2, 1, 1, 1]},
        layout="constrained",
        per_subplot_kw={
            "reward": {},
            "0": {"projection": "3d"},
            "1": {"projection": "3d"},
            "2": {"projection": "3d"},
            "3": {"projection": "3d"},
            "4": {"projection": "3d"},
            "5": {"projection": "3d"},
        },
    )

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
    axd["reward"].fill_between(
        truncated_epochs,
        mean - shift,
        mean + shift,
        color="lightgrey",
    )
    axd["reward"].plot(
        truncated_epochs,
        smooth(mean),
        color="grey",
    )
    y_limit = np.max(mean + shift) * 1.1
    axd["reward"].set_ylim(0, y_limit)
    axd["reward"].set_xlabel("Epoch", fontsize=15)
    axd["reward"].set_ylabel("Volume (number of voxels)", fontsize=15)
    axd["reward"].xaxis.set_tick_params(labelsize=14)
    axd["reward"].yaxis.set_tick_params(labelsize=14)
    for divide_idx, (divide_color, divide_point) in enumerate(
        zip(divide_colors, divide_points)
    ):
        epoch_idx = int(divide_point * (len(truncated_epochs) - 1))
        show_epoch = truncated_epochs[epoch_idx]
        all_data = []
        for record in records:
            with open(
                os.path.join(
                    record.data_dir,
                    record.epoch_files[show_epoch].data_file_name,
                ),
                "rb",
            ) as file:
                all_data += pickle.load(file)
        data = sorted(
            all_data,
            key=lambda d: np.abs(d["reward"] - mean[show_epoch - 1]),
        )[0]
        print(f"{show_epoch} {mean[show_epoch - 1]}")
        key = str(divide_idx)
        axd[key].set_title(f"Volume={int(data['reward'])}", y=-0.15)
        axd[key].set_xticks([])
        axd[key].set_yticks([])
        axd[key].set_zticks([])

        robot_voxels, robot_occupied = get_robot_voxels_from_voxels(data["voxels"])
        colors = np.empty(robot_voxels.shape, dtype=object)
        colors[robot_voxels == 1] = divide_color
        axd[key].voxels(robot_occupied, facecolors=colors, edgecolors=(1, 1, 1, 0.7))
        axd[key].axis("equal")
        axd[key].text2D(
            0.05,
            0.95,
            chr(int(key) + ord("A")),
            transform=axd[key].transAxes,
            color="black",
            fontsize=15,
            verticalalignment="top",
        )
        axd["reward"].plot(
            [show_epoch], [data["reward"]], marker="o", color=divide_color
        )
        axd["reward"].axvline(
            x=show_epoch,
            ymin=0,
            ymax=data["reward"] / y_limit,
            color=divide_color,
            linestyle="--",
        )
        axd["reward"].text(
            show_epoch,
            data["reward"] + 15,
            chr(int(key) + ord("A")),
            fontsize=12,
        )
    fig.savefig(
        f"generated_data/volume.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()
