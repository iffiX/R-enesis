import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from renesis.utils.robot import get_robot_voxels_from_voxels
from experiments.navigator.trial import TrialRecord


def draw_robots(trial_record: TrialRecord, show_epoch: int = -1, show_num: int = 8):
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
        if not 0 < show_num <= len(data):
            print(f"show num {show_num} is invalid")
            show_num = len(data)
        data = sorted(data, key=lambda d: d["reward"], reverse=True)[:show_num]
        row_size = int(np.ceil(np.sqrt(len(data) / 2)))
        col_size = row_size * 2

        fig, axs = plt.subplots(row_size, col_size, subplot_kw={"projection": "3d"})

        for row in range(row_size):
            for col in range(col_size):
                idx = row * col_size + col

                if idx < len(data):
                    if data[idx]["reward"] <= 0:
                        axs[row][col].set_facecolor([0.2, 0.2, 0.2])
                    axs[row][col].set_title(f"{data[idx]['reward']:.3f}")
                    axs[row][col].set_xticks([])
                    axs[row][col].set_yticks([])
                    axs[row][col].set_zticks([])

                    robot_voxels, robot_occupied = get_robot_voxels_from_voxels(
                        data[idx]["voxels"]
                    )
                    colors = np.empty(robot_voxels.shape, dtype=object)
                    colors[robot_voxels == 1] = "blue"
                    colors[robot_voxels == 2] = "green"
                    colors[robot_voxels == 3] = "red"
                    axs[row][col].voxels(robot_occupied, facecolors=colors)
                    axs[row][col].axis("equal")
        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except:
            pass
        # fig.set_figheight(4.5 * row_size)
        # fig.set_figwidth(4.5 * col_size)
        # fig.savefig(
        #     f"data/generated_data/robots_it_{show_epoch}.pdf",
        #     bbox_inches="tight",
        #     pad_inches=0.5,
        # )
        fig.show()
