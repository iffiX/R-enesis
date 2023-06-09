import os
import cc3d
import json
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Ellipse
from moviepy.video.io.bindings import mplfig_to_npimage
from renesis.env_model.patch import normalize, PatchModel, PatchSphereModel
from renesis.utils.media import create_video
from experiments.navigator.trial import TrialRecord

np.set_printoptions(precision=4, threshold=10000, suppress=True)

MAT_NULL = [0, 0, 0]

MAT_BLUE = [25 / 255, 130 / 255, 196 / 255]

MAT_GREEN = [138 / 255, 201 / 255, 38 / 255]

MAT_RED = [255 / 255, 89 / 255, 94 / 255]


def round_up_to_multiple_of(value, base):
    remainder = value % base
    if remainder == 0:
        return value
    else:
        return (value // base + 1) * base


def get_tick_and_labels(
    min, max, min_label=None, max_label=None, interval=5, show_intermediate=False
):
    ticks = [min]
    labels = [min_label if min_label is not None else str(min)]
    for tick in np.arange(round_up_to_multiple_of(min, interval), max, interval):
        if tick != min and tick != max:
            ticks.append(tick)
            labels.append("" if not show_intermediate else str(tick))
    ticks.append(max)
    labels.append(max_label if max_label is not None else str(max))
    return ticks, labels


def plot_step(
    title,
    prev_voxels,
    voxels,
    action_pos,
    action_pos_dist_mean,
    action_pos_dist_std_in_voxels,
):
    unchanged_non_empty = np.logical_and(prev_voxels != 0, prev_voxels == voxels)
    changed = prev_voxels != voxels
    unchanged_voxels = np.where(unchanged_non_empty, prev_voxels, 0)
    # Set unchanged parts to -1 so we can see null material
    changed_voxels = np.where(changed, voxels, -1)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    colors = np.zeros(list(voxels.shape) + [4], dtype=np.float32)
    edge_colors = np.zeros(list(voxels.shape) + [4], dtype=np.float32)
    colors[np.logical_and(prev_voxels != 0, voxels == 0)] = [1, 1, 1, 1]
    changed_material_types = set(
        ma.compressed(ma.masked_array(changed_voxels, ~changed))
    )

    if len(changed_material_types) == 1:
        changed_material_type = int(list(changed_material_types)[0])
        step_color = [MAT_NULL, MAT_BLUE, MAT_GREEN, MAT_RED][changed_material_type]
    elif len(changed_material_types) == 0:
        step_color = MAT_NULL
    else:
        raise ValueError("More than 1 new material type detected in changed section")

    # draw voxels
    if len(changed_material_types) == 1:
        for plot_voxels, is_changed in (
            (unchanged_voxels, False),
            (changed_voxels, True),
        ):
            if is_changed:
                colors[plot_voxels == 0] = [*MAT_NULL, 1]
                colors[plot_voxels == 1] = [*MAT_BLUE, 1]
                colors[plot_voxels == 2] = [*MAT_GREEN, 1]
                colors[plot_voxels == 3] = [*MAT_RED, 1]
                edge_colors[plot_voxels != -1] = [1, 1, 1, 1]
            else:
                colors[plot_voxels != 0] = [0.7] * 3 + [0.3]
        ax.voxels(
            np.logical_or(unchanged_non_empty, changed),
            facecolors=colors,
            edgecolors=edge_colors,
        )
    else:
        # no change, plot original voxels
        colors[unchanged_voxels != 0] = [0.7] * 3 + [0.3]
        ax.voxels(unchanged_non_empty, facecolors=colors)
    # draw action center
    ax.scatter(
        action_pos[0],
        action_pos[1],
        action_pos[2],
        color="red",
        marker="o",
        s=10,
    )
    # draw action projection line
    for i in range(3):
        start_end = np.stack([action_pos, action_pos])
        start_end[0, i] = 0
        ax.plot(
            start_end[:, 0],
            start_end[:, 1],
            start_end[:, 2],
            color=step_color,
            linestyle="--",
        )

    # xy -> z, xz -> y, xy -> z
    for idx, zdir in (((0, 1), "z"), ((0, 2), "y"), ((1, 2), "x")):
        # draw center
        center = np.zeros_like(action_pos_dist_mean)
        for sub_idx in idx:
            center[sub_idx] = action_pos_dist_mean[sub_idx]

        ax.scatter(center[0], center[1], center[2], color=step_color, marker=".", s=10)
        # draw range
        for sigma in (1, 2, 3):
            e = Ellipse(
                (action_pos_dist_mean[idx[0]], action_pos_dist_mean[idx[1]]),
                action_pos_dist_std_in_voxels[idx[0]] * sigma,
                action_pos_dist_std_in_voxels[idx[1]] * sigma,
                fill=False,
                edgecolor=[
                    *step_color,
                    1 - 0.2 * sigma,
                ],
            )
            ax.add_patch(e)
            art3d.pathpatch_2d_to_3d(e, z=0, zdir=zdir)
    ax.set_title(title, fontsize=17)
    ax.set_xlim([0, voxels.shape[0]])
    ax.set_ylim([0, voxels.shape[1]])
    ax.set_zlim([0, voxels.shape[2]])

    ax.set_xticks(
        *get_tick_and_labels(0, voxels.shape[0], min_label="", max_label=""),
    )
    ax.set_yticks(
        *get_tick_and_labels(0, voxels.shape[1], min_label="", max_label=""),
    )
    ax.set_zticks(
        *get_tick_and_labels(0, voxels.shape[2], min_label="", max_label=""),
    )
    # ax.set_xlabel("X", fontsize=15)
    # ax.set_ylabel("Y", fontsize=15)
    # ax.set_zlabel("Z", fontsize=15)
    ax.grid(False)
    ax.axis("equal")
    ax.invert_yaxis()

    return fig


def plot_largest_connected_components(
    title,
    voxels,
):
    labels, label_num = cc3d.connected_components(
        voxels != 0, connectivity=6, return_N=True, out_dtype=np.uint32
    )
    count = np.bincount(labels.reshape(-1), minlength=label_num)
    # Ignore label 0, which is non-occupied space
    count[0] = 0
    largest_connected_component_label = np.argmax(count)
    colors = np.zeros(list(voxels.shape) + [4], dtype=np.float32)
    edge_colors = np.zeros(list(voxels.shape) + [4], dtype=np.float32)
    # draw voxels
    colors[labels != largest_connected_component_label] = [0.7, 0.7, 0.7, 0.3]
    colors[np.logical_and(labels == largest_connected_component_label, voxels == 1)] = [
        *MAT_BLUE,
        1,
    ]
    colors[np.logical_and(labels == largest_connected_component_label, voxels == 2)] = [
        *MAT_GREEN,
        1,
    ]
    colors[np.logical_and(labels == largest_connected_component_label, voxels == 3)] = [
        *MAT_RED,
        1,
    ]
    edge_colors[labels == largest_connected_component_label] = [1, 1, 1, 1]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.voxels(voxels != 0, facecolors=colors, edgecolors=edge_colors)

    ax.set_title(title, fontsize=17)
    ax.set_xlim([0, voxels.shape[0]])
    ax.set_ylim([0, voxels.shape[1]])
    ax.set_zlim([0, voxels.shape[2]])
    ax.set_xticks(
        *get_tick_and_labels(0, voxels.shape[0], min_label="", max_label=""),
    )
    ax.set_yticks(
        *get_tick_and_labels(0, voxels.shape[1], min_label="", max_label=""),
    )
    ax.set_zticks(
        *get_tick_and_labels(0, voxels.shape[2], min_label="", max_label=""),
    )
    # ax.set_xlabel("X", fontsize=15)
    # ax.set_ylabel("Y", fontsize=15)
    # ax.set_zlabel("Z", fontsize=15)
    ax.grid(False)
    ax.axis("equal")
    ax.invert_yaxis()

    return fig


def draw_generation_process(trial_record: TrialRecord, show_epoch: int = -1):
    with open(os.path.join(trial_record.trial_dir, "params.json"), "r") as file:
        params = json.load(file)
    patch_size = params["env_config"]["patch_size"]
    env_model_cls = (
        PatchModel if "PatchSphere" not in params["env"] else PatchSphereModel
    )

    if show_epoch not in trial_record.epochs:
        if show_epoch > 0:
            print(f"Required epoch {show_epoch} not found")
        print("Use epoch with max reward")
        show_epoch = trial_record.max_reward_epoch
    with open(
        os.path.join(
            trial_record.data_dir, trial_record.epoch_files[show_epoch].data_file_name
        ),
        "rb",
    ) as file:
        data = pickle.load(file)
        data = sorted(data, key=lambda d: d["reward"], reverse=True)

    # Pick the best robot to show the process
    dimension_size = tuple(data[0]["voxels"].shape)
    best_robot_idx = int(np.argmax([d["reward"] for d in data]))
    best_robot_actions = np.array(data[best_robot_idx]["steps"])
    best_robot_action_dists = np.array(data[best_robot_idx]["step_dists"])
    env_model = env_model_cls(
        materials=(0, 1, 2, 3),
        dimension_size=dimension_size,
        patch_size=patch_size,
        max_patch_num=1000000,
    )
    std_offset = best_robot_action_dists.shape[-1] // 2
    frames = []
    for i in range(len(best_robot_actions)):
        action = normalize(best_robot_actions[i], mode="clip")
        action_pos = action[:3] * np.array(dimension_size)
        action_pos_dist_mean = normalize(
            best_robot_action_dists[i, :3], mode="clip"
        ) * np.array(dimension_size)
        action_pos_dist_std_in_voxels = (
            (np.exp(best_robot_action_dists[i, std_offset : std_offset + 3]) / 4)
            * 3
            * dimension_size
        )
        env_model.step(action)

        if np.all(env_model.voxels == env_model.prev_voxels):
            print(f"Warning: step {i} is unchanged")
        fig = plot_step(
            # f" $a_{{{i + 1}}} \sim \mathcal{{N}}("
            # f"\mathbf{{\mu}}=[{action_pos_dist_mean[0]:.1f}, {action_pos_dist_mean[1]:.1f}, {action_pos_dist_mean[2]:.1f}], "
            # f"\mathbf{{\Sigma}}=[{action_pos_dist_std_in_voxels[0]:.1f}, {action_pos_dist_std_in_voxels[1]:.1f}, {action_pos_dist_std_in_voxels[2]:.1f}])$",
            f" $a_{{{i + 1}}} \sim \mathcal{{N}}("
            f"\mathbf{{\mu_{{{i + 1}}}}}, "
            f"\mathbf{{\Sigma_{{{i + 1}}}}})$",
            env_model.prev_voxels,
            env_model.voxels,
            action_pos,
            action_pos_dist_mean,
            action_pos_dist_std_in_voxels,
        )
        frames.append(mplfig_to_npimage(fig))
        plt.savefig(
            f"generated_data/step={i + 1}.png", bbox_inches="tight", pad_inches=0
        )
    fig = plot_largest_connected_components(
        f"Body after $a_{{{len(best_robot_actions)}}}$", env_model.voxels
    )
    end_frame = mplfig_to_npimage(fig)

    frames += [end_frame] * 5
    plt.savefig(
        f"generated_data/robot_{show_epoch}.png", bbox_inches="tight", pad_inches=0
    )
    create_video(frames, path="generated_data/", filename=f"robot_{show_epoch}", fps=5)
