import pickle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from navigator.functions.multi.draw_reward_curves import smooth


def get_self_and_other_prediction(values):
    trial_num = values.shape[2]
    # self pred shape: [resolution, trial_num, batch_size]
    self_ref = values[:, 0, range(trial_num), range(trial_num)]
    self_pred = values[:, 1, range(trial_num), range(trial_num)]
    # other pred shape: [resolution, trial_num, trial_num - 1, batch_size]
    other_ref = np.stack(
        [
            np.concatenate((values[:, 0, i, :i], values[:, 0, i, i + 1 :]), axis=1)
            for i in range(trial_num)
        ],
        axis=1,
    )
    other_pred = np.stack(
        [
            np.concatenate((values[:, 1, i, :i], values[:, 1, i, i + 1 :]), axis=1)
            for i in range(trial_num)
        ],
        axis=1,
    )
    return (
        self_ref.reshape(self_ref.shape[0], -1),
        self_pred.reshape(self_pred.shape[0], -1),
        other_ref.reshape(other_ref.shape[0], -1),
        other_pred.reshape(other_pred.shape[0], -1),
    )


def get_confidence_interval(error):
    std = np.std(error, axis=1)
    shift = std * 2.576 / np.sqrt(error.shape[1])
    y_errors = (shift, shift)
    return y_errors


def plot(
    ax0,
    ax1,
    epochs,
    trained_values,
    untrained_values,
    ylabel,
    ax0_label,
    ax1_label,
    show_legend=False,
):
    (
        trained_self_ref,
        trained_self_pred,
        trained_other_ref,
        trained_other_pred,
    ) = get_self_and_other_prediction(trained_values)
    _, untrained_self_pred, _, untrained_other_pred = get_self_and_other_prediction(
        untrained_values
    )
    for ax, curve, curve_label, curve_color, fill_color in (
        (ax0, trained_self_ref, "True value", "grey", "lightgrey"),
        (ax0, trained_self_pred, "Trained critic", "steelblue", "skyblue"),
        (ax0, untrained_self_pred, "Untrained critic", "red", "lightsalmon"),
        (ax1, trained_other_ref, "True value", "grey", "lightgrey"),
        (ax1, trained_other_pred, "Trained critic", "steelblue", "skyblue"),
        (ax1, untrained_other_pred, "Untrained critic", "red", "lightsalmon"),
    ):
        center = np.mean(curve, axis=1)
        ax.plot(epochs, smooth(center), label=curve_label, color=curve_color)
        min_offset, max_offset = get_confidence_interval(curve)
        ax.fill_between(
            epochs,
            center - min_offset,
            center + max_offset,
            color=fill_color,
            alpha=0.7,
        )
        ax.text(
            0.05,
            0.95,
            ax0_label if ax is ax0 else ax1_label,
            transform=ax.transAxes,
            color="black",
            fontsize=15,
            verticalalignment="top",
        )
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
        ax.set_title("In-domain" if ax is ax0 else "Out-of-domain", fontsize=14)
        if ax is ax0:
            ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
        )


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(5, 3), layout="constrained")
    # with open("generated_data/vec_patch_shape_trained.data", "rb") as file, open(
    #     "generated_data/vec_patch_shape_untrained.data", "rb"
    # ) as file2:
    #     # shape [resolution, 2, trial_num, trial_num, batch]
    #     # 2: first is rewards, second is predicted rewards V(s_100)
    #     # first trial_num: trial critic number
    #     # second trial_num: critic values for each trial
    #     epochs, trained_values = pickle.load(file)
    #     _, untrained_values = pickle.load(file2)
    #     plot(
    #         axs[0],
    #         axs[1],
    #         epochs,
    #         trained_values,
    #         untrained_values,
    #         "Volume\n(number of voxels)",
    #         "A",
    #         "B",
    #         True,
    #     )

    with open("generated_data/vec_patch_voxcraft_trained.data", "rb") as file, open(
        "generated_data/vec_patch_voxcraft_untrained.data", "rb"
    ) as file2:
        epochs, trained_values = pickle.load(file)
        _, untrained_values = pickle.load(file2)
        plot(
            axs[0],
            axs[1],
            epochs,
            trained_values,
            untrained_values,
            "Displacement\n(voxel length)",
            "A",
            "B",
        )
    # handles, labels = axs[1].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     # loc="center",
    #     # bbox_to_anchor=(0.5, 1.1),
    #     # ncols=3,
    #     loc="center right",
    #     bbox_to_anchor=(1.4, 0.5),
    #     fontsize=12,
    # )
    fig.align_ylabels(axs)
    fig.savefig(
        f"generated_data/critic.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()
