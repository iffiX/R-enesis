import pickle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def get_self_and_other_error(values):
    trial_num = values.shape[2]
    error = np.abs(values[:, 0] - values[:, 1])
    # self error shape: [resolution, trial_num, batch_size]
    self_error = error[:, range(trial_num), range(trial_num)]
    # other error shape: [resolution, trial_num, trial_num - 1, batch_size]
    other_error = np.stack(
        [
            np.concatenate((error[:, i, :i], error[:, i, i + 1 :]), axis=1)
            for i in range(trial_num)
        ],
        axis=1,
    )
    return self_error.reshape(self_error.shape[0], -1), other_error.reshape(
        other_error.shape[0], -1
    )


def get_confidence_interval(error):
    # error shape: [resolution, any_size]
    # compute confidence interval for each vector in resolution dim
    # ci_results = [
    #     sc.stats.bootstrap((error[i],), np.mean, confidence_level=0.99)
    #     for i in range(error.shape[0])
    # ]
    # ci_min, ci_max = zip(
    #     *[(r.confidence_interval.low, r.confidence_interval.high) for r in ci_results]
    # )
    # y_errors = np.abs(np.array((ci_min, ci_max)) - np.mean(error, axis=1))
    # shape [2, resolution]

    std = np.std(error, axis=1)
    shift = std * 2.576 / np.sqrt(error.shape[1])
    y_errors = (shift, shift)
    return y_errors


def plot(ax, epochs, trained_values, untrained_values, ylabel):
    trained_self_error, trained_other_error = get_self_and_other_error(trained_values)
    untrained_self_error, untrained_other_error = get_self_and_other_error(
        untrained_values
    )
    print(np.mean(trained_self_error))
    print(np.mean(trained_other_error))
    print(np.mean(untrained_self_error))
    print(np.mean(untrained_other_error))
    for error, error_label, curve_color, fill_color in (
        (trained_self_error, "Trained in-domain", "steelblue", "skyblue"),
        (trained_other_error, "Trained out-of-domain", "red", "lightsalmon"),
        (untrained_self_error, "Untrained in-domain", "orange", "bisque"),
        (untrained_other_error, "Untrained out-of-domain", "green", "lightgreen"),
    ):
        center = np.mean(error, axis=1)
        ax.plot(epochs, center, label=error_label, color=curve_color)
        min_offset, max_offset = get_confidence_interval(error)
        ax.fill_between(
            epochs,
            center - min_offset,
            center + max_offset,
            color=fill_color,
            opacity=0.5,
        )
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=12)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    with open("generated_data/vec_patch_shape_trained.data", "rb") as file, open(
        "generated_data/vec_patch_shape_untrained.data", "rb"
    ) as file2:
        # shape [resolution, 2, trial_num, trial_num, batch]
        # 2: first is rewards, second is predicted rewards V(s_100)
        # first trial_num: trial critic number
        # second trial_num: critic values for each trial
        epochs, trained_values = pickle.load(file)
        _, untrained_values = pickle.load(file2)
        plot(
            axs[0],
            epochs,
            trained_values,
            untrained_values,
            "Critic volume error (number of voxels)",
        )

    with open("generated_data/vec_patch_voxcraft_trained.data", "rb") as file, open(
        "generated_data/vec_patch_voxcraft_untrained.data", "rb"
    ) as file2:
        epochs, trained_values = pickle.load(file)
        _, untrained_values = pickle.load(file2)
        plot(
            axs[1],
            epochs,
            trained_values,
            untrained_values,
            "Critic displacement error (voxel length)",
        )
    fig.savefig(
        f"generated_data/critic.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.show()
