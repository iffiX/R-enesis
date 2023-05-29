import os
import pickle
import numpy as np


def find_data_directories(root_dir_of_trials, trial_filter):
    trial_data_dirs = []
    for trial_dir in os.listdir(root_dir_of_trials):
        if os.path.isdir(os.path.join(root_dir_of_trials, trial_dir)) and trial_filter(
            trial_dir
        ):
            sub_dir = [
                sdir
                for sdir in os.listdir(os.path.join(root_dir_of_trials, trial_dir))
                if os.path.isdir(os.path.join(root_dir_of_trials, trial_dir, sdir))
            ][0]
            trial_data_dirs.append(
                os.path.join(root_dir_of_trials, trial_dir, sub_dir, "data")
            )
    return trial_data_dirs


def generate_max_reward_matrix(data_dirs, trim_epochs=1500):
    all_trial_max_rewards = []
    for data_dir in data_dirs:
        data_files = [f for f in os.listdir(data_dir) if f.startswith("data")]
        trial_max_rewards = []
        data_files = sorted(data_files, key=lambda x: int(x.split("_")[2]))
        for data_file in data_files:
            print(data_file)
            with open(os.path.join(data_dir, data_file), "rb") as file:
                data = pickle.load(file)
                trial_max_rewards.append(np.max([entry["reward"] for entry in data]))
        all_trial_max_rewards.append(trial_max_rewards[:trim_epochs])
    return np.array(all_trial_max_rewards)


if __name__ == "__main__":
    with_bias_dirs = find_data_directories(
        "/home/mlw0504/Projects/R-enesis_results/task_volume",
        lambda x: "std-bias=2voxels" in x,
    )
    with open("with_bias_rewards.pickle", "wb") as file:
        pickle.dump(generate_max_reward_matrix(with_bias_dirs), file)

    without_bias_dirs = find_data_directories(
        "/home/mlw0504/Projects/R-enesis_results/task_volume",
        lambda x: "std-bias=none" in x,
    )
    with open("without_bias_rewards.pickle", "wb") as file:
        pickle.dump(generate_max_reward_matrix(without_bias_dirs), file)
