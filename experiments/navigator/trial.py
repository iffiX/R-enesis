import os
import re
from typing import *


class EpochFiles:
    def __init__(
        self, data_file_name, history_file_name, robot_file_name, epoch, reward
    ):
        self.data_file_name = data_file_name
        self.history_file_name = history_file_name
        self.robot_file_name = robot_file_name
        self.epoch = epoch
        self.reward = reward


class TrialRecord:
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        self.comment = self.get_comment()
        self.epoch_files = self.get_epoch_files()  # type: Dict[int, EpochFiles]
        self.checkpoints = self.get_checkpoints()  # type: List[str]
        self.epochs = sorted(list(self.epoch_files.keys()))
        self.max_reward_epoch, self.max_reward = sorted(
            [(ef.epoch, ef.reward) for ef in self.epoch_files.values()],
            key=lambda x: x[1],
            reverse=True,
        )[0]

    @property
    def data_dir(self):
        path = os.path.join(self.trial_dir, "data")
        return path if os.path.exists(path) else None

    @property
    def code_dir(self):
        path = os.path.join(self.trial_dir, "code")
        return path if os.path.exists(path) else None

    @property
    def vxa_file(self):
        path = os.path.join(self.data_dir, "base.vxa")
        return path if os.path.exists(path) else None

    def get_comment(self):
        comment = []
        path = os.path.join(self.code_dir, "COMMENT.txt")
        if os.path.exists(path):
            with open(path, "r") as file:
                for line in file.readlines():
                    line = line.strip()
                    if not line.startswith("#"):
                        comment.append(line)
        return comment

    def get_experiment_name(self):
        path = os.path.join(self.code_dir, "LAUNCH_COMMAND.sh")
        if os.path.exists(path):
            with open(path, "r") as file:
                lines = list(file.readlines())
                if lines:
                    start = lines[0].find("experiments/")
                    if start != -1:
                        experiment_name = lines[0][start + len("experiments/") :]
                        experiment_name = experiment_name[: experiment_name.find("/")]
                        return experiment_name
        return ""

    def get_checkpoints(self):
        checkpoints = []
        for file in os.listdir(self.trial_dir):
            if file.startswith("checkpoint_") and os.path.isdir(
                os.path.join(file, self.trial_dir)
            ):
                checkpoints.append(file)
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))
        checkpoints = [os.path.join(self.trial_dir, x) for x in checkpoints]
        return checkpoints

    def get_epoch_files(self):
        epoch_files = {}
        for f in os.listdir(self.data_dir):
            match = re.match(
                r"data_(it_([0-9]+)_rew_([+-]?([0-9]*[.])?[0-9]+))\.data", f
            )
            if match:
                epoch_files[int(match.group(2))] = EpochFiles(
                    f,
                    f"run_{match.group(1)}.history",
                    f"robot_{match.group(1)}.vxd",
                    int(match.group(2)),
                    float(match.group(3)),
                )
        return epoch_files
