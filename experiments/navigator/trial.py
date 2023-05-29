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
