import os
import re
import pickle
import numpy as np
from functools import partial
from typing import List, Callable
from experiments.navigator.trial import TrialRecord, EpochFiles
from experiments.navigator.prompter import (
    Int,
    PromptApp,
    PromptChoiceDialog,
    PromptExecutable,
    PromptExecutableWithInput,
    PromptExecutableWithMultipleChoice,
)
from experiments.navigator.functions.draw_generation_process import (
    draw_generation_process,
)
from experiments.navigator.functions.draw_robots import draw_robots
from experiments.navigator.functions.draw_reward_curves import draw_reward_curves
from experiments.navigator.functions.compute_robot_metrics import compute_robot_metrics
from experiments.navigator.functions.visualize_robot import (
    visualize_robot,
    visualize_selected_robot,
)

root_dirs_of_trials = [
    "/home/mlw0504/ray_results",
    "/home/mlw0504/ray_results_vss5",
    "/home/mlw0504/ray_results_quest",
    "/home/mlw0504/Projects/R-enesis_results/task_voxcraft_20x20x20_T=100",
]


def find_directories(
    root_dir_of_trials: str, trial_filter: Callable[[str], bool] = None
):
    trial_dirs = []

    source = os.listdir(root_dir_of_trials)
    source.sort(key=lambda x: os.path.getmtime(os.path.join(root_dir_of_trials, x)))
    for root_dir_of_trial in source:
        try:
            if os.path.isdir(os.path.join(root_dir_of_trials, root_dir_of_trial)) and (
                not trial_filter or trial_filter(root_dir_of_trial)
            ):
                sub_dir = [
                    sdir
                    for sdir in os.listdir(
                        os.path.join(root_dir_of_trials, root_dir_of_trial)
                    )
                    if os.path.isdir(
                        os.path.join(root_dir_of_trials, root_dir_of_trial, sdir)
                    )
                ][0]
                trial_dirs.append(
                    os.path.join(root_dir_of_trials, root_dir_of_trial, sub_dir)
                )
        except:
            continue
    return trial_dirs


def empty_func():
    raise NotImplementedError()


if __name__ == "__main__":
    all_trial_dirs = []
    for root_dir in root_dirs_of_trials:
        all_trial_dirs += find_directories(root_dir)

    all_trial_records = []
    for trial_dir in all_trial_dirs:
        try:
            record = TrialRecord(trial_dir)
            all_trial_records.append(record)
        except:
            continue

    app = PromptApp(
        PromptChoiceDialog(
            description="",
            choices=[
                PromptExecutableWithMultipleChoice(
                    description="Draw metrics for multiple trials",
                    execute=draw_reward_curves,
                    choices=[
                        (
                            f"{trial_record.trial_dir}\n"
                            f"    comment: {' '.join(trial_record.comment)}\n"
                            f"    reward: {trial_record.max_reward:.3f}",
                            trial_record,
                        )
                        for trial_record in all_trial_records
                    ],
                ),
                PromptChoiceDialog(
                    description="Draw metrics for single trial",
                    choices=[
                        PromptChoiceDialog(
                            description=f"{trial_record.trial_dir}\n"
                            f"    comment: {' '.join(trial_record.comment)}\n"
                            f"    reward: {trial_record.max_reward:.3f}",
                            choices=[
                                PromptExecutableWithInput(
                                    "draw generation process figure for best robot from epoch...",
                                    partial(draw_generation_process, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch)",
                                    input_formats=[Int()],
                                ),
                                PromptExecutableWithInput(
                                    "draw robots from epoch...",
                                    partial(draw_robots, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch) and show how many robots?",
                                    input_formats=[Int(), Int(optional=True)],
                                ),
                                PromptExecutableWithInput(
                                    "show metrics of best robot from epoch...",
                                    partial(compute_robot_metrics, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch)",
                                    input_formats=[Int()],
                                ),
                                PromptExecutableWithInput(
                                    "visualize history of best robot from epoch...",
                                    partial(visualize_robot, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch) and show how many robots?",
                                    input_formats=[Int()],
                                ),
                                PromptExecutableWithInput(
                                    "visualize selected robot from epoch...",
                                    partial(visualize_selected_robot, trial_record),
                                    prompt_input=f"show which epoch, from 1 to {trial_record.epochs[-1]}? "
                                    f"(-1 for best epoch) and show how many robots?",
                                    input_formats=[Int()],
                                ),
                            ],
                        )
                        for trial_record in all_trial_records
                    ],
                    prompt_title="Trials",
                ),
            ],
        )
    )
    app.run()
