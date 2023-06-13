import os
import sys
import time
import shutil
import subprocess
import torch as t
from typing import List, Dict, Any
from launch.config import create_config_modifier
from launch.snapshot import init_config, get_snapshot_comment_file, get_snapshot


def run_single_experiment(
    experiment_py_relative_path: str,
):
    init_config()
    comment_file, comment_dir = get_snapshot_comment_file()
    snapshot_dir = get_snapshot(code_only=False)

    with open(os.path.join(snapshot_dir, "LAUNCH_COMMAND.sh"), "w") as file:
        file.write(f"{sys.executable} {experiment_py_relative_path}")

    # Move comment file to the snapshot dir, so it will be saved by the
    # experiment runners when they call get_snapshot()
    shutil.copy2(comment_file, snapshot_dir)
    shutil.rmtree(comment_dir)

    # Launch python process from the snapshot dir
    command = [sys.executable] + sys.argv[1:]

    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=snapshot_dir,
            env=os.environ.update({"PYTHONPATH": snapshot_dir}),
            start_new_session=True,
        )
        code = process.wait()
        print(f"Launch exited with code {code}")
        if code != 0:
            print(f"Inspect temp code directory {snapshot_dir}")
        else:
            print(f"Removing temp code directory {snapshot_dir}")
            shutil.rmtree(snapshot_dir)
    except KeyboardInterrupt:
        if process is not None:
            print("Keyboard interrupt received, killing instance")
            process.kill()
        print(f"Removing temp code directory {snapshot_dir}")
        shutil.rmtree(snapshot_dir)


def run_multiple_experiments(
    experiment_py_relative_path: str,
    modifier_dicts: List[Dict[str, Any]],
    gpu_numbers: List[int],
    comments: List[str],
):
    assert len(gpu_numbers) == len(modifier_dicts) == len(modifier_dicts)
    if sum(gpu_numbers) > t.cuda.device_count():
        raise ValueError("Total required GPUs exceed device count")
    snapshot_dirs = []
    # Create temporary snapshots for each experiment
    for modifier_dict, comment in zip(modifier_dicts, comments):
        snapshot_dir = get_snapshot(code_only=False)

        with open(os.path.join(snapshot_dir, "LAUNCH_COMMAND.sh"), "w") as file:
            file.write(f"{sys.executable} {experiment_py_relative_path}")

        # Save comment file to the snapshot dir, so it will be saved by the
        # experiment runners when they call get_snapshot()
        with open(os.path.join(snapshot_dir, "COMMENT.txt"), "w") as file:
            file.write(comment)

        create_config_modifier(modifier_dict, snapshot_dir)
        snapshot_dirs.append(snapshot_dir)

    processes = []
    process_output_files = []
    cuda_offset = 0
    try:
        for idx, snapshot_dir in enumerate(snapshot_dirs):
            # Launch python process from the snapshot dir
            command = [sys.executable, experiment_py_relative_path]

            print(f"Starting experiment {idx}")
            process_output_file = open(f"{idx}.out", "w")
            process_output_files.append(process_output_file)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=snapshot_dir,
                    env=os.environ.update(
                        {
                            "PYTHONPATH": snapshot_dir,
                            "CUDA_VISIBLE_DEVICES": ",".join(
                                [
                                    str(dev)
                                    for dev in range(
                                        cuda_offset, cuda_offset + gpu_numbers[idx]
                                    )
                                ]
                            ),
                        }
                    ),
                    start_new_session=True,
                    stdout=process_output_file,
                    stderr=subprocess.STDOUT,
                )
            )
            cuda_offset += gpu_numbers[idx]
            time.sleep(5)
        for idx in range(len(snapshot_dirs)):
            code = processes[idx].wait()
            process_output_files[idx].close()
            print(f"Launch exited with code {code}")
            if code != 0:
                print(f"Inspect temp code directory {snapshot_dirs[idx]}")
            else:
                print(f"Removing temp code directory {snapshot_dirs[idx]}")
                shutil.rmtree(snapshot_dirs[idx])
    except KeyboardInterrupt:
        if len(processes) > 0:
            print("Keyboard interrupt received, killing instances")
            for process in processes:
                process.kill()
            for process_output_file in process_output_files:
                process_output_file.close()
        for snapshot_dir in snapshot_dirs:
            print(f"Removing temp code directory {snapshot_dir}")
            shutil.rmtree(snapshot_dir)
