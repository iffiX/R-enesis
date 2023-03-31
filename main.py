import os
import sys
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Make sure c++ modules are compiled
from renesis.sim import *

from launch.snapshot import init_config, get_snapshot_comment_file, get_snapshot

if __name__ == "__main__":
    """
    Launch any experiment file in a temporary snapshot to prevent
    code from being changed while running.
    
    python main.py experiments/some_some_experiment/some_optimize.py some_args
    """
    init_config()
    comment_file, comment_dir = get_snapshot_comment_file()
    snapshot_dir = get_snapshot(code_only=False)

    # Move comment file to the snapshot dir, so it will be saved by the
    # experiment runners when they call get_snapshot()
    shutil.copy2(comment_file, snapshot_dir)
    shutil.rmtree(comment_dir)

    # Launch python process from the snapshot dir
    command = [sys.executable] + sys.argv[1:]
    process = subprocess.Popen(
        command, cwd=snapshot_dir, env=os.environ.update({"PYTHONPATH": snapshot_dir})
    )
    code = process.wait()
    print(f"Launch exited with code {code}")
    if code != 0:
        print(f"Inspect temp code directory {snapshot_dir}")
    else:
        print(f"Removing temp code directory {snapshot_dir}")
        shutil.rmtree(snapshot_dir)
