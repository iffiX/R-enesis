import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Make sure c++ modules are compiled
from renesis.sim import *
from launch.run_experiments import run_single_experiment

if __name__ == "__main__":
    """
    Launch any experiment file in a temporary snapshot to prevent
    code from being changed while running.

    python main.py -s experiments/some_some_experiment/some_optimize.py
    python main.py -m experiments/multiple/some_multi_experiment.py
    """
    parser = argparse.ArgumentParser(description="Script for running experiments.")

    # Add the necessary arguments
    parser.add_argument(
        "-s", "--single", metavar="PYTHON FILE", help="Run a single experiment"
    )
    parser.add_argument(
        "-m", "--multiple", metavar="PYTHON FILE", help="Run multiple experiments"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check which argument was provided and execute the corresponding code
    if args.single:
        print(f"Running single experiment: {args.single}")
        run_single_experiment(args.single)

    if args.multiple:
        print(f"Running multiple experiments: {args.multiple}")
        with open(args.multiple, "r") as script:
            exec(script.read())
