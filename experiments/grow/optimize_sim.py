import os
import ray
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from renesis.env.voxcraft import VoxcraftGrowthEnvironment
from experiments.grow.utils import (
    CustomCallbacks,
    DataLoggerCallback,
    CleaningCallback1,
    CleaningCallback2,
)

"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""

# 1GB heap memory, 1GB object store
ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9)

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1, 2, 3),
        "actuation_features": ("phase_offset",),
        "max_dimension_size": 50,
        "max_view_size": 7,
        "amplitude_range": (0.5, 2),
        "frequency_range": (0.5, 4),
        "phase_offset_range": (0, 1),
        "max_steps": 10,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
        "num_envs": 32,
    },
    "sgd_minibatch_size": 128,
    "train_batch_size": 320,
    "rollout_fragment_length": 10,
    "vf_clip_param": 10 ** 5,
    "seed": np.random.randint(10 ** 5),
    "num_workers": 1,
    "num_gpus": 0.2,
    "num_gpus_per_worker": 0.2,
    "num_envs_per_worker": 32,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {"render_env": False,},
    # "monitor": True,
    # "evaluation_num_workers": 7,
    "model": {
        "custom_model": "actor_model",
        "conv_filters": [
            [16, (8, 8, 8), (4, 4, 4)],
            [32, (4, 4, 4), (2, 2, 2)],
            [256, None, None],
        ],
        "post_fcnet_hiddens": [128, None],
    },
    "callbacks": CustomCallbacks,
}

if __name__ == "__main__":
    tune.run(
        PPOTrainer,
        name="",
        config=config,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        stop={"timesteps_total": 100000, "episodes_total": 10000},
        # Order is important! We want to log videos but not letting
        # loggers automatically added by ray.tune to see it
        callbacks=[
            DataLoggerCallback(),
            CleaningCallback1(),
            TBXLoggerCallback(),
            CleaningCallback2(),
        ]
        # restore=,
    )
