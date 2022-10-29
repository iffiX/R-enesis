import os
import ray
import numpy as np
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from renesis.env.voxcraft import VoxcraftGrowthEnvironment
from experiments.grow.utils import CustomCallbacks, DataLoggerCallback

"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""

ray.init()

config = {
    "env": VoxcraftGrowthEnvironment,
    "env_config": {
        "materials": (0, 1),
        "max_dimension_size": 50,
        "max_view_size": 21,
        "amplitude_range": (0, 2),
        "frequency_range": (0, 4),
        "phase_shift_range": (0, 1),
        "max_steps": 10,
        "reward_interval": 1,
        "reward_type": "distance_traveled",
        "base_template_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
    },
    "sgd_minibatch_size": 4,
    "train_batch_size": 20,
    "vf_clip_param": 10 ** 5,
    "seed": np.random.randint(10 ** 5),
    "num_workers": 2,
    "num_gpus": 0.2,
    "num_gpus_per_worker": 0.2,
    "num_envs_per_worker": 1,
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
        callbacks=[DataLoggerCallback()]
        # restore=,
    )
