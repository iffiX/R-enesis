from renesis.env.voxcraft import (
    VoxcraftSingleRewardPatchSphereEnvironment,
)
from experiments.patch_sphere_voxcraft.utils import *

dimension_size = (10, 10, 10)
materials = (0, 1, 2, 3)
iters = 3000
steps = 40


workers = 1
envs = 128
rollout = 1
patch_size = 3

config = {
    "env": VoxcraftSingleRewardPatchSphereEnvironment,
    "env_config": {
        "debug": False,
        "dimension_size": dimension_size,
        "materials": materials,
        "max_patch_num": steps,
        "patch_size": patch_size,
        "max_steps": steps,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "normalize_mode": "clip",
        "fallen_threshold": 0.25,
        "num_envs": envs,
    },
    "normalize_actions": False,
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": envs,
    "num_sgd_iter": 10,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps,
    "vf_clip_param": 10**5,
    "seed": 132434,
    "num_workers": workers,
    "num_gpus": 0.1,
    "num_gpus_per_worker": 0.2,
    "num_envs_per_worker": envs,
    "placement_strategy": "SPREAD",
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": None,
    # "evaluation_interval": 1,
    # "evaluation_duration": envs,
    # "evaluation_duration_unit": "episodes",
    # "evaluation_parallel_to_training": False,
    # "evaluation_num_workers": 0,
    # "evaluation_config": {
    #     "render_env": False,
    #     "explore": True,
    #     "env_config": {
    #         "debug": False,
    #         "dimension_size": dimension_size,
    #         "materials": materials,
    #         "max_patch_num": steps,
    #         "patch_size": patch_size,
    #         "max_steps": steps,
    #         "reward_type": "distance_traveled",
    #         "base_config_path": str(
    #             os.path.join(
    #                 os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa"
    #             )
    #         ),
    #         "voxel_size": 0.01,
    #         "normalize_mode": "clip",
    #         "fallen_threshold": 0.25,
    #         "num_envs": envs,
    #     },
    #     "num_envs_per_worker": 1,
    # },
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "hidden_dim": 256,
            "max_steps": steps,
            "dimension_size": dimension_size,
            "materials": materials,
            "normalize_mode": "clip",
            "initial_std_bias_in_voxels": max(max(dimension_size) // 10, 2),
        },
    },
    "callbacks": CustomCallbacks,
}
