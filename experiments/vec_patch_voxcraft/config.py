from renesis.env.vec_voxcraft import (
    VoxcraftSingleRewardVectorizedPatchEnvironment,
)
from experiments.vec_patch_voxcraft.utils import *

dimension_size = (20, 20, 20)
materials = (0, 1, 2, 3)
iters = 3000
steps = 40


workers = 7
envs = 2048
rollout = 1
patch_size = 2
train_batch_size = 1024

config = {
    "env": VoxcraftSingleRewardVectorizedPatchEnvironment,
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
    "sgd_minibatch_size": train_batch_size,
    "num_sgd_iter": 10,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps,
    "vf_clip_param": 10**5,
    "seed": 145345,
    "num_workers": workers,
    "num_gpus": 1,
    "num_cpus_for_driver": 16,
    "num_gpus_per_worker": 1,
    "num_envs_per_worker": envs,
    "num_cpus_per_worker": 12,
    "framework": "torch",
    "evaluation_interval": None,
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "hidden_dim": 256,
            "max_steps": steps,
            "dimension_size": dimension_size,
            "materials": materials,
            "normalize_mode": "clip",
            "initial_std_bias_in_voxels": 0,
        },
    },
    "callbacks": CustomCallbacks,
}
