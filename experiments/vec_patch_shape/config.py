from renesis.env.vec_shape import (
    ShapeVectorizedPatchEnvironment,
)
from experiments.vec_patch_shape.utils import *

dimension_size = (20, 20, 20)
materials = (0, 1)
iters = 500
steps = 100


workers = 1
envs = 256
rollout = 1
patch_size = 2

config = {
    "env": ShapeVectorizedPatchEnvironment,
    "env_config": {
        "debug": False,
        "dimension_size": dimension_size,
        "materials": materials,
        "max_patch_num": steps,
        "patch_size": patch_size,
        "max_steps": steps,
        "reward_type": "volume",
        "voxel_size": 0.01,
        "normalize_mode": "clip",
        "num_envs": envs,
    },
    "normalize_actions": False,
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": int(100 * 128 / envs),
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps,
    "vf_clip_param": 10**5,
    "seed": 635326,
    "num_workers": workers,
    "num_gpus": 0.1,
    "num_gpus_per_worker": 0.2,
    "num_envs_per_worker": envs,
    "placement_strategy": "SPREAD",
    "num_cpus_per_worker": 1,
    "framework": "torch",
    "evaluation_interval": None,
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "hidden_dim": 128,
            "max_steps": steps,
            "dimension_size": dimension_size,
            "materials": materials,
            "normalize_mode": "clip",
            "initial_std_bias_in_voxels": 0,
        },
    },
    "callbacks": CustomCallbacks,
}
