from renesis.env.voxcraft import (
    VoxcraftSingleRewardTestGMMWSObserveWithVoxelAndRemainingStepsEnvironment,
)
from experiments.gmm_ws_voxcraft_observe_seq_rl_pretrain.utils import *

dimension_size = 10
# materials = (2,)
materials = (0, 1, 2, 3)
iters = 200
steps = 20
workers = 1
envs = 128
rollout = 1
sigma = 0.05

pretrain_config = {
    "env_config": {
        "dimension_size": dimension_size,
        "materials": materials,
        "max_gaussian_num": steps,
        "sigma": sigma,
        "reset_seed": 42,
        "reset_remaining_steps_range": (steps, steps),
        "max_steps": steps,
        "reference_shape": None,
        "reward_type": "none",
        "observe_time": True,
    },
    "dataset_path": str(
        os.path.expanduser("~/data/renesis/pretrain/dataset/pretrain.h5")
    ),
    "checkpoint_path": str(os.path.expanduser("~/data/renesis/pretrain/checkpoints")),
    "log_path": str(os.path.expanduser("~/data/renesis/pretrain/logs")),
    "weight_export_path": os.path.expanduser("~/data/renesis/pretrain/result/model.pt"),
    "lr": 1e-4,
    "epochs": 2,
    "seed": 132434,
    "episode_num_for_train": 10000,
    "episode_num_for_validate": 100,
    "dataloader_args": {
        # "num_workers": 4,
        # "prefetch_factor": 512,
        "batch_size": 64,  # equals to pretrain batch size
    },
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "num_transformer_units": 1,
            "attention_dim": 256,
            "head_dim": 256,
            "position_wise_mlp_dim": 128,
            "memory": 0,
            "num_heads": 3,
            "dimension_size": dimension_size,
            "materials": materials,
        },
    },
}

config = {
    "env": VoxcraftSingleRewardTestGMMWSObserveWithVoxelAndRemainingStepsEnvironment,
    "env_config": {
        "debug": False,
        "dimension_size": dimension_size,
        "materials": materials,
        "max_gaussian_num": steps,
        "sigma": sigma,
        "reset_seed": 42,
        "reset_remaining_steps_range": (steps, steps),
        "max_steps": steps,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "normalize_mode": "clip",
        "fallen_threshold": 0.25,
        "num_envs": envs,
        "observe_time": True,
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
    "evaluation_interval": 1,
    "evaluation_duration": envs,
    "evaluation_duration_unit": "episodes",
    "evaluation_parallel_to_training": False,
    "evaluation_num_workers": 0,
    "evaluation_config": {
        "render_env": False,
        "explore": True,
        "env_config": {
            "debug": False,
            "dimension_size": dimension_size,
            "materials": materials,
            "max_gaussian_num": steps,
            "sigma": sigma,
            "reset_seed": 42,
            # For evaluation always allows max steps to be executed
            "reset_remaining_steps_range": (steps, steps),
            "max_steps": steps,
            "reward_type": "distance_traveled",
            "base_config_path": str(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa"
                )
            ),
            "normalize_mode": "clip",
            "voxel_size": 0.01,
            "fallen_threshold": 0.25,
            "num_envs": envs,
            "observe_time": True,
        },
        "num_envs_per_worker": 1,
    },
    "weight_export_path": os.path.expanduser("~/data/renesis/pretrain/result/model.pt"),
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "num_transformer_units": 1,
            "attention_dim": 32,
            "head_dim": 256,
            "position_wise_mlp_dim": 128,
            "memory": 0,
            "num_heads": 3,
            "dimension_size": dimension_size,
            "materials": materials,
        },
    },
    "callbacks": CustomCallbacks,
}
