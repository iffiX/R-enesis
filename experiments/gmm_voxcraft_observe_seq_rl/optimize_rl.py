import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.ppo import PPO
from renesis.env.voxcraft import VoxcraftGMMObserveWithVoxelEnvironment
from experiments.gmm_voxcraft_observe_seq_rl.utils import *
from experiments.gmm_voxcraft_observe_seq_rl.model import *
from renesis.utils.debug import enable_debugger

dimension = 10
iters = 400
steps = 5
workers = 1
envs = 128
rollout = 1

config = {
    "env": VoxcraftGMMObserveWithVoxelEnvironment,
    "env_config": {
        "debug": False,
        "dimension_size": dimension,
        "materials": (0, 1, 2, 3),
        "max_gaussian_num": steps,
        "max_steps": steps,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
        "num_envs": envs,
    },
    "normalize_actions": False,
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 30,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps * envs * rollout,
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
    "evaluation_duration": 1,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": False,
        "explore": True,
        "env_config": {
            "debug": False,
            "dimension_size": dimension,
            "materials": (0, 1, 2, 3),
            "max_gaussian_num": steps,
            "max_steps": steps,
            "reward_type": "distance_traveled",
            "base_config_path": str(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa"
                )
            ),
            "voxel_size": 0.01,
            "fallen_threshold": 0.25,
            "num_envs": envs,
        },
    },
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "num_transformer_units": 1,
            "gaussian_dim": 16,
            "voxel_dim": 112,
            "attention_dim": 128,
            "head_dim": 128,
            "position_wise_mlp_dim": 128,
            "memory": 0,
            "num_heads": 1,
            "dimension_size": dimension,
            "materials": (0, 1, 2, 3),
        },
    },
    "callbacks": CustomCallbacks,
}


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)
    trainer = PPO(config=config)
    tune.run(
        PPO,
        name="",
        config=config,
        checkpoint_freq=5,
        keep_checkpoints_num=10,
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * iters,
            "episodes_total": config["train_batch_size"] * iters / steps,
        },
        # Order is important!
        callbacks=[
            DataLoggerCallback(config["env_config"]["base_config_path"]),
            TBXLoggerCallback(),
        ],
        # restore="",
    )

    ray.shutdown()
