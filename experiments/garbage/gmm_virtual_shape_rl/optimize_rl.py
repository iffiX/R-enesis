import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.ppo import PPO
from renesis.env.virtual_shape import VirtualShapeGMMEnvironment
from renesis.utils.virtual_shape import (
    generate_3d_shape,
    generate_sphere,
    generate_random_ellipsoids,
    generate_inversed_random_ellipsoids,
    generate_cross,
)
from experiments.gmm_virtual_shape_rl.utils import *

from renesis.utils.debug import enable_debugger

dimension = 10
iters = 400
steps = 20
workers = 20
envs = 5
rollout = 5

# reference_shape = generate_sphere(dimension)
# reference_shape = generate_3d_shape(
#     10, 200, change_material_when_same_minor_prob=0.2, fill_num=(3,)
# )
reference_shape = generate_random_ellipsoids(dimension, materials=(1, 2, 3), num=10)
# reference_shape = generate_inversed_random_ellipsoids(dimension, material=1, num=10)
# reference_shape = generate_cross(dimension)

config = {
    "env": VirtualShapeGMMEnvironment,
    "env_config": {
        "dimension_size": dimension,
        "materials": (0, 1, 2, 3),
        "max_gaussian_num": steps,
        "max_steps": steps,
        "reference_shape": reference_shape,
        "reward_type": "multi_f1",
        "render_config": {"distance": 30},
    },
    "normalize_actions": False,
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 30,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps,
    "vf_clip_param": 10**5,
    "seed": 132434,
    "num_workers": workers,
    "num_gpus": 1,
    "num_envs_per_worker": envs,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_duration": 1,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": False,
        "explore": False,
    },
    "model": {
        "custom_model": "actor_model",
        "max_seq_len": steps,
        "custom_model_config": {
            "num_transformer_units": 1,
            "attention_dim": 32,
            "head_dim": 32,
            "position_wise_mlp_dim": 32,
            "memory_inference": steps,
            "memory_training": steps,
            "num_heads": 1,
        },
    },
    "callbacks": CustomCallbacks,
}


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

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
            DataLoggerCallback(reference_shape, dimension, render=False),
            TBXLoggerCallback(),
        ],
        # restore="",
    )

    ray.shutdown()
