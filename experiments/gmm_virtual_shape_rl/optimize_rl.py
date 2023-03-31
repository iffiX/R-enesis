import os
import ray
import argparse
import numpy as np
from PIL import Image
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO
from renesis.utils.plotter import Plotter
from renesis.env.virtual_shape import VirtualShapeGMMEnvironment, sigmoid
from experiments.cppn_virtual_shape_evolution.utils import (
    generate_3d_shape,
    generate_sphere,
    generate_random_ellipsoids,
    generate_cross,
)
from experiments.gmm_virtual_shape_rl.utils import *

from renesis.utils.debug import enable_debugger

dimension = 10
iters = 400
steps = 20
workers = 10
envs = 5
rollout = 20
# reference_shape = generate_sphere(dimension)
# reference_shape = generate_3d_shape(
#     10, 200, change_material_when_same_minor_prob=0.2, fill_num=(3,)
# )
reference_shape = generate_random_ellipsoids(dimension, materials=(1, 2, 3), num=10)
# reference_shape = generate_cross(dimension)
# plotter = Plotter(interactive=True)
# plotter.plot_voxel(reference_shape, distance=dimension * 3)
# vector_env_num_per_worker = 5
config = {
    "env": VirtualShapeGMMEnvironment,
    "env_config": {
        "dimension_size": dimension,
        "materials": (0, 1, 2, 3),
        "max_gaussian_num": 100,
        "max_steps": steps,
        "reference_shape": reference_shape,
        "reward_type": "f1",
        "render_config": {"distance": 30},
    },
    "normalize_actions": False,
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 30,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps * envs * rollout,
    "vf_clip_param": 10 ** 5,
    "seed": 132434,
    "num_workers": workers,
    "num_gpus": 1,
    "num_envs_per_worker": 1,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_duration": 10,
    "evaluation_num_workers": 1,
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
    ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9)

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
        # restore="/home/iffi/ray_results/PPO_2023-03-27_22-56-57/PPO_VirtualShapeGMMEnvironment_95aee_00000_0_2023-03-27_22-56-57/checkpoint_000100/checkpoint-100",
    )

    # algo = PPO(config=config)
    # algo.restore(
    #     "/home/iffi/ray_results/PPO_2023-03-26_13-49-44/PPO_VirtualShapeGMMEnvironment_f9823_00000_0_2023-03-26_13-49-44/checkpoint_000100"
    # )
    #
    # # Create the env to do inference in.
    # env = VirtualShapeGMMEnvironment(config["env_config"])
    # done = False
    # obs = env.reset()
    #
    # transformer_attention_size = config["model"]["custom_model_config"].get(
    #     "attention_dim", 64
    # )
    # transformer_memory_size = config["model"]["custom_model_config"].get(
    #     "memory_inference", 50
    # )
    # transformer_length = config["model"]["custom_model_config"].get(
    #     "num_transformer_units", 1
    # )
    # state = [
    #     np.zeros([transformer_memory_size, transformer_attention_size])
    #     for _ in range(transformer_length)
    # ]
    # episode_reward = 0
    #
    # for i in range(20):
    #     # Compute an action (`a`).
    #     a, state_out, *_ = algo.compute_single_action(
    #         observation=obs, state=state, explore=True,
    #     )
    #     # Send the computed action `a` to the env.
    #     obs, reward, done, _ = env.step(a)
    #     # print(a)
    #     print(env.env_model.scale(sigmoid(a)))
    #     episode_reward += reward
    #
    #     state = [
    #         np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
    #         for i in range(transformer_length)
    #     ]
    #
    # plotter = Plotter(interactive=True)
    # plotter.plot_voxel_error(
    #     reference_shape, env.env_model.voxels, distance=dimension * 3
    # )
    # # img = plotter.plot_voxel_error(
    # #     reference_shape, env.env_model.voxels, distance=dimension * 3
    # # )
    # # Image.fromarray(img, mode="RGB").save(f"error.png")
    # print(episode_reward)
    # algo.stop()

    ray.shutdown()
