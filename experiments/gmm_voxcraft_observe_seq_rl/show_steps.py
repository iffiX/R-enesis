import ray
import numpy as np
import experiments.gmm_voxcraft_observe_seq_rl.model
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from renesis.env.voxcraft import VoxcraftGMMObserveWithVoxelEnvironment, normalize
from experiments.gmm_voxcraft_observe_seq_rl.config import config, dimension, steps
from renesis.utils.plotter import Plotter
from renesis.utils.debug import enable_debugger


def initialize_obs(observation_space):
    return np.array([[observation_space.low] * steps])


def update_obs(obs, new_obs):
    return np.concatenate([obs[:, -(steps - 1) :], np.array([[new_obs]])], axis=1)


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

    algo = PPO(config=config)
    # algo.restore(
    #     "/home/mlw0504/ray_results/PPO_2023-04-06_17-32-12/PPO_VoxcraftGMMObserveSeqEnvironment_e046e_00000_0_2023-04-06_17-32-13/checkpoint_000080"
    # )
    policy = algo.get_policy()
    obs_space = policy.observation_space

    # Create the env to do inference in.
    env_config = config["env_config"].copy()
    env_config["num_envs"] = 1
    env = VoxcraftGMMObserveWithVoxelEnvironment(env_config)
    done = False
    obs = update_obs(initialize_obs(obs_space), env.reset_at(0))

    transformer_attention_size = config["model"]["custom_model_config"]["attention_dim"]
    transformer_length = config["model"]["custom_model_config"]["num_transformer_units"]
    episode_reward = 0
    best_reward = 0
    best_robot = ""
    plotter = Plotter(interactive=False)
    for i in range(20):
        # Compute an action (`a`).
        a, state_out, *_ = policy.compute_actions_from_input_dict(
            input_dict={"obs": obs[:, -1], "custom_obs": obs},
            explore=False,
        )
        # remove batch dimension
        a = a[0]
        # Send the computed action `a` to the env.
        new_obs, reward, done, _ = env.vector_step([a])
        print(env.env_models[0].scale(normalize(a)))
        obs = update_obs(obs, new_obs[0])
        episode_reward += reward[0]

        if episode_reward > best_reward:
            best_robot = env.env_models[0].voxels

    img = plotter.plot_voxel(best_robot, distance=dimension * 3)
    Image.fromarray(img).save("robot.png")
    algo.stop()

    ray.shutdown()
