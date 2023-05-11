import ray
import numpy as np
from PIL import Image
from experiments.gmm_ws_voxcraft_observe_seq_rl_pretrain.model import CustomPPO
from renesis.env.voxcraft import (
    VoxcraftSingleRewardGMMWSObserveWithVoxelAndRemainingStepsEnvironment,
    normalize,
)
from experiments.gmm_ws_voxcraft_observe_seq_rl_pretrain.config import (
    config,
    dimension_size,
    steps,
)
from renesis.utils.plotter import Plotter
from renesis.utils.debug import enable_debugger


def initialize_obs(observation_space):
    return np.array([[observation_space.low] * steps])


def update_obs(obs, new_obs):
    return np.concatenate([obs[:, -(steps - 1) :], np.array([[new_obs]])], axis=1)


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

    algo = CustomPPO(config=config)
    algo.restore(
        "/home/mlw0504/ray_results/CustomPPO_2023-05-03_00-41-54/CustomPPO_VoxcraftSingleRewardGMMWSObserveWithVoxelAndRemainingStepsEnvironment_35f68_00000_0_2023-05-03_00-41-54/checkpoint_000200"
    )
    policy = algo.get_policy()
    obs_space = policy.observation_space

    # Create the env to do inference in.
    env_config = config["env_config"].copy()
    env_config["num_envs"] = 1
    env_config["reset_remaining_steps_range"] = (steps, steps)
    env = VoxcraftSingleRewardGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
        env_config
    )
    done = False
    obs = update_obs(initialize_obs(obs_space), env.reset_at(0))

    transformer_attention_size = config["model"]["custom_model_config"]["attention_dim"]
    transformer_length = config["model"]["custom_model_config"]["num_transformer_units"]
    robot = ""
    plotter = Plotter(interactive=False)
    for i in range(20):
        # Compute an action (`a`).
        a, state_out, *_ = policy.compute_actions_from_input_dict(
            input_dict={"obs": obs[:, -1], "custom_obs": obs},
            explore=False if i > 20 else True,
        )
        # remove batch dimension
        a = a[0]
        # Send the computed action `a` to the env.
        print(a)
        new_obs, reward, done, _ = env.vector_step([a])
        obs = update_obs(obs, new_obs[0])

    robot = env.env_models[0].voxels
    for voxel_layer in robot:
        print(voxel_layer.astype(np.int))
    # img = plotter.plot_voxel(robot, distance=dimension_size * 3)
    # Image.fromarray(img).save("robot.png")
    algo.stop()

    ray.shutdown()
