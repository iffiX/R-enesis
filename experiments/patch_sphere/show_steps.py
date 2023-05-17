import ray
import numpy as np
from PIL import Image
from experiments.patch_sphere_voxcraft.model import CustomPPO
from renesis.env.voxcraft import (
    VoxcraftSingleRewardTestPatchEnvironment,
    normalize,
)
from experiments.patch_sphere_voxcraft.config import (
    config,
    dimension_size,
    steps,
)
from renesis.utils.plotter import Plotter
from renesis.utils.debug import enable_debugger


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

    algo = CustomPPO(config=config)
    algo.restore(
        "/home/mlw0504/ray_results/CustomPPO_2023-05-09_01-13-04/CustomPPO_VoxcraftSingleRewardTestPatchEnvironment_8ef7f_00000_0_2023-05-09_01-13-04/checkpoint_000400"
    )
    policy = algo.get_policy()
    obs_space = policy.observation_space

    # Create the env to do inference in.
    env_config = config["env_config"].copy()
    env_config["num_envs"] = 1
    env_config["reset_remaining_steps_range"] = (steps, steps)
    env = VoxcraftSingleRewardTestPatchEnvironment(env_config)
    done = False
    obs = env.reset_at(0)

    robot = ""
    plotter = Plotter(interactive=False)
    for i in range(steps):
        # Compute an action (`a`).
        a, state_out, *_ = policy.compute_actions_from_input_dict(
            input_dict={"obs": [obs]},
            explore=True,
        )
        # remove batch dimension
        a = a[0]
        # Send the computed action `a` to the env.
        print(a)
        obs, reward, done, _ = env.vector_step([a])

    # robot = env.env_models[0].voxels
    # for voxel_layer in robot:
    #     print(voxel_layer.astype(np.int))
    # img = plotter.plot_voxel(robot, distance=dimension_size * 3)
    # Image.fromarray(img).save("robot.png")
    algo.stop()

    ray.shutdown()
