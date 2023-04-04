import ray
from ray.rllib.algorithms.ppo import PPO
from renesis.utils.media import create_video_subproc
from renesis.env.virtual_shape import VirtualShapeGMMEnvironment, normalize
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
workers = 1
envs = 5
rollout = 20
# reference_shape = generate_sphere(dimension)
# reference_shape = generate_3d_shape(
#     10, 200, change_material_when_same_minor_prob=0.2, fill_num=(3,)
# )
# reference_shape = generate_random_ellipsoids(dimension, materials=(1, 2, 3), num=10)
reference_shape = generate_inversed_random_ellipsoids(dimension, material=1, num=10)
# reference_shape = generate_cross(dimension)
config = {
    "env": VirtualShapeGMMEnvironment,
    "env_config": {
        "dimension_size": dimension,
        "materials": (0, 1),
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
    "num_envs_per_worker": envs,
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
    ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9, num_gpus=0)

    algo = PPO(config=config)
    algo.restore("/home/iffi/data2/checkpoint_000380")

    # Create the env to do inference in.
    env = VirtualShapeGMMEnvironment(config["env_config"])
    done = False
    obs = env.reset()

    transformer_attention_size = config["model"]["custom_model_config"].get(
        "attention_dim", 64
    )
    transformer_memory_size = config["model"]["custom_model_config"].get(
        "memory_inference", steps
    )
    transformer_length = config["model"]["custom_model_config"].get(
        "num_transformer_units", 1
    )
    state = [
        np.zeros([transformer_memory_size, transformer_attention_size])
        for _ in range(transformer_length)
    ]
    episode_reward = 0
    plotter = Plotter(interactive=False)
    voxel_inputs = []
    for i in range(20):
        # Compute an action (`a`).
        a, state_out, *_ = algo.compute_single_action(
            observation=obs, state=state, explore=True,
        )
        # Send the computed action `a` to the env.
        obs, reward, done, _ = env.step(a)
        # print(a)
        print(env.env_model.scale(normalize(a)))
        episode_reward += reward

        state = [
            np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
            for i in range(transformer_length)
        ]
        voxel_inputs.append(np.copy(env.env_model.voxels))

    pv.global_theme.window_size = [2048, 768]
    imgs = plotter.plot_voxel_steps(
        reference_shape, voxel_inputs, distance=dimension * 3
    )
    wait = create_video_subproc(
        imgs,
        path="./",
        filename=f"steps_rew={episode_reward}",
        fps=1,
        extension=".mp4",
    )
    wait()
    algo.stop()

    ray.shutdown()
