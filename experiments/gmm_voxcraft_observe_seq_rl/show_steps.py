import ray
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from renesis.utils.media import create_video_subproc
from renesis.env.voxcraft import VoxcraftGMMObserveSeqEnvironment, normalize
from experiments.gmm_voxcraft_observe_seq_rl.utils import *
from experiments.gmm_voxcraft_observe_seq_rl.model import *
from renesis.utils.plotter import Plotter
from renesis.utils.debug import enable_debugger

dimension = 10
iters = 400
steps = 20
workers = 1
envs = 128
rollout = 1

config = {
    "env": VoxcraftGMMObserveSeqEnvironment,
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
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 30,
    "train_batch_size": steps * workers * envs * rollout,
    "lr": 1e-4,
    "rollout_fragment_length": steps * envs * rollout,
    "vf_clip_param": 10**5,
    "seed": 132434,
    "num_workers": workers,
    "num_gpus": 0.1,
    "num_gpus_per_worker": 0.1,
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
            "materials": (0, 1, 2, 3),
        },
    },
    "callbacks": CustomCallbacks,
}


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

    algo = PPO(config=config)
    algo.restore(
        "/home/mlw0504/ray_results/PPO_2023-04-06_17-32-12/PPO_VoxcraftGMMObserveSeqEnvironment_e046e_00000_0_2023-04-06_17-32-13/checkpoint_000080"
    )

    # Create the env to do inference in.
    env_config = config["env_config"].copy()
    env_config["num_envs"] = 1
    env = VoxcraftGMMObserveSeqEnvironment(env_config)
    done = False
    obs = env.reset_at(0)

    transformer_attention_size = config["model"]["custom_model_config"].get(
        "attention_dim", 64
    )
    transformer_length = config["model"]["custom_model_config"].get(
        "num_transformer_units", 1
    )
    episode_reward = 0
    best_reward = 0
    best_robot = ""
    plotter = Plotter(interactive=False)
    for i in range(20):
        # Compute an action (`a`).
        a, state_out, *_ = algo.compute_single_action(
            observation=obs,
            explore=True,
        )
        # Send the computed action `a` to the env.
        obs, reward, done, _ = env.vector_step([a])
        print(env.env_model.scale(normalize(a)))
        episode_reward += reward
        if episode_reward > best_reward:
            best_robot = env.env_models[0].voxels

    pv.global_theme.window_size = [2048, 768]
    img = plotter.plot_voxel(best_robot, distance=dimension * 3)
    algo.stop()

    ray.shutdown()
