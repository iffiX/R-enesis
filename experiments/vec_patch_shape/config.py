from ray.rllib.algorithms.ppo import PPOConfig
from renesis.env.vec_shape import (
    VectorizedPatchModel,
    ShapeVectorizedPatchEnvironment,
)
from experiments.vec_patch_shape.utils import *

dimension_size = (20, 20, 20)
materials = (0, 1)
iters = 1000
steps = 100


workers = 1
envs = 1024
rollout = 1
patch_size = 2

example_env_model = VectorizedPatchModel(
    materials=materials,
    dimension_size=dimension_size,
    patch_size=patch_size,
    max_patch_num=steps,
    env_num=envs,
    device="cpu",
)

config = PPOConfig()
config.environment(
    env=ShapeVectorizedPatchEnvironment,
    action_space=example_env_model.action_space,
    observation_space=example_env_model.observation_space,
    env_config={
        "debug": False,
        "dimension_size": dimension_size,
        "materials": materials,
        "max_patch_num": steps,
        "patch_size": patch_size,
        "max_steps": steps,
        "reward_type": "shape_copy_f1",
        # below option is only effective when reward_type=shape_copy_(recall/precision/f1)
        "reward_reference": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "shapes", "dog.npy"
        ),
        "voxel_size": 0.01,
        "normalize_mode": "clip",
        "num_envs": envs,
    },
    render_env=False,
    disable_env_checking=True,
    normalize_actions=False,
)
config.training(
    lr=1e-4,
    gamma=0.99,
    train_batch_size=steps * workers * envs * rollout,
    vf_clip_param=10**5,
    sgd_minibatch_size=128,
    num_sgd_iter=10,
    model={
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
)
config.evaluation(
    evaluation_interval=None,
)
config.debugging(seed=145345)
config.rollouts(
    num_rollout_workers=workers if workers > 1 else 0,
    num_envs_per_worker=envs,
    rollout_fragment_length=steps,
)
config.resources(num_gpus=1, num_cpus_per_worker=12, num_gpus_per_worker=1)
config.framework(framework="torch")
config.callbacks(CustomCallbacks)
