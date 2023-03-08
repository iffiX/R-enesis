import os
import ray
import argparse
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from renesis.env_model.cppn import CPPNBaseModel
from renesis.env.voxcraft import VoxcraftCPPNBinaryTreeEnvironment
from experiments.cppn.utils import CustomCallbacks, DataLoggerCallback

from renesis.utils.debug import enable_debugger

"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""

# 1GB heap memory, 1GB object store
ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9)

# vector_env_num_per_worker = 5
config = {
    "env": VoxcraftCPPNBinaryTreeEnvironment,
    "env_config": {
        "debug": False,
        "dimension_size": 6,
        "cppn_hidden_node_num": 10,
        "max_steps": 40,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa")
        ),
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
        "num_envs": 16,  # vector_env_num_per_worker,
    },
    "sgd_minibatch_size": 32,  # vector_env_num_per_worker * 2,
    "train_batch_size": 640,  # 40 * vector_env_num_per_worker * 2,
    "lr": 1e-3,
    "rollout_fragment_length": 5,
    "vf_clip_param": 10 ** 5,
    "seed": np.random.randint(10 ** 5),
    # "num_workers": 2,
    # "num_gpus": 0.1,
    # "num_gpus_per_worker": 0.5,
    # "num_envs_per_worker": vector_env_num_per_worker,
    "num_workers": 1,
    "num_gpus": 0.1,
    "num_gpus_per_worker": 0.1,
    "num_envs_per_worker": 16,  # vector_env_num_per_worker,
    "placement_strategy": "SPREAD",
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_duration": 2,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {"render_env": False, "explore": True},
    # "monitor": True,
    "model": {
        "custom_action_dist": "actor_dist",
        "custom_model": "actor_model",
        "custom_model_config": {
            "debug": False,
            "input_feature_num": 3 + len(CPPNBaseModel.DEFAULT_CPPN_FUNCTIONS),
            "hidden_feature_num": 32,
            "output_feature_num": 32,
            "layer_num": 3,
            "head_num": 2,
            "cppn_input_node_num": 4,
            "cppn_output_node_num": 3,
            "target_function_num": len(CPPNBaseModel.DEFAULT_CPPN_FUNCTIONS),
        },
    },
    "callbacks": CustomCallbacks,
}


class DebugPPOTrainer(PPOTrainer):
    def training_step(self):
        if self.config["model"]["custom_model_config"]["debug"]:
            enable_debugger(
                self.config["model"]["custom_model_config"]["debug_ip"],
                self.config["model"]["custom_model_config"]["debug_port"],
            )
        return super(DebugPPOTrainer, self).training_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="Attach to a remote PyCharm debug server", action="store_true"
    )
    parser.add_argument(
        "--debug_ip", help="PyCharm debug server IP", type=str, default="localhost"
    )
    parser.add_argument(
        "--debug_port", help="PyCharm debug server port", type=int, default=8223
    )

    args = parser.parse_args()
    if args.debug:
        config["env_config"]["debug"] = True
        config["env_config"]["debug_ip"] = args.debug_ip
        config["env_config"]["debug_port"] = args.debug_port
        # config["model"]["custom_model_config"]["debug"] = True
        # config["model"]["custom_model_config"]["debug_ip"] = args.debug_ip
        # config["model"]["custom_model_config"]["debug_port"] = args.debug_port

    tune.run(
        # DebugPPOTrainer,
        PPOTrainer,
        name="",
        config=config,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        stop={
            "timesteps_total": config["train_batch_size"] * 100,
            "episodes_total": config["train_batch_size"] * 100 / 40,
        },
        # Order is important!
        callbacks=[DataLoggerCallback(), TBXLoggerCallback()]
        # restore=,
    )
