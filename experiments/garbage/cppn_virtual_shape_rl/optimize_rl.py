import os
import ray
import argparse
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from renesis.env_model.cppn import CPPNBaseModel
from renesis.env.virtual_shape import VirtualShapeCPPNEnvironment
from renesis.utils.virtual_shape import generate_3d_shape
from experiments.cppn_virtual_shape_rl.utils import *

# from experiments.cppn_virtual_shape_rl.utils import (
#     # CustomCallbacks,
#     # DataLoggerCallback,
#     # ActorSampling
# )

from renesis.utils.debug import enable_debugger

# 1GB heap memory, 1GB object store
ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9)

reference_shape = generate_3d_shape(10, 100)
# vector_env_num_per_worker = 5
config = {
    "env": VirtualShapeCPPNEnvironment,
    "env_config": {
        "dimension_size": 10,
        "cppn_hidden_node_num": 30,
        "max_steps": 100,
        "reference_shape": reference_shape,
        "reward_type": "correct_rate",
        "render_config": {"distance": 30},
    },
    "disable_env_checking": True,
    "render_env": False,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 15,
    "train_batch_size": 12800,
    "lr": 1e-4,
    "rollout_fragment_length": 100,
    "vf_clip_param": 10 ** 5,
    "seed": np.random.randint(10 ** 5),
    "num_workers": 8,
    "num_gpus": 1,
    "num_envs_per_worker": 16,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_duration": 10,
    "evaluation_num_workers": 1,
    "model": {
        "custom_action_dist": "actor_dist",
        "custom_model": "actor_model",
        "custom_model_config": {
            "debug": False,
            "input_feature_num": 4 + len(CPPNBaseModel.DEFAULT_CPPN_FUNCTIONS),
            "hidden_feature_num": 64,
            "output_feature_num": 64,
            "layer_num": 3,
            "head_num": 2,
            "cppn_input_node_num": 4,
            "cppn_output_node_num": 3,
            "target_function_num": len(CPPNBaseModel.DEFAULT_CPPN_FUNCTIONS),
            "dropout_prob": 0.3
            # "initial_temperature": 5,
            # "exploration_timesteps": -1,
        },
    },
    "evaluation_config": {"model": {"custom_model_config": {"dropout_prob": 0}}}
    # "exploration_config": {"type": ActorSampling},
    # "callbacks": CustomCallbacks,
}

# config["model"]["custom_model_config"]["exploration_timesteps"] = (
#     config["train_batch_size"] * 3
# )


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
        checkpoint_freq=5,
        keep_checkpoints_num=2,
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * 100,
            "episodes_total": config["train_batch_size"] * 100 / 100,
        },
        # Order is important!
        # callbacks=[DataLoggerCallback(), TBXLoggerCallback()]
        # restore=,
    )
