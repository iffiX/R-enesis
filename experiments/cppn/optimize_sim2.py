import os
import ray
import argparse
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from renesis.env_model.cppn import CPPNModel
from renesis.env.voxcraft import VoxcraftCPPNEnvironment
from experiments.cppn.utils import CustomCallbacks, DataLoggerCallback

from renesis.utils.debug import enable_debugger

"""
IMPORTANT: You MUST configure data/base.vxa to match the relevant
configurations in this file.
"""

# 1GB heap memory, 1GB object store
ray.init(_memory=1 * (10 ** 9), object_store_memory=10 ** 9)

config = {
    "env": VoxcraftCPPNEnvironment,
    "env_config": {
        "debug": False,
        "materials": (0, 1, 2, 3, 4),
        "dimension_size": 10,
        "actuation_features": (),
        "amplitude_range": (0.5, 2),
        "frequency_range": (0.5, 4),
        "phase_offset_range": (0, 1),
        "max_steps": 20,
        "reward_type": "distance_traveled",
        "base_config_path": str(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "base2.vxa"
            )
        ),
        "voxel_size": 0.01,
        "fallen_threshold": 0.25,
        "num_envs": 32,
    },
    "sgd_minibatch_size": 128,
    "train_batch_size": 640,
    "rollout_fragment_length": 20,
    "vf_clip_param": 10 ** 5,
    "seed": np.random.randint(10 ** 5),
    "num_workers": 1,
    "num_gpus": 0.1,
    "num_gpus_per_worker": 0.1,
    "num_envs_per_worker": 32,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_interval": 1,
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {"render_env": False,},
    # "monitor": True,
    # "evaluation_num_workers": 7,
    "model": {
        "custom_action_dist": "actor_dist",
        "custom_model": "actor_model",
        "custom_model_config": {
            "debug": False,
            "input_feature_num": 3 + len(CPPNModel.DEFAULT_CPPN_FUNCTIONS),
            "hidden_feature_num": 128,
            "output_feature_num": 128,
            "layer_num": 4,
            "head_num": 3,
            "cppn_input_node_num": 4,
            # len(materials) + len(actuation_features), computed later
            "cppn_output_node_num": -1,
            "target_function_num": len(CPPNModel.DEFAULT_CPPN_FUNCTIONS),
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

    config["model"]["custom_model_config"]["cppn_output_node_num"] = len(
        config["env_config"]["materials"]
    ) + len(config["env_config"]["actuation_features"])

    tune.run(
        # DebugPPOTrainer,
        PPOTrainer,
        name="",
        config=config,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        stop={"timesteps_total": 100000, "episodes_total": 10000},
        # Order is important!
        callbacks=[DataLoggerCallback(), TBXLoggerCallback()]
        # restore=,
    )
