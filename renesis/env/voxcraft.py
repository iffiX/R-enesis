import os
import copy
import numpy as np
from time import time
from typing import List, Dict, Any, Optional
from ray.rllib import VectorEnv
from ray.rllib.utils import override
from renesis.sim import Voxcraft
from renesis.utils.debug import enable_debugger
from renesis.utils.voxcraft import vxd_creator, get_voxel_positions
from renesis.utils.fitness import max_z, table, distance_traveled, has_fallen
from renesis.utils.debug import enable_debugger
from renesis.env_model.base import BaseModel
from renesis.env_model.cppn import (
    CPPNBinaryTreeModel,
    CPPNBinaryTreeWithPhaseOffsetModel,
)
from renesis.env_model.growth import GrowthModel


class VoxcraftBaseEnvironment(VectorEnv):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], env_models: List[BaseModel]):
        self.env_models = env_models
        self.max_steps = config["max_steps"]
        self.action_space = env_models[0].action_space
        self.observation_space = env_models[0].observation_space
        self.reward_range = (0, float("inf"))
        with open(config["base_config_path"], "r") as file:
            self.base_config = file.read()
        self.reward_type = config["reward_type"]
        self.voxel_size = config["voxel_size"]
        self.fallen_threshold = config["fallen_threshold"]

        self.previous_rewards = [0 for _ in range(config["num_envs"])]
        self.robots = ["\n" for _ in range(config["num_envs"])]
        self.robot_sim_histories = ["" for _ in range(config["num_envs"])]
        self.state_data = [None for _ in range(config["num_envs"])]

        self.all_rewards_history = []
        self.best_reward = -np.inf
        self.best_finished_robot = "\n"
        self.best_finished_robot_sim_history = ""
        self.best_finished_robot_state_data = None

        self.simulator = Voxcraft()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None):
        if index is None:
            index = 0
        self.env_models[index].reset()
        self.previous_rewards[index] = 0
        self.robots[index] = "\n"
        self.robot_sim_histories[index] = ""
        self.state_data[index] = None
        return self.env_models[index].observe()

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        for model in self.env_models:
            model.reset()

        self.previous_rewards = [0 for _ in range(self.num_envs)]
        self.robots = ["\n" for _ in range(self.num_envs)]
        self.robot_sim_histories = ["" for _ in range(self.num_envs)]
        self.state_data = [None for _ in range(self.num_envs)]
        return [model.observe() for model in self.env_models]

    @override(VectorEnv)
    def vector_step(self, actions):
        all_finished = self.check_finished()
        for model, action, finished in zip(self.env_models, actions, all_finished):
            if not finished:
                model.step(action)

        rewards = self.get_rewards(all_finished)
        # print(f"Rewards: {rewards}")

        # for i, finish in enumerate(self.check_finished()):
        #     if finish:
        #         if rewards[i] > self.best_reward:
        #             self.best_reward = rewards[i]
        #             self.best_finished_robot = self.robots[i]
        #             self.best_finished_robot_sim_history = self.robot_sim_histories[i]
        #             self.best_finished_robot_state_data = self.state_data[i]

        for i in range(self.num_envs):
            if rewards[i] > self.best_reward:
                self.best_reward = rewards[i]
                self.best_finished_robot = self.robots[i]
                self.best_finished_robot_sim_history = self.robot_sim_histories[i]
                self.best_finished_robot_state_data = self.state_data[i]

        reward_diffs = [
            reward - previous_reward
            for reward, previous_reward in zip(rewards, self.previous_rewards)
        ]
        self.previous_rewards = rewards

        self.all_rewards_history.append(rewards)
        if len(self.all_rewards_history) > self.max_steps:
            self.all_rewards_history = self.all_rewards_history[-self.max_steps :]
        # print(f"Actions: \n {actions}")
        # print(f"Rewards: {self.previous_rewards}")
        # print(f"Finished: {self.check_finished()}")
        return (
            [model.observe() for model in self.env_models],
            reward_diffs,
            self.check_finished(),
            [{} for _ in range(self.num_envs)],
        )

    def get_rewards(self, all_finished):
        """
        Returns the list of rewards of all sub environments.

        For sub environments that are not finished in current step, i.e. has executed
        a step, their reward are updated. Otherwise the old reward is returned.

        Args:
            all_finished: A list of bool values indicating whether current sub
                environment is finished.

        Returns:
            A list of float reward value for all sub environments.
        """
        rewards = copy.deepcopy(self.previous_rewards)
        valid_models = []
        valid_model_indices = []
        for idx, (model, finished) in enumerate(zip(self.env_models, all_finished)):
            if not model.is_robot_empty() and not finished:
                valid_models.append(model)
                valid_model_indices.append(idx)

        robots, (results, records) = self.run_simulations(valid_models)
        for robot, result, record, model, idx in zip(
            robots, results, records, valid_models, valid_model_indices
        ):
            initial_positions, final_positions = get_voxel_positions(
                result, voxel_size=self.voxel_size
            )
            reward = self.compute_reward_from_sim_result(
                initial_positions, final_positions
            )
            rewards[idx] = reward

            self.robots[idx] = robot
            self.robot_sim_histories[idx] = record
            self.state_data[idx] = model.get_state_data()
        return rewards

    def run_simulations(self, env_models):
        """
        Run a simulation using current representation.

        Returns:
            Path to the output.xml file.
            Path to the temporary directory (needs to be deleted).
        """
        robots = []
        for model in env_models:
            sizes, representation = model.get_robot()
            robots.append(vxd_creator(sizes, representation, record_history=True))

        begin = time()
        for attempt in range(3):
            try:
                out = self.simulator.run_sims([self.base_config] * len(robots), robots)
                end = time()
            except Exception as e:
                print(f"Failed attempt {attempt + 1}")
                if attempt == 2:
                    print(f"Final attempt failed")
                    dump_dir = os.path.expanduser(f"~/renesis_sim_dump/{begin}")
                    os.makedirs(dump_dir, exist_ok=True)
                    print(f"Debug info saved to {dump_dir}")
                    with open(os.path.join(dump_dir, "base.vxa"), "w") as file:
                        file.write(self.base_config)
                    for i, robot in enumerate(robots):
                        with open(os.path.join(dump_dir, f"{i}.vxd"), "w") as file:
                            file.write(robot)
                    raise e
            else:
                break
        print(
            f"{len(robots)} simulations total {end - begin:.3f}s, "
            f"average {(end - begin) / len(robots):.3f}s"
        )
        return robots, out

    def compute_reward_from_sim_result(self, initial_positions, final_positions):
        if self.reward_type == "max_z":
            if has_fallen(initial_positions, final_positions, self.fallen_threshold):
                reward = 0
            else:
                reward = max_z(final_positions)
        elif self.reward_type == "table":
            if has_fallen(initial_positions, final_positions, self.fallen_threshold):
                reward = 0
            else:
                reward = table(final_positions)
        elif self.reward_type == "distance_traveled":
            reward = distance_traveled(initial_positions, final_positions)
            if reward < 1e-3:
                reward = 0
        else:
            raise Exception("Unknown reward type: {self.reward_type}")
        if np.isnan(reward):
            reward = 0
        return reward

    def check_finished(self):
        return [
            model.is_finished() or (model.steps == self.max_steps)
            for model in self.env_models
        ]

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot[0] + "\n"


class VoxcraftGrowthEnvironment(VoxcraftBaseEnvironment):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(config["debug_ip"], config["debug_port"])
        env_models = [
            GrowthModel(
                materials=config["materials"],
                max_dimension_size=config["max_dimension_size"],
                max_view_size=config["max_view_size"],
                actuation_features=config["actuation_features"],
                amplitude_range=config["amplitude_range"],
                frequency_range=config["frequency_range"],
                phase_offset_range=config["phase_offset_range"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)


class VoxcraftCPPNBinaryTreeEnvironment(VoxcraftBaseEnvironment):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        self.initial_max_random_steps = config.get("initial_max_random_steps", 0)
        if config.get("debug", False):
            enable_debugger(config["debug_ip"], config["debug_port"])
        env_models = [
            CPPNBinaryTreeModel(
                dimension_size=config["dimension_size"],
                cppn_hidden_node_num=config["cppn_hidden_node_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    # @override(VectorEnv)
    # def reset_at(self, index: Optional[int] = None):
    #     if index is None:
    #         index = 0
    #     self.env_models[index].reset(
    #         np.random.randint(0, self.initial_max_random_steps + 1)
    #     )
    #     self.previous_rewards[index] = 0
    #     self.robots[index] = "\n"
    #     self.robot_sim_histories[index] = ""
    #     self.state_data[index] = None
    #     return self.env_models[index].observe()
    #
    # @override(VectorEnv)
    # def vector_reset(self):
    #     for model in self.env_models:
    #         model.reset(np.random.randint(0, self.initial_max_random_steps + 1))
    #
    #     self.previous_rewards = [0 for _ in range(self.num_envs)]
    #     self.robots = ["\n" for _ in range(self.num_envs)]
    #     self.robot_sim_histories = ["" for _ in range(self.num_envs)]
    #     self.state_data = [None for _ in range(self.num_envs)]
    #     return [model.observe() for model in self.env_models]

    # def get_rewards(self, all_finished):
    #     base_rewards = super().get_rewards(all_finished)
    #     for idx, model in enumerate(self.env_models):
    #         base_rewards[idx] += model.get_cppn_reward()
    #     return base_rewards


class VoxcraftCPPNBinaryTreeWithPhaseOffsetEnvironment(VoxcraftBaseEnvironment):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        self.initial_max_random_steps = config.get("initial_max_random_steps", 0)
        if config.get("debug", False):
            enable_debugger(config["debug_ip"], config["debug_port"])
        env_models = [
            CPPNBinaryTreeWithPhaseOffsetModel(
                dimension_size=config["dimension_size"],
                cppn_hidden_node_num=config["cppn_hidden_node_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    # @override(VectorEnv)
    # def reset_at(self, index: Optional[int] = None):
    #     if index is None:
    #         index = 0
    #     self.env_models[index].reset(
    #         np.random.randint(0, self.initial_max_random_steps + 1)
    #     )
    #     self.previous_rewards[index] = 0
    #     self.robots[index] = "\n"
    #     self.robot_sim_histories[index] = ""
    #     self.state_data[index] = None
    #     return self.env_models[index].observe()
    #
    # @override(VectorEnv)
    # def vector_reset(self):
    #     for model in self.env_models:
    #         model.reset(np.random.randint(0, self.initial_max_random_steps + 1))
    #
    #     self.previous_rewards = [0 for _ in range(self.num_envs)]
    #     self.robots = ["\n" for _ in range(self.num_envs)]
    #     self.robot_sim_histories = ["" for _ in range(self.num_envs)]
    #     self.state_data = [None for _ in range(self.num_envs)]
    #     return [model.observe() for model in self.env_models]
