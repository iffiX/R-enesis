import os
import copy
import numpy as np
from time import time
from typing import List, Dict, Any, Optional
from ray.rllib import VectorEnv
from ray.rllib.utils import override
from renesis.sim import Voxcraft
from renesis.utils.voxcraft import vxd_creator, get_voxel_positions
from renesis.utils.metrics import max_z, table, distance_traveled, has_fallen
from renesis.utils.debug import enable_debugger
from renesis.env_model.base import BaseModel
from renesis.env_model.cppn import (
    CPPNBinaryTreeModel,
    CPPNBinaryTreeWithPhaseOffsetModel,
)
from renesis.env_model.gmm import (
    GMMModel,
    GMMObserveWithVoxelModel,
    GMMObserveWithVoxelAndRemainingStepsModel,
    GMMWSObserveWithVoxelModel,
    GMMWSObserveWithVoxelAndRemainingStepsModel,
    normalize,
    time_observe_wrapper,
)
from renesis.env_model.patch import PatchModel
from renesis.env_model.growth import GrowthModel
from renesis.utils.metrics import get_surface_area, get_volume


class VoxcraftBaseEnvironment(VectorEnv):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], env_models: List[BaseModel]):
        self.config = config
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
        self.previous_robots = ["\n" for _ in range(config["num_envs"])]
        self.previous_state_data = [None for _ in range(config["num_envs"])]

        self.previous_best_rewards = [0 for _ in range(config["num_envs"])]
        self.previous_best_robots = ["\n" for _ in range(config["num_envs"])]
        self.previous_best_state_data = [None for _ in range(config["num_envs"])]

        self.simulator = Voxcraft()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None):
        if index is None:
            index = 0

        self.env_models[index].reset()

        self.previous_rewards[index] = 0
        self.previous_robots[index] = "\n"
        self.previous_state_data[index] = None

        self.previous_best_rewards[index] = 0
        self.previous_best_robots[index] = "\n"
        self.previous_best_state_data[index] = None

        return self.env_models[index].observe()

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        return [self.reset_at(idx) for idx in range(len(self.env_models))]

    @override(VectorEnv)
    def vector_step(self, actions):
        all_finished = self.check_finished()
        for model, action, finished in zip(self.env_models, actions, all_finished):
            if not finished:
                model.step(action)

        rewards = self.get_rewards(all_finished)
        # print(f"Rewards: {rewards}")

        for i in range(self.num_envs):
            if rewards[i] > self.previous_best_rewards[i]:
                self.previous_best_rewards[i] = rewards[i]
                self.previous_best_robots[i] = self.previous_robots[i]
                self.previous_best_state_data[i] = self.previous_state_data[i]

        reward_diffs = [
            reward - previous_reward
            for reward, previous_reward in zip(rewards, self.previous_rewards)
        ]

        self.previous_rewards = rewards

        # print(f"Actions: \n {actions}")
        # print(f"Rewards: {self.previous_rewards}")
        # print(f"Reward diffs: {reward_diffs}")
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
        empty_count = 0
        for idx, (model, finished) in enumerate(zip(self.env_models, all_finished)):
            if not model.is_robot_invalid() and not finished:
                valid_models.append(model)
                valid_model_indices.append(idx)

            if model.is_robot_invalid() and not finished:
                empty_count += 1
                rewards[idx] = 0
                self.previous_robots[idx] = "\n"
                self.previous_state_data[idx] = None
        print(f"Empty count: {empty_count}")

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
            self.previous_robots[idx] = robot
            self.previous_state_data[idx] = model.get_state_data()
        return rewards

    def run_simulations(self, env_models):
        """
        Run a simulation using current representation.

        Returns:
            A list of robots (each robot is a VXD file in string).
            A tuple, first member is a list of simulation summary
            result string, the second member is a list of simulation
            recording string.
        """
        robots = []
        for model in env_models:
            sizes, representation = model.get_robot()
            robots.append(vxd_creator(sizes, representation, record_history=True))
        if not robots:
            print("No robots in simulation, skipping")
            return [], ([], [])
        begin = time()
        for attempt in range(3):
            try:
                out = self.simulator.run_sims([self.base_config] * len(robots), robots)
                # out = ([None] * len(robots), [None] * len(robots))
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
                print(
                    f"{len(robots)} simulations total {end - begin:.3f}s, "
                    f"average {(end - begin) / len(robots):.3f}s"
                )
                return robots, out

    def compute_reward_from_sim_result(self, initial_positions, final_positions):
        """
        Note: Reward should always have an initial value of 0 for empty robots.
        """
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
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
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
    def __init__(self, config):
        self.initial_max_random_steps = config.get("initial_max_random_steps", 0)
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            CPPNBinaryTreeModel(
                dimension_size=config["dimension_size"],
                cppn_hidden_node_num=config["cppn_hidden_node_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    # def get_rewards(self, all_finished):
    #     base_rewards = super().get_rewards(all_finished)
    #     for idx, model in enumerate(self.env_models):
    #         base_rewards[idx] += model.get_cppn_reward()
    #     return base_rewards


class VoxcraftCPPNBinaryTreeWithPhaseOffsetEnvironment(VoxcraftBaseEnvironment):
    def __init__(self, config):
        self.initial_max_random_steps = config.get("initial_max_random_steps", 0)
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            CPPNBinaryTreeWithPhaseOffsetModel(
                dimension_size=config["dimension_size"],
                cppn_hidden_node_num=config["cppn_hidden_node_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    # def get_rewards(self, all_finished):
    #     base_rewards = super().get_rewards(all_finished)
    #     for idx, model in enumerate(self.env_models):
    #         base_rewards[idx] += model.get_cppn_reward()
    #     return base_rewards


class VoxcraftGMMEnvironment(VoxcraftBaseEnvironment):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                ),
                wrap=config.get("observe_time", False),
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftGMMObserveWithVoxelEnvironment(VoxcraftGMMEnvironment):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMObserveWithVoxelModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                ),
                wrap=config.get("observe_time", False),
            )
            for _ in range(config["num_envs"])
        ]
        super(VoxcraftGMMEnvironment, self).__init__(config, env_models)


class VoxcraftSingleRewardBaseEnvironment(VectorEnv):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], env_models: List[BaseModel]):
        self.config = config
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

        self.end_rewards = [0 for _ in range(config["num_envs"])]
        self.end_robots = ["\n" for _ in range(config["num_envs"])]
        self.end_state_data = [None for _ in range(config["num_envs"])]
        self.empty_record = []

        self.simulator = Voxcraft()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None):
        if index is None:
            index = 0

        self.env_models[index].reset()

        self.end_rewards[index] = 0
        self.end_robots[index] = "\n"
        self.end_state_data[index] = None

        return self.env_models[index].observe()

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        return [self.reset_at(idx) for idx in range(len(self.env_models))]

    @override(VectorEnv)
    def vector_step(self, actions):
        # if len(self.env_models) == 1:
        #     print(actions)
        before_step_all_finished = self.check_finished()
        for model, action, before_finished in zip(
            self.env_models, actions, before_step_all_finished
        ):
            if not before_finished:
                model.step(action)
        after_step_all_finished = self.check_finished()
        should_evaluate = [
            not before_finished and after_finished
            for before_finished, after_finished in zip(
                before_step_all_finished, after_step_all_finished
            )
        ]
        self.update_rewards(should_evaluate)

        reward_signals = [0 for _ in range(len(self.env_models))]
        for idx, (model, before_finished, evaluate) in enumerate(
            zip(self.env_models, before_step_all_finished, should_evaluate)
        ):
            if not before_finished:
                if model.steps == 1:
                    reward_signals[idx] = 1
                else:
                    patch_pos = [
                        np.ceil(model.scale(patch)[:3] - model.patch_size / 2)
                        for patch in model.patches
                    ]
                    diffs = [patch_pos[-1] - p for p in patch_pos[:-1]]
                    dist = np.min([np.linalg.norm(diff) for diff in diffs])
                    if dist < model.patch_size:
                        reward_signals[idx] = 1
                    else:
                        diagonal_length = np.sqrt(
                            np.sum(np.array(model.dimension_size) ** 2)
                        )
                        reward_signals[idx] = 1 - (dist - model.patch_size) / (
                            diagonal_length - model.patch_size
                        )
            if evaluate:
                reward_signals[idx] += self.end_rewards[idx]

        # if len(self.env_models) == 1:
        # print(actions)
        # print([f"{r:.3f}" for r in reward_signals])
        return (
            [model.observe() for model in self.env_models],
            reward_signals,
            self.check_finished(),
            [{} for _ in range(self.num_envs)],
        )

    def update_rewards(self, need_evaluation):
        """
        Returns the list of rewards of all sub environments.

        For sub environments that are not finished in current step, i.e. has executed
        a step, their reward are updated. Otherwise the old reward is returned.

        Args:
            need_evaluation: A list of bool values indicating whether current sub
                environment should be evaluated.

        Returns:
            None
        """
        valid_models = []
        valid_model_indices = []
        for idx, (model, evaluate) in enumerate(zip(self.env_models, need_evaluation)):
            if evaluate:
                if model.is_robot_invalid():
                    self.empty_record.append(True)
                    self.end_rewards[idx] = 0
                    self.end_robots[idx] = "\n"
                    self.end_state_data[idx] = model.get_state_data()
                else:
                    self.empty_record.append(False)
                    valid_models.append(model)
                    valid_model_indices.append(idx)

        if len(self.empty_record) >= 100:
            print(
                f"Empty ratio of past 100 finished episodes: {np.mean(self.empty_record[-100:]):.3f}"
            )
            self.empty_record = []

        robots, (results, records) = self.run_simulations(valid_models)
        for robot, result, record, model, idx in zip(
            robots, results, records, valid_models, valid_model_indices
        ):
            initial_positions, final_positions = get_voxel_positions(
                result, voxel_size=self.voxel_size
            )
            reward = self.compute_reward_from_sim_result(
                initial_positions,
                final_positions,
                model.get_robot_voxels(),
            )
            self.end_rewards[idx] = reward
            self.end_robots[idx] = robot
            self.end_state_data[idx] = model.get_state_data()

    def run_simulations(self, env_models):
        """
        Run a simulation using current representation.

        Returns:
            A list of robots (each robot is a VXD file in string).
            A tuple, first member is a list of simulation summary
            result string, the second member is a list of simulation
            recording string.
        """
        robots = []
        for model in env_models:
            sizes, representation = model.get_robot()
            robots.append(vxd_creator(sizes, representation, record_history=True))
        if not robots:
            return [], ([], [])
        begin = time()
        for attempt in range(3):
            try:
                out = self.simulator.run_sims([self.base_config] * len(robots), robots)
                # out = ([None] * len(robots), [None] * len(robots))
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
                print(
                    f"{len(robots)} simulations total {end - begin:.3f}s, "
                    f"average {(end - begin) / len(robots):.3f}s"
                )
                return robots, out

    def compute_reward_from_sim_result(
        self, initial_positions, final_positions, robot_voxels
    ):
        """
        Note: Reward should always have an initial value of 0 for empty robots.
        """
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
        elif self.reward_type == "distance_traveled_efficiency":
            reward = distance_traveled(initial_positions, final_positions) * (
                np.sum(robot_voxels == 1) / np.sum(robot_voxels != 0)
            )
            print(f"ratio:{np.sum(robot_voxels == 1) / np.sum(robot_voxels != 0)}")
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


class VoxcraftSingleRewardGMMObserveWithVoxelAndRemainingStepsEnvironment(
    VoxcraftSingleRewardBaseEnvironment
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMObserveWithVoxelAndRemainingStepsModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                    reset_seed=config.get(config["reset_seed"], 42) + i,
                    reset_remaining_steps_range=config.get(
                        "reset_remaining_steps_range", None
                    ),
                ),
                wrap=config.get("observe_time", False),
            )
            for i in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardGMMWSObserveWithVoxelEnvironment(
    VoxcraftSingleRewardBaseEnvironment
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMWSObserveWithVoxelModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                ),
                wrap=config.get("observe_time", False),
            )
            for i in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
    VoxcraftSingleRewardBaseEnvironment
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMWSObserveWithVoxelAndRemainingStepsModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                    sigma=config["sigma"],
                    reset_seed=config.get(config["reset_seed"], 42) + i,
                    reset_remaining_steps_range=config.get(
                        "reset_remaining_steps_range", None
                    ),
                ),
                wrap=config.get("observe_time", False),
            )
            for i in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardPatchEnvironment(VoxcraftSingleRewardBaseEnvironment):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            PatchModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardTestBaseEnvironment(VectorEnv):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], env_models: List[BaseModel]):
        self.config = config
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

        self.end_rewards = [0 for _ in range(config["num_envs"])]
        self.end_robots = ["\n" for _ in range(config["num_envs"])]
        self.end_state_data = [None for _ in range(config["num_envs"])]
        self.empty_record = []

        self.simulator = Voxcraft()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None):
        if index is None:
            index = 0

        self.env_models[index].reset()

        self.end_rewards[index] = 0
        self.end_robots[index] = "\n"
        self.end_state_data[index] = None

        return self.env_models[index].observe()

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        return [self.reset_at(idx) for idx in range(len(self.env_models))]

    @override(VectorEnv)
    def vector_step(self, actions):
        # if len(self.env_models) == 1:
        #     print(actions)
        before_step_all_finished = self.check_finished()
        for model, action, before_finished in zip(
            self.env_models, actions, before_step_all_finished
        ):
            if not before_finished:
                model.step(action)
        after_step_all_finished = self.check_finished()
        should_evaluate = [
            not before_finished and after_finished
            for before_finished, after_finished in zip(
                before_step_all_finished, after_step_all_finished
            )
        ]
        self.update_rewards(should_evaluate)

        reward_signals = [0 for _ in range(len(self.env_models))]
        for idx, (model, before_finished, evaluate) in enumerate(
            zip(self.env_models, before_step_all_finished, should_evaluate)
        ):
            if not before_finished:
                # if model.is_robot_invalid():
                #     reward_signals[idx] = 0
                # else:
                #     reward_signals[idx] = (
                #         1 if np.any(model.prev_voxels != model.voxels) else 0.1
                #     )
                if model.steps == 1:
                    reward_signals[idx] = 1
                else:
                    # diffs = [model.patches[-1] - p for p in model.patches[:-1]]
                    # dist = np.min(
                    #     [
                    #         np.linalg.norm(
                    #             diff[:3],
                    #         )
                    #         * model.dimension_size
                    #         for diff in diffs
                    #     ]
                    # )
                    # if dist < model.patch_size:
                    #     reward_signals[idx] = 1
                    # else:
                    #     reward_signals[idx] = 1 - (dist - model.patch_size) / (
                    #         model.dimension_size * np.sqrt(3) - model.patch_size
                    #     )
                    patch_pos = [
                        np.ceil(model.scale(patch)[:3] - model.patch_size / 2)
                        for patch in model.patches
                    ]
                    diffs = [patch_pos[-1] - p for p in patch_pos[:-1]]
                    dist = np.min([np.linalg.norm(diff) for diff in diffs])
                    if dist < model.patch_size:
                        reward_signals[idx] = 1
                    else:
                        reward_signals[idx] = 1 - (dist - model.patch_size) / (
                            model.dimension_size * np.sqrt(3) - model.patch_size
                        )
            if evaluate:
                reward_signals[idx] += self.end_rewards[idx]

        # if len(self.env_models) == 1:
        # print(actions)
        print([f"{r:.3f}" for r in reward_signals])
        return (
            [model.observe() for model in self.env_models],
            reward_signals,
            self.check_finished(),
            [{} for _ in range(self.num_envs)],
        )

    def update_rewards(self, need_evaluation):
        """
        Returns the list of rewards of all sub environments.

        For sub environments that are not finished in current step, i.e. has executed
        a step, their reward are updated. Otherwise the old reward is returned.

        Args:
            need_evaluation: A list of bool values indicating whether current sub
                environment should be evaluated.

        Returns:
            None
        """
        valid_models = []
        valid_model_indices = []
        for idx, (model, evaluate) in enumerate(zip(self.env_models, need_evaluation)):
            if evaluate:
                if model.is_robot_invalid():
                    self.empty_record.append(True)
                    self.end_rewards[idx] = 0
                    self.end_robots[idx] = "\n"
                    self.end_state_data[idx] = model.get_state_data()
                else:
                    self.empty_record.append(False)
                    valid_models.append(model)
                    valid_model_indices.append(idx)

        if len(self.empty_record) >= 100:
            print(
                f"Empty ratio of past 100 finished episodes: {np.mean(self.empty_record[-100:]):.3f}"
            )
            self.empty_record = []

        for model, idx in zip(valid_models, valid_model_indices):
            # sizes, representation = model.get_robot()
            # self.end_rewards[idx] = sizes[2]
            # self.end_rewards[idx] = np.sum(
            #     np.array([np.sum(layer != 0) for layer in representation])
            # )
            voxels = model.get_largest_connected_component_voxels()
            self.end_rewards[idx] = get_volume(voxels) * 20 / get_surface_area(voxels)
            self.end_robots[idx] = ""
            self.end_state_data[idx] = model.get_state_data()

    def check_finished(self):
        return [
            model.is_finished() or (model.steps == self.max_steps)
            for model in self.env_models
        ]

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot[0] + "\n"


class VoxcraftSingleRewardTestGMMWSObserveWithVoxelAndRemainingStepsEnvironment(
    VoxcraftSingleRewardTestBaseEnvironment
):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            time_observe_wrapper(
                GMMWSObserveWithVoxelAndRemainingStepsModel(
                    materials=config["materials"],
                    dimension_size=config["dimension_size"],
                    max_gaussian_num=config["max_gaussian_num"],
                    sigma=config["sigma"],
                    reset_seed=config.get(config["reset_seed"], 42) + i,
                    reset_remaining_steps_range=config.get(
                        "reset_remaining_steps_range", None
                    ),
                ),
                wrap=config.get("observe_time", False),
            )
            for i in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class VoxcraftSingleRewardTestPatchEnvironment(VoxcraftSingleRewardTestBaseEnvironment):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        env_models = [
            PatchModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
            )
            for _ in range(config["num_envs"])
        ]
        super().__init__(config, env_models)

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )
