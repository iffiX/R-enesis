import gym
import numpy as np
from gym.spaces import Box
from renesis.utils.plotter import Plotter
from renesis.env_model.cppn import CPPNVirtualShapeBinaryTreeModel
from renesis.env_model.gmm import GMMModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class VirtualShapeCPPNEnvironment(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self.plotter = Plotter()
        self.env_model = CPPNVirtualShapeBinaryTreeModel(
            dimension_size=config["dimension_size"],
            cppn_hidden_node_num=config["cppn_hidden_node_num"],
        )
        self.reference_shape = config["reference_shape"]
        self.max_steps = config["max_steps"]
        self.action_space = self.env_model.action_space
        self.observation_space = self.env_model.observation_space
        self.previous_reward = 0
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.render_config = config.get(
            "render_config", {"distance": config["dimension_size"] * 2}
        )
        self.best_reward = 0
        # self.actions = []

    def reset(self, **kwargs):
        self.env_model.reset()
        self.previous_reward = 0
        # str = "Actions:\n"
        # for act in self.actions:
        #     str += f"{act}\n"
        # print(str)
        return self.env_model.observe()

    def step(self, action):
        self.env_model.step(action)
        reward = self.get_reward()
        reward_diff = reward - self.previous_reward
        self.previous_reward = reward

        done = self.env_model.is_finished() or (self.env_model.steps == self.max_steps)
        # print(f"Step {self.env_model.steps}: reward {reward} action {action}")
        # print(reward_diff)
        # self.actions.append(action)
        return self.env_model.observe(), reward_diff, done, {}

    def get_reward(self):
        if self.reward_type == "correct_rate":
            reward = (
                10
                * np.sum(
                    np.logical_and(
                        self.reference_shape != 0,
                        self.env_model.voxels == self.reference_shape,
                    )
                )
                / np.sum(self.reference_shape != 0)
            )
        else:
            raise Exception(f"Unknown reward type: {self.reward_type}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.plotter.plot_voxel(
                self.env_model.get_non_zero_voxel_positions(),
                self.env_model.get_non_zero_voxel_materials(),
                **self.render_config,
            )
            return img


class VirtualShapeGMMEnvironment(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self.plotter = Plotter()
        self.env_model = GMMModel(
            materials=config["materials"],
            dimension_size=config["dimension_size"],
            max_gaussian_num=config["max_gaussian_num"],
        )
        self.reference_shape = config["reference_shape"]
        self.max_steps = config["max_steps"]
        self.action_space = self.env_model.action_space
        self.observation_space = self.env_model.observation_space
        # self.observation_space = Box(low=0, high=1, shape=(8,))
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.render_config = config.get(
            "render_config", {"distance": config["dimension_size"] * 2}
        )
        self.previous_reward = self.get_reward()
        self.best_reward = 0
        # self.actions = []

    def reset(self, **kwargs):
        self.env_model.reset()
        self.previous_reward = self.get_reward()
        # str = "Actions:\n"
        # for act in self.actions:
        #     str += f"{act}\n"
        # print(str)
        # return np.concatenate(
        #     (self.env_model.observe(), np.array([self.previous_reward / 10]))
        # )
        return self.env_model.observe()

    def step(self, action):
        # self.env_model.step(sigmoid(action))
        self.env_model.step((np.clip(action, -2, 2) + 2) / 4)
        reward = self.get_reward()
        reward_diff = reward - self.previous_reward
        self.previous_reward = reward

        done = self.env_model.is_finished() or (self.env_model.steps == self.max_steps)
        # print(f"Step {self.env_model.steps}: reward {reward} action {action}")
        # print(reward_diff)
        # self.actions.append(action)
        return (
            # np.concatenate((self.env_model.observe(), np.array([reward / 10]))),
            self.env_model.observe(),
            reward_diff,
            done,
            {},
        )

    def get_reward(self):
        correct_num = np.sum(
            np.logical_and(
                self.reference_shape != 0,
                self.env_model.voxels == self.reference_shape,
            )
        )
        if self.reward_type == "recall":
            reward = 10 * correct_num / (np.sum(self.reference_shape != 0) + 1e-3)
        elif self.reward_type == "precision":
            reward = 10 * correct_num / (np.sum(self.env_model.voxels != 0) + 1e-3)
        elif self.reward_type == "f1":
            recall = correct_num / (np.sum(self.reference_shape != 0) + 1e-3)
            precision = correct_num / (np.sum(self.env_model.voxels != 0) + 1e-3)
            reward = 20 * precision * recall / (precision + recall + 1e-3)
        else:
            raise Exception(f"Unknown reward type: {self.reward_type}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.plotter.plot_voxel(
                self.env_model.voxels,
                **self.render_config,
            )
            return img
