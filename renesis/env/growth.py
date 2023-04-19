import gym
import numpy as np
from renesis.utils.metrics import (
    max_z,
    table,
    max_volume,
    max_surface_area,
    get_convex_hull_volume,
    max_hull_volume_min_density,
)
from renesis.utils.plotter import Plotter
from renesis.env_model.growth import GrowthModel


"""A 3D grid environment in which creatures iteratively grow."""


class GrowthEnvironment(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self.plotter = Plotter()
        self.env_model = GrowthModel(
            materials=config["materials"],
            max_dimension_size=config["max_dimension_size"],
            max_view_size=config["max_view_size"],
        )

        self.max_steps = config["max_steps"]
        self.action_space = self.env_model.action_space
        self.observation_space = self.env_model.observation_space
        self.previous_reward = 0
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]

    def reset(self, **kwargs):
        self.env_model.reset()
        self.previous_reward = 0
        return self.env_model.observe()

    def step(self, action):
        self.env_model.step(action.reshape(self.env_model.action_shape))
        if (
            0 < self.env_model.steps
            and self.env_model.steps % self.reward_interval == 0
        ):
            reward = self.get_reward(
                self.env_model.occupied_non_zero_positions,
                self.env_model.voxels[:, :, :, 0].astype(np.int),
            )
            self.previous_reward = reward

        done = self.env_model.is_finished() or (self.env_model.steps == self.max_steps)
        return self.env_model.observe(), self.previous_reward, done, {}

    def get_reward(self, non_zero_voxel_positions, voxels):
        if self.reward_type == "max_z":
            reward = max_z(non_zero_voxel_positions)
        elif self.reward_type == "table":
            reward = table(non_zero_voxel_positions)
        elif self.reward_type == "max_volume":
            reward = max_volume(voxels)
        elif self.reward_type == "max_surface_area":
            reward = max_surface_area(voxels)
        elif self.reward_type == "convex_hull_volume":
            reward = get_convex_hull_volume(non_zero_voxel_positions)
        elif self.reward_type == "max_hull_volume_min_density":
            reward = max_hull_volume_min_density(non_zero_voxel_positions)
        else:
            raise Exception(f"Unknown reward type: {self.reward_type}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.plotter.plot_voxel(
                self.env_model.occupied_non_zero_positions,
                self.env_model.occupied_non_zero_values,
            )
            return img
