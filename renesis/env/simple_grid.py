import gym
import numpy as np
from gym.spaces import Box
from renesis.utils.fitness import (
    max_z,
    table,
    max_volume,
    max_surface_area,
    get_convex_hull_volume,
    max_hull_volume_min_density,
)
from renesis.utils.plotting import plot_voxels
from renesis.env_model.growth import Growth


"""A 3D grid environment in which creatures iteratively grow."""


class SimpleGridGrowthEnvironment(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        self.genome = Growth(
            materials=config["materials"],
            max_dimension_size=config["max_dimension_size"],
            max_view_size=config["max_view_size"],
        )

        self.max_steps = config["max_steps"]
        self.action_space = Box(
            low=0, high=1, shape=[int(np.prod(self.genome.action_shape))]
        )
        self.observation_space = Box(low=0, high=1, shape=self.genome.view_shape)
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]

    def reset(self, **kwargs):
        self.genome.reset()
        self.previous_reward = 0
        return self.genome.get_local_view()

    def step(self, action):
        self.genome.step(action.reshape(self.genome.action_shape))
        if 0 < self.genome.steps and self.genome.steps % self.reward_interval == 0:
            reward = self.get_reward(
                self.genome.occupied_positions,
                self.genome.voxels[:, :, :, 0].astype(np.int),
            )
            self.previous_reward = reward

        done = (not self.genome.building()) or (self.genome.steps == self.max_steps)
        return self.genome.get_local_view(), self.previous_reward, done, {}

    def get_reward(self, final_positions, X):
        if self.reward_type == "max_z":
            reward = max_z(final_positions)
        elif self.reward_type == "table":
            reward = table(final_positions)
        elif self.reward_type == "max_volume":
            reward = max_volume(X)
        elif self.reward_type == "max_surface_area":
            reward = max_surface_area(X)
        elif self.reward_type == "convex_hull_volume":
            reward = get_convex_hull_volume(final_positions)
        elif self.reward_type == "max_hull_volume_min_density":
            reward = max_hull_volume_min_density(final_positions)
        else:
            raise Exception("Unknown reward type: {self.reward}")
        return reward

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            # Most unfortunetly this calls vtk which has a memory leak.
            # Best to only call during a short evaluation.
            img = plot_voxels(
                self.genome.occupied_positions, self.genome.occupied_values
            )
            return img
