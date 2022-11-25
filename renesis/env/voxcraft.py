import os.path

import gym
import numpy as np
from uuid import uuid4
from gym.spaces import Box
from renesis.sim import voxcraft_bin_path
from renesis.utils.voxcraft import vxd_creator, get_voxel_positions
from renesis.utils.fitness import max_z, table, distance_traveled, has_fallen
from renesis.entities.growth_function import GrowthFunction
import subprocess


class VoxcraftGrowthEnvironment(gym.Env):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        print("Init")
        self.genome = GrowthFunction(
            materials=config["materials"],
            max_dimension_size=config["max_dimension_size"],
            max_view_size=config["max_view_size"],
        )
        self.amplitude_range = config["amplitude_range"]
        self.frequency_range = config["frequency_range"]
        self.phase_shift_range = config["phase_shift_range"]
        self.max_steps = config["max_steps"]
        self.spec = gym.envs.registration.EnvSpec("voxcraft")
        self.spec.max_episode_steps = self.max_steps
        self.action_space = Box(
            low=0, high=1, shape=[int(np.prod(self.genome.action_shape))]
        )
        self.observation_space = Box(low=0, high=1, shape=self.genome.view_shape)
        self.reward_range = (0, float("inf"))
        self.base_template_path = config["base_template_path"]
        self.reward_type = config["reward_type"]
        self.reward_interval = config["reward_interval"]
        self.voxel_size = config["voxel_size"]
        self.fallen_threshold = config["fallen_threshold"]

        self.previous_reward = 0
        self.robot = "\n"
        self.robot_sim_history = ""

    def reset(self, **kwargs):
        self.genome.reset()
        self.previous_reward = 0
        return self.genome.get_local_view()

    def step(self, action):
        print(f"Step: {self.genome.steps}")
        self.genome.step(action.reshape(self.genome.action_shape))
        reward_diff = 0
        if self.genome.steps != 0 and self.genome.steps % self.reward_interval == 0:
            reward = self.get_reward()
            reward_diff = reward - self.previous_reward
            self.previous_reward = reward

        done = not self.genome.building() or (self.genome.steps == self.max_steps)
        return self.genome.get_local_view(), reward_diff, done, {}

    def get_reward(self):
        if self.genome.num_non_zero_voxel == 0:
            # return directly since the simulator will throw an error
            return 0

        out_path, sim_path = self.run_simulation()
        initial_positions, final_positions = get_voxel_positions(
            out_path, voxel_size=self.voxel_size
        )
        reward = self.compute_reward_from_sim_result(initial_positions, final_positions)
        subprocess.run(f"rm -fr {sim_path}".split())
        print(f"Reward: {reward}")
        return reward

    def run_simulation(self):
        """
        Run a simulation using current representation.

        Returns:
            Path to the output.xml file.
            Path to the temporary directory (needs to be deleted).
        """
        folder = uuid4()
        sim_path = f"/tmp/{folder}"
        out_path = f"{sim_path}/output.xml"
        history_path = f"{sim_path}/run.history"
        base_path = f"{sim_path}/base.vxa"
        subprocess.run(f"mkdir -p {sim_path}".split())
        subprocess.run(f"cp {self.base_template_path} {base_path}".split())
        self.generate_sim_data(sim_path)
        run_command = f"./voxcraft-sim -f -i {sim_path} -o {out_path}"
        with open(history_path, "w") as file:
            if (
                subprocess.run(
                    run_command.split(),
                    cwd=os.path.dirname(voxcraft_bin_path),
                    stdout=file,
                ).returncode
                != 0
            ):
                raise ValueError("Exception occurred in simulation")
        with open(history_path, "r") as file:
            self.robot_sim_history = file.read()
        return out_path, sim_path

    def generate_sim_data(self, data_dir_path):
        sizes, representation = self.genome.get_representation(
            self.amplitude_range, self.frequency_range, self.phase_shift_range
        )
        robot_path = data_dir_path + "/robot.vxd"
        self.robot = vxd_creator(
            sizes, representation, robot_path, record_history=False
        )
        # print(self.robot)

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
        print(reward)
        return reward

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot + "\n"
