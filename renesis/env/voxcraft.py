import copy
import numpy as np
from time import time
from gym.spaces import Box
from ray.rllib import VectorEnv
from ray.rllib.utils import override
from renesis.sim import Voxcraft
from renesis.utils.voxcraft import vxd_creator, get_voxel_positions
from renesis.utils.fitness import max_z, table, distance_traveled, has_fallen
from renesis.entities.growth_function import GrowthFunction


class VoxcraftGrowthEnvironment(VectorEnv):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config):
        self.genomes = [
            GrowthFunction(
                materials=config["materials"],
                max_dimension_size=config["max_dimension_size"],
                max_view_size=config["max_view_size"],
                actuation_features=config["actuation_features"],
            )
            for _ in range(config["num_envs"])
        ]
        self.amplitude_range = config["amplitude_range"]
        self.frequency_range = config["frequency_range"]
        self.phase_offset_range = config["phase_offset_range"]
        self.max_steps = config["max_steps"]
        self.action_space = Box(
            low=0, high=1, shape=[int(np.prod(self.genomes[0].action_shape))]
        )
        self.observation_space = Box(low=0, high=1, shape=self.genomes[0].view_shape)
        self.reward_range = (0, float("inf"))
        with open(config["base_config_path"], "r") as file:
            self.base_config = file.read()
        self.reward_type = config["reward_type"]
        self.voxel_size = config["voxel_size"]
        self.fallen_threshold = config["fallen_threshold"]

        self.previous_rewards = [0 for _ in range(config["num_envs"])]
        self.robots = ["\n" for _ in range(config["num_envs"])]
        self.robot_sim_histories = ["" for _ in range(config["num_envs"])]
        self.simulator = Voxcraft()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def vector_reset(self):
        for genome in self.genomes:
            genome.reset()

        self.previous_rewards = [0 for _ in range(self.num_envs)]
        self.robots = ["\n" for _ in range(self.num_envs)]
        self.robot_sim_histories = ["" for _ in range(self.num_envs)]
        return [genome.get_local_view() for genome in self.genomes]

    @override(VectorEnv)
    def reset_at(self, index=None):
        if index is None:
            index = 0
        self.genomes[index].reset()
        self.previous_rewards[index] = 0
        self.robots[index] = "\n"
        self.robot_sim_histories[index] = ""
        return self.genomes[index].get_local_view()

    @override(VectorEnv)
    def vector_step(self, actions):
        all_finished = self.check_finished()
        for genome, action, finished in zip(self.genomes, actions, all_finished):
            # print(action.reshape(genome.action_shape))
            if not finished:
                genome.step(action.reshape(genome.action_shape))

        rewards = self.get_rewards(all_finished)
        print(f"Rewards: {rewards}")
        reward_diffs = [
            reward - previous_reward
            for reward, previous_reward in zip(rewards, self.previous_rewards)
        ]
        print(f"Reward diffs: {reward_diffs}")
        self.previous_rewards = rewards
        print(f"Finished: {self.check_finished()}")
        return (
            [genome.get_local_view() for genome in self.genomes],
            reward_diffs,
            self.check_finished(),
            [{} for _ in range(self.num_envs)],
        )

    def get_rewards(self, all_finished):
        rewards = copy.deepcopy(self.previous_rewards)
        valid_genomes = []
        valid_genome_indices = []
        for idx, (genome, finished) in enumerate(zip(self.genomes, all_finished)):
            if genome.num_non_zero_voxel > 0 and not finished:
                valid_genomes.append(genome)
                valid_genome_indices.append(idx)

        robots, (results, records) = self.run_simulations(valid_genomes)
        for robot, result, record, idx in zip(
            robots, results, records, valid_genome_indices
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
        return rewards

    def run_simulations(self, genomes):
        """
        Run a simulation using current representation.

        Returns:
            Path to the output.xml file.
            Path to the temporary directory (needs to be deleted).
        """
        robots = []
        for genome in genomes:
            sizes, representation = genome.get_representation(
                self.amplitude_range, self.frequency_range, self.phase_offset_range
            )

            robots.append(vxd_creator(sizes, representation, record_history=True))
        begin = time()
        out = self.simulator.run_sims([self.base_config] * len(robots), robots)
        end = time()
        print(
            f"{self.num_envs} simulations total {end - begin:.3f}s, "
            f"average {(end - begin) / self.num_envs:.3f}s"
        )
        # print(out[1][0])
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
        # print(reward)
        if np.isnan(reward):
            reward = 0
        return reward

    def check_finished(self):
        return [
            not genome.building() or (genome.steps == self.max_steps)
            for genome in self.genomes
        ]

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot[0] + "\n"
