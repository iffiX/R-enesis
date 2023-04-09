import gym
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from renesis.utils.plotter import Plotter
from renesis.env_model.cppn import CPPNVirtualShapeBinaryTreeModel
from renesis.env_model.gmm import GMMModel, GMMObserveSeqModel, normalize


class VirtualShapeBaseEnvironment(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config, env_model, materials):
        self.plotter = Plotter()
        self.env_model = env_model
        self.materials = materials
        self.reference_shape = config["reference_shape"]
        self.max_steps = config["max_steps"]
        self.action_space = self.env_model.action_space
        self.observation_space = self.env_model.observation_space
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.render_config = config.get(
            "render_config", {"distance": config["dimension_size"] * 2}
        )
        self.previous_reward = self.get_reward()
        self.best_reward = 0

    def reset(self, **kwargs):
        self.env_model.reset()
        self.previous_reward = self.get_reward()
        return self.env_model.observe()

    def step(self, action):
        self.env_model.step(action)
        reward = self.get_reward()
        reward_diff = reward - self.previous_reward
        self.previous_reward = reward

        done = self.env_model.is_finished() or (self.env_model.steps == self.max_steps)
        return (
            self.env_model.observe(),
            reward_diff,
            done,
            {},
        )

    def get_reward(self):
        """
        if not start with "multi_":
        recall, precision, f1: Treat the problem as a binary classification problem.

        first we only consider non zero voxels, then we treat it as a binary
        classification problem. Non zero voxels placed correctly are counted as
        true positives. Non zero voxels in the reference shape but placed
        incorrectly are false negatives. Non zero voxels in the input shape but
        not in the reference shape or is placed incorrectly in the reference shape
        are true negatives.


        if starts with "multi_"
        multi_recall, multi_precision, multi_f1: Treats the problem as a multi
            classification problem. Returns the weighted score.
        """
        reward = None
        if not self.reward_type.startswith("multi_"):
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
            if self.env_model.is_robot_empty():
                reward = 0
            else:
                occurences = [
                    np.sum(self.reference_shape == mat) for mat in self.materials
                ]
                # Test notes:
                # use 1/occurrence is too small for voxels with large quantities
                weights = np.array(
                    [occurrence if occurrence > 0 else 0 for occurrence in occurences]
                )
                if self.reward_type == "multi_recall":
                    scores = recall_score(
                        self.reference_shape.reshape(-1),
                        self.env_model.voxels.reshape(-1),
                        labels=self.env_model.materials,
                        average=None,
                        zero_division=0,
                    )
                    reward = 10 * np.average(scores, weights=weights)
                elif self.reward_type == "multi_precision":
                    scores = precision_score(
                        self.reference_shape.reshape(-1),
                        self.env_model.voxels.reshape(-1),
                        labels=self.env_model.materials,
                        average=None,
                        zero_division=0,
                    )
                    reward = 10 * np.average(scores, weights=weights)
                elif self.reward_type == "multi_f1":
                    scores = f1_score(
                        self.reference_shape.reshape(-1),
                        self.env_model.voxels.reshape(-1),
                        labels=self.env_model.materials,
                        average=None,
                        zero_division=0,
                    )
                    reward = 10 * np.average(scores, weights=weights)

        if reward is None:
            raise Exception(f"Unknown reward type: {self.reward_type}")
        return reward

    def render(self, mode="rgb_array"):
        raise NotImplementedError()


class VirtualShapeCPPNEnvironment(VirtualShapeBaseEnvironment):
    def __init__(self, config):
        env_model = CPPNVirtualShapeBinaryTreeModel(
            dimension_size=config["dimension_size"],
            cppn_hidden_node_num=config["cppn_hidden_node_num"],
        )
        super().__init__(config, env_model, (0, 1, 2, 3))

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.plotter.plot_voxel(
                self.env_model.get_non_zero_voxel_positions(),
                self.env_model.get_non_zero_voxel_materials(),
                **self.render_config,
            )
            return img


class VirtualShapeGMMEnvironment(VirtualShapeBaseEnvironment):
    def __init__(self, config):
        env_model = GMMModel(
            materials=config["materials"],
            dimension_size=config["dimension_size"],
            max_gaussian_num=config["max_gaussian_num"],
        )
        super().__init__(config, env_model, env_model.materials)

    def step(self, action):
        return super().step(normalize(action))

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            img = self.plotter.plot_voxel(self.env_model.voxels, **self.render_config,)
            return img


class VirtualShapeGMMObserveSeqEnvironment(VirtualShapeGMMEnvironment):
    def __init__(self, config):
        env_model = GMMModel(
            materials=config["materials"],
            dimension_size=config["dimension_size"],
            max_gaussian_num=config["max_gaussian_num"],
        )
        super(VirtualShapeGMMEnvironment, self).__init__(
            config, env_model, env_model.materials
        )
