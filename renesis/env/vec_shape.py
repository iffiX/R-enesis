import numpy as np
from typing import List, Dict, Any, Optional
from ray.rllib import VectorEnv
from ray.rllib.utils import override
from renesis.utils.debug import enable_debugger
from renesis.env_model.base import BaseVectorizedModel
from renesis.env_model.gmm import normalize
from renesis.env_model.vec_patch import (
    VectorizedPatchModel,
    VectorizedPatchSphereModel,
)
from renesis.utils.metrics import get_volume, get_bounding_box_sizes


class ShapeBaseEnvironmentForVecEnvModel(VectorEnv):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, config: Dict[str, Any], vec_env_model: BaseVectorizedModel):
        self.config = config
        self.vec_env_model = vec_env_model
        self.max_steps = config["max_steps"]
        self.action_space = vec_env_model.action_space
        self.observation_space = vec_env_model.observation_space
        self.reward_range = (0, float("inf"))
        self.reward_type = config["reward_type"]
        self.voxel_size = config["voxel_size"]

        self.end_rewards = [0 for _ in range(config["num_envs"])]
        self.reset_envs = set()
        super().__init__(self.observation_space, self.action_space, config["num_envs"])

    @override(VectorEnv)
    def reset_at(self, index: Optional[int] = None, *args, **kwargs):
        # Will reset all models at end
        self.end_rewards[index] = 0
        if index in self.reset_envs:
            raise RuntimeError("Environment already reset")
        else:
            self.reset_envs.add(index)
            if len(self.reset_envs) == self.num_envs:
                self.vec_env_model.reset()
                self.reset_envs.clear()
        return self.vec_env_model.initial_observation_after_reset_single_env, None

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None):
        self.reset_at(index)

    @override(VectorEnv)
    def vector_reset(self):
        self.vec_env_model.reset()
        self.end_rewards = [0 for _ in range(self.num_envs)]
        return self.vec_env_model.observe(), [{} for _ in range(self.num_envs)]

    @override(VectorEnv)
    def vector_step(self, actions):
        before_finished = self.vec_env_model.is_finished() or (
            self.vec_env_model.steps == self.max_steps
        )
        if not before_finished:
            self.vec_env_model.step(actions)
        after_finished = self.vec_env_model.is_finished() or (
            self.vec_env_model.steps == self.max_steps
        )

        if not before_finished and after_finished:
            self.update_rewards()

        # print(actions)
        # print([f"{r:.3f}" for r in reward_signals])
        return (
            self.vec_env_model.observe(),
            self.end_rewards
            if not before_finished and after_finished
            else [0] * self.num_envs,
            [after_finished] * self.num_envs,
            [False for _ in range(self.num_envs)],
            [{} for _ in range(self.num_envs)],
        )

    def update_rewards(self):
        """
        Update rewards of all sub environments.
        """
        for idx, voxels in enumerate(self.vec_env_model.get_robots_voxels()):
            if self.reward_type == "volume":
                self.end_rewards[idx] = get_volume(voxels)
            elif self.reward_type == "height":
                self.end_rewards[idx] = get_bounding_box_sizes(voxels)[2]
            else:
                raise Exception(f"Unknown reward type: {self.reward_type}")
        self.end_rewards = [
            reward if not np.isnan(reward) else 0 for reward in self.end_rewards
        ]

    def render(self, mode="ansi"):
        if mode == "ansi":
            return self.robot[0] + "\n"


class ShapeVectorizedPatchEnvironment(ShapeBaseEnvironmentForVecEnvModel):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        super().__init__(
            config,
            VectorizedPatchModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
                env_num=config["num_envs"],
            ),
        )

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )


class ShapeVectorizedPatchSphereEnvironment(ShapeBaseEnvironmentForVecEnvModel):
    def __init__(self, config):
        if config.get("debug", False):
            enable_debugger(
                config.get("debug_ip", "localhost"), config.get("debug_port", 8223)
            )
        super().__init__(
            config,
            VectorizedPatchSphereModel(
                materials=config["materials"],
                dimension_size=config["dimension_size"],
                patch_size=config["patch_size"],
                max_patch_num=config["max_patch_num"],
                env_num=config["num_envs"],
            ),
        )

    def vector_step(self, actions):
        normalize_mode = self.config.get("normalize_mode", "clip")
        return super().vector_step(
            [normalize(action, mode=normalize_mode) for action in actions]
        )
