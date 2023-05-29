import os
import shutil
import pickle
import graphviz
import numpy as np
import torch as t
import pyvista as pv
from PIL import Image
from torch import nn
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import (
    gym,
    Optional,
    ModelConfigDict,
    GTrXLNet,
)
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from renesis.utils.plotter import Plotter
from renesis.utils.debug import print_model_size
from launch.snapshot import get_snapshot

t.set_printoptions(threshold=10000)


class Actor(GTrXLNet, nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        *,
        num_transformer_units: int = 1,
        attention_dim: int = 64,
        num_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            num_outputs,
            model_config,
            name,
            num_transformer_units=num_transformer_units,
            attention_dim=attention_dim,
            num_heads=num_heads,
            memory_inference=memory_inference,
            memory_training=memory_training,
            head_dim=head_dim,
            position_wise_mlp_dim=position_wise_mlp_dim,
            init_gru_gate_bias=init_gru_gate_bias,
        )
        # Remove relu
        self.logits._model = nn.Sequential(*list(self.logits._model.children())[:-1])
        print_model_size(self)


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = {}

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs,
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )

        env = base_env.get_sub_environments()[env_index]
        episode.media["episode_data"]["reward"] = env.get_reward()
        episode.media["episode_data"]["voxels"] = env.env_model.voxels

    def on_train_result(self, *, algorithm, result, trainer, **kwargs,) -> None:
        # Remove non-evaluation data
        result["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", [])

            if len(data) > 0:
                # Aggregate results
                rewards = []
                best_reward = -np.inf
                best_voxels = None
                for episode_data in data:
                    rewards.append(episode_data["reward"])
                    if episode_data["reward"] > best_reward:
                        best_reward = episode_data["reward"]
                        best_voxels = episode_data["voxels"]

                result["evaluation"]["episode_media"] = {
                    "episode_data": {
                        "rewards": rewards,
                        "best_reward": best_reward,
                        "best_voxels": best_voxels,
                    }
                }


class DataLoggerCallback(LoggerCallback):
    def __init__(self, reference_shape, dimension_size, render=True):
        self._trial_continue = {}
        self._trial_local_dir = {}
        self.reference_shape = reference_shape
        self.dimension_size = dimension_size
        self.render = render

    def log_trial_start(self, trial):
        trial.init_logdir()
        snapshot = get_snapshot()
        shutil.move(snapshot, os.path.join(trial.logdir, "code"))
        self._trial_local_dir[trial] = os.path.join(trial.logdir, "data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        iteration = result[TRAINING_ITERATION]
        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", None)
            result["evaluation"]["episode_media"] = {}
            if data:
                log_file = os.path.join(self._trial_local_dir[trial], "metric.data")
                metrics = []
                if os.path.exists(log_file):
                    with open(log_file, "rb") as file:
                        metrics = pickle.load(file)
                with open(log_file, "wb") as file:
                    history_best_reward = -np.inf
                    for history_metric in metrics:
                        if history_metric[0] > history_best_reward:
                            history_best_reward = history_metric[0]
                    metrics += [
                        (
                            max(history_best_reward, np.max(data["rewards"])),
                            np.max(data["rewards"]),
                            np.mean(data["rewards"]),
                            np.min(data["rewards"]),
                        )
                    ]
                    pickle.dump(metrics, file)

                if self.render:
                    pv.global_theme.window_size = [1536, 768]
                    plotter = Plotter()
                    img = plotter.plot_voxel(
                        data["best_voxels"], distance=3 * self.dimension_size
                    )
                    Image.fromarray(img, mode="RGB").save(
                        os.path.join(
                            self._trial_local_dir[trial],
                            f"generated_it_{iteration}_rew_{data['best_reward']}.png",
                        )
                    )

                    img = plotter.plot_voxel_error(
                        self.reference_shape,
                        data["best_voxels"],
                        distance=3 * self.dimension_size,
                    )
                    Image.fromarray(img, mode="RGB").save(
                        os.path.join(
                            self._trial_local_dir[trial], f"error_it_{iteration}.png"
                        )
                    )
                else:
                    with open(
                        os.path.join(
                            self._trial_local_dir[trial],
                            f"generated_data_it_{iteration}_rew_{data['best_reward']}.data",
                        ),
                        "wb",
                    ) as file:
                        pickle.dump(
                            (
                                self.reference_shape,
                                data["best_voxels"],
                                self.dimension_size,
                                os.path.join(
                                    self._trial_local_dir[trial],
                                    f"generated_it_{iteration}_rew_{data['best_reward']}.png",
                                ),
                                os.path.join(
                                    self._trial_local_dir[trial],
                                    f"error_it_{iteration}.png",
                                ),
                            ),
                            file,
                        )

            print("Saving completed")


ModelCatalog.register_custom_model("actor_model", Actor)
