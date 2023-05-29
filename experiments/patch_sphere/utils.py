import os
import shutil
import pickle
import numpy as np
import torch as t
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from renesis.env.voxcraft import (
    VoxcraftSingleRewardTestPatchSphereEnvironment,
)
from renesis.env_model.patch import PatchModel
from renesis.utils.metrics import (
    get_volume,
    get_surface_area,
    get_surface_voxels,
    get_section_num,
    get_reflection_symmetry,
)
from renesis.utils.debug import enable_debugger
from launch.snapshot import get_snapshot

t.set_printoptions(threshold=10000)


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = {
            "steps": [],
            "step_dists": [],
            "reward": 0,
            "robot": "",
            "patches": None,
            "voxels": None,
        }

    def on_episode_step(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        **kwargs,
    ) -> None:
        episode.media["episode_data"]["step_dists"].append(
            episode.last_extra_action_outs_for()["action_dist_inputs"]
        )
        episode.media["episode_data"]["steps"].append(episode.last_action_for())

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
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

        env = (
            base_env.vector_env
        )  # type: VoxcraftSingleRewardTestPatchSphereEnvironment
        episode.media["episode_data"]["steps"] = np.stack(
            episode.media["episode_data"]["steps"]
        )
        episode.media["episode_data"]["reward"] = env.end_rewards[env_index]
        episode.media["episode_data"]["robot"] = env.end_robots[env_index]
        episode.media["episode_data"]["patches"] = env.end_state_data[env_index][0]
        episode.media["episode_data"]["voxels"] = env.end_state_data[env_index][
            1
        ].astype(np.int8)
        episode.custom_metrics["real_reward"] = env.end_rewards[env_index]
        metrics = self.get_robot_metric(env.env_models[env_index])
        episode.custom_metrics.update(metrics)

    def on_train_result(
        self,
        *,
        algorithm,
        result,
        trainer,
        **kwargs,
    ) -> None:
        # # Remove non-evaluation data
        # result["episode_media"] = {}
        # # print("Custom metrics:")
        # # print(result["custom_metrics"])
        # if "sampler_results" in result:
        #     result["sampler_results"]["episode_media"] = {}
        #
        # if "evaluation" in result:
        #     data = result["evaluation"]["episode_media"].get("episode_data", [])
        #
        #     if len(data) > 0:
        #         # Aggregate results
        #         rewards = []
        #         best_reward = -np.inf
        #         best_robot = None
        #         for episode_data in data:
        #             rewards.append(episode_data["reward"])
        #             if episode_data["reward"] > best_reward:
        #                 best_reward = episode_data["reward"]
        #                 best_robot = episode_data["robot"]
        #
        #         result["evaluation"]["episode_media"] = {
        #             "raw_data": data,
        #             "episode_data": {
        #                 "rewards": rewards,
        #                 "best_reward": best_reward,
        #                 "best_robot": best_robot,
        #             },
        #         }

        # Use sampled data from training instead of evaluation to speed up
        # Note that the first epoch of data, the model is untrained, while for
        # evaluation, since it is performed after training PPO, the model is
        # trained for 1 epoch
        data = result["sampler_results"]["episode_media"].get("episode_data", [])

        result["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}

        # Aggregate results
        rewards = []
        best_reward = -np.inf
        best_robot = None
        for episode_data in data:
            rewards.append(episode_data["reward"])
            if episode_data["reward"] > best_reward:
                best_reward = episode_data["reward"]
                best_robot = episode_data["robot"]

        result["episode_media"] = {
            "raw_data": data,
            "episode_data": {
                "rewards": rewards,
                "best_reward": best_reward,
                "best_robot": best_robot,
            },
        }

    def get_robot_metric(self, env_model: PatchModel):
        voxels = env_model.get_voxels()
        metrics = {}
        metrics["volume"] = get_volume(voxels)
        metrics["surface_area"] = get_surface_area(voxels)
        metrics["surface_voxels"] = get_surface_voxels(voxels)
        metrics["surface_area_to_total_volume_ratio"] = (
            metrics["surface_area"] / metrics["volume"]
        )
        metrics["surface_voxels_to_total_volume_ratio"] = (
            metrics["surface_voxels"] / metrics["volume"]
        )
        metrics["section_num"] = get_section_num(voxels)
        metrics["reflection_symmetry"] = get_reflection_symmetry(voxels)
        return metrics


class DataLoggerCallback(LoggerCallback):
    def __init__(self, base_config_path):
        self._trial_continue = {}
        self._trial_local_dir = {}
        with open(base_config_path, "r") as file:
            self.base_config = file.read()

    def log_trial_start(self, trial):
        trial.init_logdir()
        snapshot = get_snapshot()
        shutil.move(snapshot, os.path.join(trial.logdir, "code"))
        self._trial_local_dir[trial] = os.path.join(trial.logdir, "data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        iteration = result[TRAINING_ITERATION]
        step = result[TIMESTEPS_TOTAL]

        # raw_data = result["evaluation"]["episode_media"].get("raw_data", None)
        # data = result["evaluation"]["episode_media"].get("episode_data", None)
        # custom_metrics = result["evaluation"].get("custom_metrics", None)
        # result["evaluation"]["episode_media"] = {}

        raw_data = result["episode_media"].get("raw_data", None)
        data = result["episode_media"].get("episode_data", None)
        custom_metrics = result.get("custom_metrics", None)
        result["episode_media"] = {}

        self.process_data(
            iteration=iteration,
            step=step,
            trial=trial,
            raw_data=raw_data,
            data=data,
            custom_metrics=custom_metrics,
        )

    def process_data(self, iteration, step, trial, raw_data, data, custom_metrics):
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
                        custom_metrics,
                    )
                ]
                pickle.dump(metrics, file)

            with open(
                os.path.join(
                    self._trial_local_dir[trial],
                    f"data_it_{iteration}_rew_{data['best_reward']}.data",
                ),
                "wb",
            ) as file:
                pickle.dump(raw_data, file)

        print("Saving completed")
