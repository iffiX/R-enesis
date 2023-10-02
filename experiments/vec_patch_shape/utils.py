import os
import shutil
import pickle
import numpy as np
import torch as t
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from renesis.utils.metrics import (
    get_volume,
    get_surface_area,
    get_surface_voxels,
    get_section_num,
    get_reflection_symmetry,
)
from launch.snapshot import get_snapshot

t.set_printoptions(threshold=10000)


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = {
            "steps": [],
            "step_dists": [],
            "reward": 0,
            "patches": None,
            "voxels": None,
        }

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch: SampleBatch,
        original_batches,
        **kwargs,
    ) -> None:
        episode.media["episode_data"]["step_dists"] = postprocessed_batch[
            SampleBatch.ACTION_DIST_INPUTS
        ]
        episode.media["episode_data"]["steps"] = postprocessed_batch[
            SampleBatch.ACTIONS
        ]

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
            assert base_env.vector_env.vec_env_model.is_finished(), (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )

        env = base_env.vector_env
        episode.media["episode_data"]["steps"] = np.stack(
            episode.media["episode_data"]["steps"]
        )
        episode.media["episode_data"]["reward"] = env.end_rewards[env_index]
        episode.media["episode_data"]["patches"] = np.array(
            env.vec_env_model.vec_patches
        )[:, env_index]
        episode.media["episode_data"]["voxels"] = env.vec_env_model.vec_voxels[
            env_index
        ].astype(np.int8)
        episode.custom_metrics["real_reward"] = env.end_rewards[env_index]
        metrics = self.get_robot_metric(env.vec_env_model.vec_voxels[env_index])
        episode.custom_metrics.update(metrics)

    def on_train_result(
        self,
        *,
        algorithm,
        result,
        **kwargs,
    ) -> None:
        # Use sampled data from training instead of evaluation to speed up
        # Note that the first epoch of data, the model is untrained, while for
        # evaluation, since it is performed after training PPO, the model is
        # trained for 1 epoch
        data = result["sampler_results"]["episode_media"].get("episode_data", [])

        result["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}

        # Aggregate results
        result["episode_media"] = {
            "raw_data": data,
            "episode_data": {
                "rewards": [episode_data["reward"] for episode_data in data],
            },
        }

    def get_robot_metric(self, voxels):
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
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

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
                    f"data_it_{iteration}_rew_{np.max(data['rewards'])}.data",
                ),
                "wb",
            ) as file:
                pickle.dump(raw_data, file)

        print("Saving completed")
