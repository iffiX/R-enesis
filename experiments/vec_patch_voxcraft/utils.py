import os
import shutil
import pickle
import numpy as np
import torch as t
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
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
        self, *, worker, base_env, policies, episode, env_index, **kwargs
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
        episode.media["episode_data"]["robot"] = env.end_robots[env_index]
        episode.media["episode_data"]["record"] = env.end_records[env_index]
        episode.media["episode_data"]["patches"] = np.array(
            env.vec_env_model.vec_patches
        )[:, env_index]
        episode.media["episode_data"]["voxels"] = env.vec_env_model.vec_voxels[
            env_index
        ].astype(np.int8)

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

        result["episode_media"] = {
            "data": data,
        }


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
        with open(os.path.join(trial.logdir, "data", "base.vxa"), "w") as file:
            file.write(self.base_config)

    def log_trial_result(self, iteration, trial, result):
        iteration = result[TRAINING_ITERATION]
        step = result[TIMESTEPS_TOTAL]

        data = result["episode_media"].get("data", None)
        result["episode_media"] = {}

        self.process_data(
            iteration=iteration,
            step=step,
            trial=trial,
            data=data,
        )

    def process_data(self, iteration, step, trial, data):
        if data:
            best_reward = max([d["reward"] for d in data])
            with open(
                os.path.join(
                    self._trial_local_dir[trial],
                    f"data_it_{iteration}_rew_{best_reward}.data",
                ),
                "wb",
            ) as file:
                pickle.dump(data, file)

        print("Saving completed")
