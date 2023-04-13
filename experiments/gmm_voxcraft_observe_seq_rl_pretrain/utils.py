import os
import shutil
import pickle
import numpy as np
import torch as t
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from renesis.env.voxcraft import VoxcraftGMMEnvironment
from renesis.utils.media import create_video_subproc
from renesis.sim import Voxcraft, VXHistoryRenderer
from launch.snapshot import get_snapshot

t.set_printoptions(threshold=10000)


def render(history):
    try:
        renderer = VXHistoryRenderer(history=history, width=640, height=480)
        renderer.render()
        frames = renderer.get_frames()
        if frames.ndim == 4:
            print("History saved")
            return frames
        else:
            print("Rendering finished, but no frames produced")
            print("History:")
            print(history)
            return None
    except Exception as e:
        print(e)
        print("Exception occurred, no frames produced")
        print("History:")
        print(history)
        return None


class CustomCallbacks(DefaultCallbacks):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        load_pretrain_path = algorithm.config.get("weight_export_path", None)
        if load_pretrain_path is not None:
            algorithm.workers.local_worker().get_policy().set_weights(
                t.load(load_pretrain_path)
            )
            algorithm.workers.sync_weights()
            print("Pretrain weights loaded")

    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = {}

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

        env = base_env.vector_env  # type: VoxcraftGMMEnvironment
        episode.media["episode_data"]["reward"] = env.previous_best_rewards[env_index]
        episode.media["episode_data"]["robot"] = env.previous_best_robots[env_index]
        # May be also record state data (gaussians) ?
        # episode.media["episode_data"]["state_data"] = env.previous_best_state_data[
        #     env_index
        # ]

    def on_train_result(
        self,
        *,
        algorithm,
        result,
        trainer,
        **kwargs,
    ) -> None:
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
                best_robot = None
                for episode_data in data:
                    rewards.append(episode_data["reward"])
                    if episode_data["reward"] > best_reward:
                        best_reward = episode_data["reward"]
                        best_robot = episode_data["robot"]

                result["evaluation"]["episode_media"] = {
                    "episode_data": {
                        "rewards": rewards,
                        "best_reward": best_reward,
                        "best_robot": best_robot,
                    }
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

                robot = data["best_robot"]
                simulator = Voxcraft()
                _, (sim_history,) = simulator.run_sims([self.base_config], [robot])
                frames = render(sim_history)

                if frames is not None:
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"rendered_it_{iteration}_rew_{data['best_reward']}.gif",
                    )
                    print(f"Saving rendered results to {path}")
                    wait = create_video_subproc(
                        [f for f in frames],
                        path=self._trial_local_dir[trial],
                        filename=f"rendered_it_{iteration}",
                        extension=".gif",
                    )
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"robot_it_{iteration}_rew_{data['best_reward']}.vxd",
                    )
                    with open(path, "w") as file:
                        print(f"Saving robot to {path}")
                        file.write(robot)
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"run_it_{iteration}_rew_{data['best_reward']}.history",
                    )
                    with open(path, "w") as file:
                        print(f"Saving history to {path}")
                        file.write(sim_history)
                    wait()

            print("Saving completed")
