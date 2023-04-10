import ray
import experiments.gmm_voxcraft_observe_seq_rl.model
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.algorithms.ppo import PPO
from experiments.gmm_voxcraft_observe_seq_rl.utils import DataLoggerCallback
from experiments.gmm_voxcraft_observe_seq_rl.config import config, iters, steps


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)
    tune.run(
        PPO,
        name="",
        config=config,
        checkpoint_freq=5,
        keep_checkpoints_num=10,
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * iters,
            "episodes_total": config["train_batch_size"] * iters / steps,
        },
        # Order is important!
        callbacks=[
            DataLoggerCallback(config["env_config"]["base_config_path"]),
            TBXLoggerCallback(),
        ],
        # restore="",
    )

    ray.shutdown()
