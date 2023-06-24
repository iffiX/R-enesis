import ray
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from experiments.vec_patch_voxcraft.model import CustomPPO
from experiments.vec_patch_voxcraft.utils import DataLoggerCallback
from experiments.vec_patch_voxcraft.config import (
    config,
    iters,
)


if __name__ == "__main__":
    ray.init()
    tune.run(
        CustomPPO,
        config=config,
        name="",
        checkpoint_freq=10,
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * iters,
            "episodes_total": np.infty,
        },
        # Order is important!
        callbacks=[
            DataLoggerCallback(config["env_config"]["base_config_path"]),
            TBXLoggerCallback(),
        ],
        restore="",
    )
    ray.shutdown()
