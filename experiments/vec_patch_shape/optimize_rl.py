import ray
import numpy as np
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from experiments.vec_patch_shape.model import CustomPPO
from experiments.vec_patch_shape.utils import DataLoggerCallback
from experiments.vec_patch_shape.config import (
    config,
    iters,
)


if __name__ == "__main__":
    # 1GB heap memory, 1GB object store
    ray.init(_memory=1 * (10**9), object_store_memory=10**9)

    tune.run(
        CustomPPO,
        config=config,
        name="",
        checkpoint_freq=10,
        keep_checkpoints_num=int(np.ceil(iters / 10)),
        log_to_file=True,
        stop={
            "timesteps_total": config["train_batch_size"] * iters,
            "episodes_total": np.infty,
        },
        # Order is important!
        callbacks=[
            DataLoggerCallback(),
            TBXLoggerCallback(),
        ],
        restore="",
    )
    ray.shutdown()
