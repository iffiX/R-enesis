import os
import gym
import numpy as np
from torch import nn
from collections import defaultdict
from typing import List, Dict, Tuple
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.utils import get_activation_fn
from renesis.env.voxcraft import VoxcraftGrowthEnvironment
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer


class Actor(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not num_outputs:
            num_outputs = np.prod(action_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation", "relu")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation", "relu"), framework="torch"
        )

        layers = []
        (x_size, y_size, z_size, in_channels) = obs_space.shape
        in_size = (x_size, y_size, z_size)
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = self.get_padding_and_output_size(
                in_size, kernel, stride
            )
            layers.append(nn.Conv3d(in_channels, out_channels, kernel, stride, padding))
            layers.append(get_activation_fn(activation, framework="torch")())
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        out_channels = out_channels if post_fcnet_hiddens else num_outputs
        # Usually we just need to automatically determine the last kernel size
        # throw an error if user manually specified a wrong kernel size
        if kernel is not None:
            assert kernel == in_size and stride == (
                1,
                1,
                1,
            ), " Please adjust your Conv3D stack such that the output is of shape [B, C, 1, 1, 1]"

        layers.append(nn.Conv3d(in_channels, out_channels, in_size, (1, 1, 1)))
        layers.append(get_activation_fn(activation, framework="torch")())
        layers.append(nn.Flatten())

        # Add (optional) post-fc-stack after last Conv3D layer.
        layer_sizes = post_fcnet_hiddens[:-1] + (
            [num_outputs] if post_fcnet_hiddens else []
        )
        for i, out_size in enumerate(layer_sizes):
            layers.append(
                SlimFC(
                    in_size=out_channels,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation,
                    initializer=normc_initializer(1.0),
                )
            )
            out_channels = out_size

        self.layers = nn.Sequential(*layers)

        self.value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
        )

        self._action = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        input_features = input_dict["obs"].float()

        # Permute dimensions
        # data comes in as [B, x, y, z, channels]
        # and comes out as [B, channels, x, y, z]
        input_features = input_features.permute(0, 4, 1, 2, 3)

        out = self.layers(input_features)
        self._action = out
        return out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self.value_branch(self._action).squeeze(1)

    @staticmethod
    def get_padding_and_output_size(
        in_size: Tuple[int, int, int],
        filter_size: Tuple[int, int, int],
        stride_size: Tuple[int, int, int],
    ) -> (Tuple[int, int, int, int, int, int], Tuple[int, int, int]):
        """Note: Padding is added to match TF conv2d `same` padding. See
        www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

        Args:
            in_size: input size in x, y, z order
            stride_size: stride size in x, y, z order
            filter_size: filter size in x, y, z order

        Returns:
            padding: Padding size on left and right side of each dimension, in the order
                of x_l, x_r, y_l, y_r, z_l, z_r.
            output: Output shape of x, y, z after padding and convolution.
        """
        assert isinstance(in_size, tuple)
        assert isinstance(filter_size, tuple)
        assert isinstance(stride_size, tuple)
        padding, output = (), ()

        for i in range(3):
            pad_size = filter_size[i] // 2
            out_size = int(
                np.floor(
                    float(in_size[i] + 2 * pad_size - filter_size[i])
                    / float(stride_size[i])
                    + 1
                )
            )
            padding = padding + (pad_size,)
            output = output + (out_size,)
        return padding, output


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = defaultdict(list)
        # episode.user_data = {"final": {}, "running": defaultdict(list)}

    # def on_episode_step(self, *, worker, base_env, episode, **kwargs):
    #     # Running metrics -> keep all values
    #     # Final metrics -> only keep the current value
    #     for data_type, data_subset in episode.user_data.items():
    #         data = episode.last_info_for().get("data", {}).get(data_type)
    #         for name, value in data.items():
    #             if data_type == "running":
    #                 data_subset[name].append(value)
    #             else:
    #                 data_subset[name] = value
    #
    #     # Arbitrary episode media
    #     media = episode.last_info_for().get("media", {})
    #     for name, value in media.items():
    #         episode.media["episode_data"][name].append(value)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Render the robot simulation
        env = base_env.get_sub_environments()[0]  # type: VoxcraftGrowthEnvironment
        renderer = VXHistoryRenderer(
            history=env.robot_sim_history, width=640, height=480
        )
        renderer.render()
        frames = renderer.get_frames()
        episode.media["episode_data"]["frames"] = frames
        episode.media["episode_data"]["robot"] = env.robot
        episode.media["episode_data"]["robot_sim_history"] = env.robot_sim_history

        # Summary writer requires video to be in (N, T, C, H, W) shape

        # See https://tensorboardx.readthedocs.io/en/latest/tensorboard.html?
        # highlight=add_video#tensorboardX.SummaryWriter.add_video

        # See https://github.com/ray-project/ray/blob/
        # 0452a3a435e023eada85f670e70ffef02ceb5943/python/ray/tune/logger.py#L212

        # T H W C to T C H W, then add batch dimension
        # episode.custom_metrics["frames"] = np.expand_dims(
        #     np.transpose(frames, axes=(0, 3, 1, 2)), 0
        # )


class DataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = os.path.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        if "episode_data" not in result["episode_media"]:
            return

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        num_episodes = result["episodes_this_iter"]
        data = result["episode_media"]["episode_data"]
        episode_data = data[-num_episodes:]

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"]["episode_data"]
            episode_data += data[-num_episodes:]

        # Save a video only for the last episode in the list
        path = os.path.join(self._trial_local_dir[trial], f"rendered_{step:08d}.gif")
        print(f"Saving rendered results to {path}")
        wait = create_video_subproc(
            [f for f in episode_data[-1]["frames"]],
            path=self._trial_local_dir[trial],
            filename=f"rendered_{step:08d}",
            extension=".gif",
        )
        path = os.path.join(self._trial_local_dir[trial], f"robot-{step:08d}.vxd")
        with open(path, "w") as file:
            print(f"Saving simulation config to {path}")
            file.write(episode_data[-1]["robot"])
        path = os.path.join(self._trial_local_dir[trial], f"run-{step:08d}.history")
        with open(path, "w") as file:
            print(f"Saving history to {path}")
            file.write(episode_data[-1]["robot_sim_history"])
        wait()
        print("Saving completed")


ModelCatalog.register_custom_model("actor_model", Actor)
