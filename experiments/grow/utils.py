import os
import gym
import numpy as np
from torch import nn
from typing import List, Dict, Tuple
from ray.tune import Callback
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
from renesis.utils.sys_debug import print_model_size
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer


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
    except:
        print("Exception occurred, no frames produced")
        print("History:")
        print(history)
        return None


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
        shared_base_output_channels = out_channels
        self.shared_base_layers = nn.Sequential(*layers)

        action_layers = []
        for i, out_size in enumerate(layer_sizes):
            action_layers.append(
                SlimFC(
                    in_size=out_channels,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation
                    if i < len(layer_sizes) - 1
                    else None,
                    initializer=normc_initializer(1.0),
                )
            )
            out_channels = out_size
        self.action_layers = nn.Sequential(*action_layers)

        self.value_layers = nn.Sequential(
            post_fcnet_activation(),
            SlimFC(
                shared_base_output_channels,
                256,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            ),
            SlimFC(256, 128, initializer=normc_initializer(0.01), activation_fn=None),
            SlimFC(128, 1, initializer=normc_initializer(0.01), activation_fn=None),
        )

        self._shared_base_output = None
        print_model_size(self)

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

        self._shared_base_output = self.shared_base_layers(input_features)
        action = self.action_layers(self._shared_base_output)
        return action, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self.value_layers(self._shared_base_output).squeeze(1)

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
        episode.media["episode_data"] = {}

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Render the robot simulation
        env = base_env.vector_env  # type: VoxcraftGrowthEnvironment
        robot = env.previous_best_robots[0]
        history = env.previous_best_sim_histories[0]
        episode.media["episode_data"]["robot"] = robot
        episode.media["episode_data"]["robot_sim_history"] = history

    def on_train_result(self, *, algorithm, result, trainer, **kwargs,) -> None:
        num_episodes = result["episodes_this_iter"]
        data = result["episode_media"].get("episode_data", [])
        episode_data = data[-num_episodes:]

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", [])
            episode_data += data[-num_episodes:]

        # Summary writer requires video to be in (N, T, C, H, W) shape

        # See https://tensorboardx.readthedocs.io/en/latest/tensorboard.html?
        # highlight=add_video#tensorboardX.SummaryWriter.add_video

        # See https://github.com/ray-project/ray/blob/
        # 0452a3a435e023eada85f670e70ffef02ceb5943/python/ray/tune/logger.py#L212

        # Disabled due to OOM issues
        # T H W C to T C H W, then add batch dimension
        # if len(data) > 0:
        #     # Only render the last episode
        #     history = episode_data[-1]["robot_sim_history"]
        #     frames = render(history)
        #     if frames is not None:
        #         result["custom_metrics"].update(
        #             {
        #                 "video": np.expand_dims(
        #                     np.transpose(frames, (0, 3, 1, 2),), axis=0,
        #                 )
        #             }
        #         )


class DataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = os.path.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        num_episodes = result["episodes_this_iter"]

        data = result["episode_media"].get("episode_data", [])
        episode_data = data[-num_episodes:]

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", [])
            episode_data += data[-num_episodes:]

        if len(episode_data) > 0:
            # Save a video only for the last episode in the list
            robot = episode_data[-1]["robot"]
            history = episode_data[-1]["robot_sim_history"]
            frames = render(history)
            if frames is not None:
                path = os.path.join(
                    self._trial_local_dir[trial], f"rendered_{step:08d}.gif"
                )
                print(f"Saving rendered results to {path}")
                wait = create_video_subproc(
                    [f for f in frames],
                    path=self._trial_local_dir[trial],
                    filename=f"rendered_{step:08d}",
                    extension=".gif",
                )
                path = os.path.join(
                    self._trial_local_dir[trial], f"robot-{step:08d}.vxd"
                )
                with open(path, "w") as file:
                    print(f"Saving robot to {path}")
                    file.write(robot)
                path = os.path.join(
                    self._trial_local_dir[trial], f"run-{step:08d}.history"
                )
                with open(path, "w") as file:
                    print(f"Saving history to {path}")
                    file.write(history)
                wait()
        # Clear results to reduce load on checkpointing
        print("Saving completed")


class CleaningCallback1(Callback):
    def on_trial_result(self, iteration: int, trials, trial, result, **info):
        result["episode_media"] = {}
        if "evaluation" in result:
            result["evaluation"]["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}
        print("Cleaning 1 completed")


class CleaningCallback2(Callback):
    def on_trial_result(self, iteration: int, trials, trial, result, **info):
        result["custom_metrics"] = {}
        if "evaluation" in result:
            result["evaluation"]["custom_metrics"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["custom_metrics"] = {}
        print("Cleaning 2 completed")


ModelCatalog.register_custom_model("actor_model", Actor)
