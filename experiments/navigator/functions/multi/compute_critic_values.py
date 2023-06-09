import shutil
import tqdm
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import *
from ray.rllib.models.torch.modules.relative_multi_head_attention import *
from ray.rllib.algorithms.ppo import PPO
from renesis.utils.debug import print_model_size, enable_debugger


torch.set_printoptions(threshold=10000, sci_mode=False)


class Actor(TorchModelV2, nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        *,
        hidden_dim: int = 256,
        max_steps: int = None,
        dimension_size=None,
        materials=None,
        normalize_mode: str = None,
        initial_std_bias_in_voxels: int = None,
    ):
        assert dimension_size is not None
        assert materials is not None
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.dimension_size = dimension_size
        self.materials = materials
        if initial_std_bias_in_voxels is not None and initial_std_bias_in_voxels > 0:
            self.initial_std_bias_in_voxels = initial_std_bias_in_voxels
            if normalize_mode == "clip":
                self.initial_std_bias = [
                    np.log(initial_std_bias_in_voxels / (size * 3) * 4)
                    for size in dimension_size
                ]
            elif normalize_mode == "clip1":
                self.initial_std_bias = [
                    np.log(initial_std_bias_in_voxels / (size * 3) * 2)
                    for size in dimension_size
                ]
            else:
                print(
                    f"Initial std bias not supported for normalize mode {normalize_mode}, use 0 by default"
                )
                self.initial_std_bias = [0, 0, 0]
        else:
            self.initial_std_bias_in_voxels = 0
            self.initial_std_bias = [0, 0, 0]

        self.input_layer = nn.Sequential(
            nn.Conv3d(len(self.materials), 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(1, 1, (5, 5, 5), (2, 2, 2), (2, 2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                int(np.prod([(size + 1) // 2 for size in dimension_size])),
                self.hidden_dim,
            ),
        )

        self.action_out = nn.Sequential(
            SlimFC(
                in_size=self.hidden_dim,
                out_size=self.hidden_dim,
                activation_fn="relu",
            ),
            SlimFC(
                in_size=self.hidden_dim,
                out_size=num_outputs,
                activation_fn=None,
            ),
        )
        self.value_out = nn.Sequential(
            SlimFC(
                in_size=self.hidden_dim,
                out_size=self.hidden_dim,
                activation_fn="relu",
            ),
            SlimFC(in_size=self.hidden_dim, out_size=1, activation_fn=None),
        )
        # Last value output.
        self._value_out = None
        print_model_size(self)

    def forward(
        self,
        input_dict,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        past_voxel = input_dict["obs"].reshape(
            (input_dict["obs"].shape[0],) + tuple(self.dimension_size)
        )
        past_voxel_one_hot = torch.stack(
            [past_voxel == mat for mat in self.materials],
            dim=1,
        ).to(dtype=torch.float32)
        out = self.input_layer(past_voxel_one_hot)
        self._value_out = self.value_out(out)
        action_out = self.action_out(out)
        offset = action_out.shape[-1] // 2
        for i in range(3):
            action_out[:, offset + i] += self.initial_std_bias[i]
        return action_out, []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])


class CustomPPO(PPO):
    _allow_unknown_configs = True


ModelCatalog.register_custom_model("actor_model", Actor)


import os
import ray
import pickle
import importlib.util
import torch as t
import numpy as np
from typing import List
from ray.rllib.algorithms.ppo import PPO
from experiments.navigator.trial import TrialRecord


def generate_critic_values(
    records: List[TrialRecord],
    experiment_name: str,
    save_file_name: str,
    restore_checkpoint: bool,
    resolution: int = 100,
):
    per_critic_result = []
    truncated_max_epoch = min(record.epochs[-1] for record in records)
    # epochs = list(np.round(np.linspace(1, truncated_max_epoch, resolution)))
    epochs = list(range(1, truncated_max_epoch + 1))
    for idx in range(len(records)):
        spec = importlib.util.spec_from_file_location(
            "config_mod",
            os.path.join(
                records[idx].code_dir, "experiments", experiment_name, "config.py"
            ),
        )
        config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_mod)
        algo = CustomPPO(config=config_mod.config)
        if restore_checkpoint:
            algo.restore(records[idx].checkpoints[-1])
        policy = algo.get_policy()
        model = policy.model.to("cuda:0")

        result = []
        print(f"Processing critic {idx}")
        for epoch in tqdm.tqdm(epochs):
            robots = []
            for record in records:
                with open(
                    os.path.join(
                        record.data_dir,
                        record.epoch_files[epoch].data_file_name,
                    ),
                    "rb",
                ) as file:
                    data = pickle.load(file)
                    for robot_data in data:
                        robots.append((robot_data["reward"], robot_data["voxels"]))
            obs = t.from_numpy(np.array([r[1] for r in robots])).to(
                dtype=t.float32, device="cuda:0"
            )
            _ = model(
                input_dict={"obs": obs},
            )
            # values and predicted_values shape: [len(records), batch_size]
            values = np.array([r[0] for r in robots]).reshape(len(records), -1)
            predicted_values = (
                model.value_function().detach().cpu().numpy().reshape(len(records), -1)
            )

            result.append(np.stack([values, predicted_values]))
        # np.stack(result) shape: [resolution, 2, len(records), batch_size)
        per_critic_result.append(np.stack(result))
        algo.stop()
        try:
            shutil.rmtree(algo.logdir)
        except:
            continue
    with open(f"generated_data/{save_file_name}.data", "wb") as file:
        # data shape [resolution, 2, len(records), batch_size)
        data = np.stack(per_critic_result, axis=2)
        print(data.shape)
        # epochs shape [resolution]
        pickle.dump((epochs, data), file)


def compute_critic_values(records: List[TrialRecord]):
    experiment_name = records[0].get_experiment_name()
    ray.init()
    generate_critic_values(records, experiment_name, f"{experiment_name}_trained", True)
    generate_critic_values(
        records, experiment_name, f"{experiment_name}_untrained", False
    )
    ray.shutdown()
