import shutil
import tqdm
import inspect
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
        anneal_func=None,
    ):
        assert dimension_size is not None
        assert materials is not None
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.anneal_func = anneal_func
        self.dimension_size = dimension_size
        self.materials = materials

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
        timesteps = input_dict["obs"][:, 0]
        past_voxel = input_dict["obs"][:, 1:].reshape(
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
        bias = self.anneal_func(timesteps).to(device=action_out.device).unsqueeze(1)
        action_out[:, offset : offset + 3] += bias
        return action_out, []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])


class CustomPPO(PPO):
    _allow_unknown_configs = True


import os
import ray
import pickle
import importlib.util
import numpy as np
from experiments.navigator.trial import TrialRecord
from renesis.env.vec_voxcraft import VoxcraftSingleRewardVectorizedPatchEnvironment


def debug_model_output(record: TrialRecord):
    ray.init()
    ModelCatalog.register_custom_model("actor_model", Actor)
    spec = importlib.util.spec_from_file_location(
        "config_mod",
        os.path.join(
            record.code_dir, "experiments", record.get_experiment_name(), "config.py"
        ),
    )
    config_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_mod)
    algo = CustomPPO(config=config_mod.config)
    print(record.checkpoints[-1])
    algo.restore(record.checkpoints[-1])
    policy = algo.get_policy()
    print(config_mod.config["env_config"]["base_config_path"])
    env = config_mod.config["env"](config_mod.config["env_config"])
    env.max_steps = 1
    env.set_timestep(policy.global_timestep // config_mod.envs)
    obs = env.vector_reset()

    for i in range(config_mod.steps):
        # Compute an action (`a`).
        a, state_out, *_ = policy.compute_actions_from_input_dict(
            input_dict={"obs": obs},
            explore=True,
        )
        obs, reward, done, _ = env.vector_step(a)
    algo.stop()
    try:
        shutil.rmtree(algo.logdir)
    except:
        pass
    ray.shutdown()
