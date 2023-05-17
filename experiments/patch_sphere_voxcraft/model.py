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

        self.input_layer = nn.Sequential(
            nn.Conv3d(len(self.materials), 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(1, 1, (5, 5, 5), (2, 2, 2), (2, 2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                int(np.prod([size // 2 for size in dimension_size])),
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
