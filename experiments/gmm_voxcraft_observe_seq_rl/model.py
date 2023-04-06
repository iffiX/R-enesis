import gym.spaces
import torch
from gym.spaces import Dict as DictSpace, Box as BoxSpace
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import *
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.models.torch.modules.relative_multi_head_attention import *
from renesis.utils.debug import print_model_size, enable_debugger


class RelativeMultiHeadAttentionUnevenSequence(RelativeMultiHeadAttention):
    def forward(
        self, inputs: TensorType, lengths: TensorType, memory: TensorType = None
    ) -> TensorType:
        # inputs shape: [Batch, Time, Dim]
        # lengths shape: [Batch]
        # memory shape: [Batch, Time, Dim]

        T = list(inputs.size())[1]  # length of segment (time)
        H = self._num_heads  # number of attention heads
        d = self._head_dim  # attention head dimension

        # Add previous memory chunk (as const, w/o gradient) to input.
        # Tau (number of (prev) time slices in each memory chunk).
        if memory is not None:
            Tau = list(memory.shape)[1]
            inputs = torch.cat((memory.detach(), inputs), dim=1)
        else:
            Tau = 0

        # Apply the Layer-Norm.
        if self._input_layernorm is not None:
            inputs = self._input_layernorm(inputs)

        qkv = self._qkv_layer(inputs)

        queries, keys, values = torch.chunk(input=qkv, chunks=3, dim=-1)
        # Cut out Tau memory timesteps from query.
        queries = queries[:, -T:]

        queries = torch.reshape(queries, [-1, T, H, d])
        keys = torch.reshape(keys, [-1, Tau + T, H, d])
        values = torch.reshape(values, [-1, Tau + T, H, d])

        R = self._pos_proj(self._rel_pos_embedding(Tau + T))
        R = torch.reshape(R, [Tau + T, H, d])

        # b=batch
        # i and j=time indices (i=max-timesteps (inputs); j=Tau memory space)
        # h=head
        # d=head-dim (over which we will reduce-sum)
        score = torch.einsum("bihd,bjhd->bijh", queries + self._uvar, keys)
        pos_score = torch.einsum("bihd,jhd->bijh", queries + self._vvar, R)
        score = score + self.rel_shift(pos_score)
        score = score / d**0.5

        # causal mask of the same length as the sequence

        # Note: Make the causal mask part corresponding to the left padding 0.
        masks = [
            sequence_mask(torch.arange(Tau + 1, Tau + T + 1), dtype=score.dtype).to(
                score.device
            )
            for _ in range(len(lengths))
        ]
        for idx, l in enumerate(lengths):
            masks[idx][:, Tau : Tau + T - int(l)] = 0
        mask = torch.stack(masks, dim=0).view(score.shape)

        masked_score = score * mask + 1e30 * (mask.float() - 1.0)
        wmat = nn.functional.softmax(masked_score, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.shape)[:2] + [H * d]
        out = torch.reshape(out, shape)

        return self._linear_layer(out)


class Actor(TorchModelV2, nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        *,
        num_transformer_units: int = 1,
        gaussian_dim: int = 16,
        voxel_dim: int = 48,
        attention_dim: int = 64,
        num_heads: int = 2,
        memory: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
        for_online_policy: bool = True,
    ):
        assert gaussian_dim + voxel_dim == attention_dim
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)
        observation_space = observation_space.original_space  # type: DictSpace
        self.num_transformer_units = num_transformer_units
        self.gaussian_dim = gaussian_dim
        self.voxel_dim = voxel_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory = memory
        self.head_dim = head_dim
        self.max_seq_len = model_config["max_seq_len"]
        self.for_online_policy = for_online_policy

        self.input_gaussian_layer = SlimFC(
            in_size=observation_space["gaussians"].shape[-1], out_size=gaussian_dim
        )
        self.input_voxel_layer = SlimFC(
            in_size=np.prod(observation_space["all_past_voxels"].shape[1:]),
            out_size=voxel_dim,
        )

        if for_online_policy:
            # output half for mean, half for sigma
            self.output_gaussian_layer = SlimFC(
                in_size=self.attention_dim,
                out_size=observation_space["gaussians"].shape[-1] * 2,
            )
        else:
            self.output_gaussian_layer = SlimFC(
                in_size=self.attention_dim,
                out_size=observation_space["gaussians"].shape[-1],
            )
        self.output_voxel_layer = SlimFC(
            in_size=self.attention_dim,
            out_size=np.prod(observation_space["all_past_voxels"].shape[1:]),
        )
        self._voxel_out = None

        self.values_out = SlimFC(
            in_size=self.attention_dim, out_size=1, activation_fn=None
        )
        # Last value output.
        self._value_out = None

        attention_layers = []
        # 2) Create L Transformer blocks according to [2].
        for i in range(self.num_transformer_units):
            # RelativeMultiHeadAttention part.
            MHA_layer = SkipConnection(
                RelativeMultiHeadAttentionUnevenSequence(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    input_layernorm=True,
                    output_activation=nn.ReLU,
                ),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            # Position-wise MultiLayerPerceptron part.
            E_layer = SkipConnection(
                nn.Sequential(
                    torch.nn.LayerNorm(self.attention_dim),
                    SlimFC(
                        in_size=self.attention_dim,
                        out_size=position_wise_mlp_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU,
                    ),
                    SlimFC(
                        in_size=position_wise_mlp_dim,
                        out_size=self.attention_dim,
                        use_bias=False,
                        activation_fn=nn.ReLU,
                    ),
                ),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias),
            )

            # Build a list of all attanlayers in order.
            attention_layers.extend([MHA_layer, E_layer])

        # Create a Sequential such that all parameters inside the attention
        # layers are automatically registered with this top-level model.
        self.attention_layers = nn.Sequential(*attention_layers)
        print_model_size(self)

    @override(ModelV2)
    def forward(
        self,
        input_dict,
        state: List[TensorType],
        seq_lens: TensorType,
        return_all: bool = False,
    ) -> (TensorType, List[TensorType]):
        # shape [batch_size, max_gaussian_num, gaussian_input_dim]
        gaussians = input_dict["obs"]["gaussians"]
        # shape [batch_size, 1]
        gaussian_num = input_dict["obs"]["gaussian_num"]
        # shape [batch_size, max_gaussian_num, dimension_size, dimension_size, dimension_size, material_num]
        all_past_voxels = input_dict["obs"]["all_past_voxels"]
        # enable_debugger()
        all_out = torch.cat(
            (
                self.input_gaussian_layer(gaussians),
                self.input_voxel_layer(torch.flatten(all_past_voxels, start_dim=2)),
            ),
            dim=2,
        )
        memory_outs = []
        for i in range(len(self.attention_layers)):
            # MHA layers which need memory passed in.
            if i % 2 == 0:
                all_out = self.attention_layers[i](
                    all_out,
                    lengths=gaussian_num,
                    memory=state[i // 2] if self.memory > 0 else None,
                )
            # Either self.linear_layer (initial obs -> attn. dim layer) or
            # MultiLayerPerceptrons. The output of these layers is always the
            # memory for the next forward pass.
            else:
                all_out = self.attention_layers[i](all_out)
                memory_outs.append(all_out)

        # Discard last output (not needed as a memory since it's the last
        # layer).
        memory_outs = memory_outs[:-1]

        # all_out shape [batch_size, max_gaussian_num, attention_dim]
        gaussian_logits = self.output_gaussian_layer(all_out)
        voxel_mean = self.output_voxel_layer(all_out)
        self._voxel_out = voxel_mean

        # Use last for computing value output.
        self._value_out = self.values_out(all_out[:, -1, :])

        if return_all:
            out = (gaussian_logits, voxel_mean)
        else:
            out = gaussian_logits[:, -1, :]
        return out, [torch.reshape(m, [-1, self.attention_dim]) for m in memory_outs]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])

    @override(ModelV2)
    def custom_loss(
        self, policy_loss: List[TensorType], loss_inputs: Dict[str, TensorType]
    ) -> List[TensorType]:
        obs_predict_loss = torch.mean(
            nn.functional.mse_loss(
                self._voxel_out, torch.flatten(loss_inputs["next_obs"], dim=2)
            )
        )
        return [_loss + obs_predict_loss for _loss in policy_loss]


ModelCatalog.register_custom_model("actor_model", Actor)
