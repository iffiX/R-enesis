import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import *
from ray.rllib.models.torch.modules.relative_multi_head_attention import *
from ray.rllib.policy.sample_batch import SampleBatch
from renesis.utils.debug import print_model_size, enable_debugger


torch.set_printoptions(threshold=10000)


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

        # For a sequence of length 3 in the batch: <s> <some_token> <some_token2>
        # Suppose total time length T = 5
        # Since the right 2 tokens are padded with 0s
        # The sub mask is like:
        # [1, 0, 0, 0, 0]
        # [1, 1, 0, 0, 0]
        # [1, 1, 1, 0, 0]
        # [0, 0, 0, 0, 0]
        # [0, 0, 0, 0, 0]
        masks = []
        for l in lengths:
            sub_mask = sequence_mask(
                torch.tensor([Tau + 1 + j if j <= int(l) else 0 for j in range(T)]),
                maxlen=T,
                dtype=score.dtype,
            ).to(score.device)
            masks.append(sub_mask)
        mask_shape = list(score.shape)
        mask_shape[-1] = 1
        mask = torch.stack(masks, dim=0).view(mask_shape)

        masked_score = score * mask + 1e30 * (mask.float() - 1.0)
        wmat = nn.functional.softmax(masked_score, dim=2) * mask

        out = torch.einsum("bijh,bjhd->bihd", wmat, values)
        shape = list(out.shape)[:2] + [H * d]
        out = torch.reshape(out, shape)

        # output shape: [Batch, Time, Dim]
        # Set positions larger than input length to 0.
        for l in lengths:
            out[:, Tau + int(l) + 1 :] = 0

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
        dimension_size=None,
        materials=None,
        for_online_policy: bool = True,
    ):
        assert dimension_size is not None
        assert materials is not None
        assert gaussian_dim + voxel_dim == attention_dim
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)
        self.num_transformer_units = num_transformer_units
        self.input_gaussian_dim = observation_space.shape[0] - 1 - dimension_size**3
        self.gaussian_dim = gaussian_dim
        self.voxel_dim = voxel_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.memory = memory
        self.head_dim = head_dim
        self.max_seq_len = model_config["max_seq_len"]
        self.dimension_size = dimension_size
        self.materials = materials
        self.for_online_policy = for_online_policy

        self.input_gaussian_layer = SlimFC(
            in_size=self.input_gaussian_dim, out_size=gaussian_dim
        )
        self.input_voxel_layer = SlimFC(
            in_size=dimension_size**3 * len(self.materials),
            out_size=voxel_dim,
        )
        self.output_voxel_layer = SlimFC(
            in_size=self.attention_dim,
            out_size=dimension_size**3 * len(self.materials),
        )
        self._voxel_out = None
        self._voxel_predict_loss = 0

        self.action_out = nn.Sequential(
            SlimFC(
                in_size=self.attention_dim,
                out_size=self.attention_dim,
                activation_fn="relu",
            ),
            SlimFC(
                in_size=self.attention_dim, out_size=num_outputs, activation_fn=None
            ),
        )
        self.value_out = nn.Sequential(
            SlimFC(
                in_size=self.attention_dim,
                out_size=self.attention_dim,
                activation_fn="relu",
            ),
            SlimFC(in_size=self.attention_dim, out_size=1, activation_fn=None),
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

        # Setup trajectory views
        # t is different for evaulation and training! ???
        # for evaluation it starts at -1, for training it starts at 0 ???
        # self.view_requirements["custom_t"] = ViewRequirement(data_col=SampleBatch.T)
        # Use recorded t from environment instead

        self.view_requirements["custom_obs"] = ViewRequirement(
            data_col=SampleBatch.OBS,
            space=observation_space,
            shift=f"-{self.max_seq_len-1}:0",
        )
        self.view_requirements["custom_next_obs"] = ViewRequirement(
            data_col=SampleBatch.OBS,
            space=observation_space,
            shift=f"-{self.max_seq_len-2}:1",
            used_for_compute_actions=False,
        )
        # Setup memory views, (`memory-inference` x past memory outs).
        # if self.memory > 0:
        #     for i in range(self.num_transformer_units):
        #         space = Box(-1.0, 1.0, shape=(self.attention_dim,))
        #         self.view_requirements["state_in_{}".format(i)] = ViewRequirement(
        #             "state_out_{}".format(i),
        #             shift="-{}:-1".format(self.memory),
        #             # Repeat the incoming state every max-seq-len times.
        #             batch_repeat_value=self.max_seq_len,
        #             space=space,
        #         )
        #         self.view_requirements["state_out_{}".format(i)] = ViewRequirement(
        #             space=space, used_for_training=False
        #         )
        print_model_size(self)

    def forward(
        self,
        input_dict,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # shape [batch_size, max_seq_len, obs_dim]
        time, past_gaussians, past_voxels = self.unpack_observations(
            input_dict["custom_obs"]
        )

        # First observation corresponds to special start token
        past_gaussians = self.reorder_observation(past_gaussians, time)
        # Note: the last dimension is ordered by material first, then by voxel dimensions
        # shape [batch_size, max_seq_len, material_num * dimension_size ** 3]
        past_voxels_one_hot = self.to_one_hot_voxels(
            self.reorder_observation(past_voxels, time),
            time,
        )

        all_out = torch.cat(
            (
                self.input_gaussian_layer(past_gaussians),
                self.input_voxel_layer(past_voxels_one_hot),
            ),
            dim=2,
        )
        memory_outs = []
        for i in range(len(self.attention_layers)):
            # MHA layers which need memory passed in.
            if i % 2 == 0:
                all_out = self.attention_layers[i](
                    all_out,
                    lengths=time,
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
        voxel_mean = self.output_voxel_layer(all_out)
        self._voxel_out = voxel_mean

        # Use last for computing value output.
        last = time.to(dtype=torch.long).flatten()
        last_output = all_out[range(len(last)), last, :]
        self._value_out = self.value_out(last_output)

        if input_dict.get("return_voxel", False):
            out = voxel_mean
        else:
            out = self.action_out(last_output)
        return out, [torch.reshape(m, [-1, self.attention_dim]) for m in memory_outs]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])

    def metrics(self) -> Dict[str, TensorType]:
        return {"voxel_predict_loss": self._voxel_predict_loss}

    @override(ModelV2)
    def custom_loss(
        self, policy_loss: List[TensorType], loss_inputs: Dict[str, TensorType]
    ) -> List[TensorType]:
        next_time, _, next_past_voxels = self.unpack_observations(
            loss_inputs["custom_next_obs"]
        )

        next_past_voxels = self.reorder_observation(next_past_voxels, next_time)
        next_past_voxels_one_hot = self.to_one_hot_voxels(
            next_past_voxels, next_time
        ).to(policy_loss[0].device)
        voxel_predict_loss = torch.mean(
            torch.stack(
                [
                    nn.functional.mse_loss(
                        self._voxel_out[:, : int(t)],
                        next_past_voxels_one_hot[:, : int(t)],
                    )
                    for t in next_time
                ]
            )
        )
        self._voxel_predict_loss = float(voxel_predict_loss)
        # print(f"Voxel prediction loss: {voxel_predict_loss}")
        return [_loss + voxel_predict_loss for _loss in policy_loss]

    def unpack_observations(self, observations):
        # shape of observations
        # [batch_size, max_seq_len, 1 + gaussian_input_dim + dimension_size ** 3]
        # Unpack the last dim into time observation, gaussian observation and voxels observation
        return (
            observations[:, -1, 0],
            observations[:, :, 1 : 1 + self.input_gaussian_dim],
            observations[:, :, 1 + self.input_gaussian_dim :],
        )

    def reorder_observation(self, observation, time):
        # Since ray will add padding to the left, we have to cut the left
        # padding off and add it to the right side.

        # observation shape [batch_size, max_seq_len, ...]
        # T starts at 0 for first step.
        result = []
        for idx, t in enumerate(time):
            padding = observation[idx, : self.max_seq_len - int(t) - 1]
            real_observation = observation[idx, self.max_seq_len - int(t) - 1 :]
            # Add padding to the right.
            result.append(torch.cat([real_observation, padding], dim=0))

        return torch.stack(result)

    def to_one_hot_voxels(self, voxels, time=None):
        voxels_one_hot = torch.cat(
            [voxels == mat for mat in self.materials],
            dim=-1,
        ).to(dtype=torch.float32)

        if time is not None:
            # Fill remaining slots outside of time length as zero
            for idx, t in enumerate(time):
                voxels_one_hot[idx, int(t) + 1 :] = 0
        return voxels_one_hot


ModelCatalog.register_custom_model("actor_model", Actor)
