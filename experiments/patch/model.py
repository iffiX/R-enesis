import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import *
from ray.rllib.models.torch.modules.relative_multi_head_attention import *
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPO
from renesis.utils.debug import print_model_size, enable_debugger


torch.set_printoptions(threshold=10000, sci_mode=False)


# class ActorSampling(StochasticSampling):
#     def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
#         raise NotImplementedError("Not implemented for Tensorflow")
#
#     def _get_torch_exploration_action(
#         self,
#         action_dist,
#         timestep,
#         explore,
#     ):
#         # Set last timestep or (if not given) increase by one.
#         self.last_timestep = (
#             timestep if timestep is not None else self.last_timestep + 1
#         )
#
#         # Apply exploration.
#         if explore:
#             action = action_dist.sample(timestep=float(self.last_timestep))
#             logp = action_dist.sampled_action_logp()
#
#         # No exploration -> Return deterministic actions.
#         else:
#             action = action_dist.deterministic_sample()
#             logp = torch.zeros_like(action_dist.sampled_action_logp())
#
#         return action, logp
#
#
# class ActorDistribution(TorchDiagGaussian):
#     @override(TorchDiagGaussian)
#     def __init__(
#         self,
#         inputs: List[TensorType],
#         model: TorchModelV2,
#         *,
#         action_space: Optional[gym.spaces.Space] = None,
#     ):
#         super().__init__(inputs, model)
#         mean, log_std = torch.chunk(self.inputs, 2, dim=1)
#         self.log_std = log_std
#         self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
#         # Remember to squeeze action samples in case action space is Box(shape)
#         self.zero_action_dim = action_space and action_space.shape == ()
#
#     @override(TorchDistributionWrapper)
#     def sample(self) -> TensorType:
#         sample = super().sample()
#         if self.zero_action_dim:
#             return torch.squeeze(sample, dim=-1)
#         return sample
#
#     @override(ActionDistribution)
#     def deterministic_sample(self) -> TensorType:
#         self.last_sample = self.dist.mean
#         return self.last_sample
#
#     @override(TorchDistributionWrapper)
#     def logp(self, actions: TensorType) -> TensorType:
#         return super().logp(actions).sum(-1)
#
#     @override(TorchDistributionWrapper)
#     def entropy(self) -> TensorType:
#         return super().entropy().sum(-1)
#
#     @override(TorchDistributionWrapper)
#     def kl(self, other: ActionDistribution) -> TensorType:
#         return super().kl(other).sum(-1)
#
#     @staticmethod
#     @override(ActionDistribution)
#     def required_model_output_shape(
#         action_space: gym.Space, model_config: ModelConfigDict
#     ) -> Union[int, np.ndarray]:
#         return np.prod(action_space.shape, dtype=np.int32)


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
            self.initial_std_bias = np.log(
                initial_std_bias_in_voxels / (dimension_size * 3) * 4
            )
        elif normalize_mode == "clip1":
            self.initial_std_bias = np.log(
                initial_std_bias_in_voxels / (dimension_size * 3) * 2
            )
        else:
            print(
                f"Initial std bias not supported for normalize mode {normalize_mode}, use 0 by default"
            )
            self.initial_std_bias = 0

        self.input_layer = nn.Sequential(
            nn.Conv3d(len(self.materials), 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(1, 1, (5, 5, 5), (2, 2, 2), (2, 2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((dimension_size // 2) ** 3, self.hidden_dim),
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
        # self.view_requirements["history_obs"] = ViewRequirement(
        #     data_col=SampleBatch.OBS,
        #     space=action_space,
        #     shift=f"-{self.max_steps-1}:0",
        #     used_for_compute_actions=False,
        # )
        print_model_size(self)

    def forward(
        self,
        input_dict,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        past_voxel = input_dict["obs"].reshape(
            (input_dict["obs"].shape[0],) + (self.dimension_size,) * 3
        )
        past_voxel_one_hot = torch.stack(
            [past_voxel == mat for mat in self.materials],
            dim=1,
        ).to(dtype=torch.float32)
        out = self.input_layer(past_voxel_one_hot)
        self._value_out = self.value_out(out)
        action_out = self.action_out(out)
        offset = action_out.shape[-1] // 2
        action_out[:, offset : offset + 3] += self.initial_std_bias
        return action_out, []

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        return torch.reshape(self._value_out, [-1])

    # @override(ModelV2)
    # def custom_loss(
    #     self, policy_loss: List[TensorType], loss_inputs: Dict[str, TensorType]
    # ) -> List[TensorType]:
    #     _, past_gaussians, past_voxels = (
    #         loss_inputs["history_obs"][:, -1, 0],
    #         loss_inputs["history_obs"][:, :, 1 : 1 + self.input_gaussian_dim],
    #         loss_inputs["history_obs"][:, :, 1 + self.input_gaussian_dim :],
    #     )
    #     B, T = past_gaussians.shape[:2]
    #     past_voxels = past_voxels.reshape((B * T,) + (self.dimension_size,) * 3)
    #     past_voxels_one_hot = torch.stack(
    #         [past_voxels == mat for mat in self.materials],
    #         dim=1,
    #     ).to(dtype=torch.float32)
    #     history_out = torch.cat(
    #         (
    #             self.input_gaussian_layer(past_gaussians),
    #             self.input_voxel_layer(past_voxels_one_hot).reshape(
    #                 B, T, self.hidden_dim // 2
    #             ),
    #         ),
    #         dim=-1,
    #     )
    #     actions = self.action_out(history_out)
    #     mean = torch.chunk(actions, 2, dim=-1)[0]
    #     position_episode_means = torch.mean(mean[:, :, :3], dim=1, keepdim=True)
    #     material_episode_means = torch.mean(mean[:, :, 3:], dim=1, keepdim=True)
    #     position_mean_loss = -((mean[:, :, :3] - position_episode_means) ** 2 - 0.5)
    #     material_mean_loss = -((mean[:, :, 3:] - material_episode_means) ** 2 - 0.5)
    #     mean_bound_loss = torch.abs(mean) - 1
    #     position_mean_loss = torch.mean(
    #         torch.where(
    #             position_mean_loss > 0,
    #             position_mean_loss,
    #             torch.zeros_like(position_mean_loss),
    #         )
    #     )
    #     material_mean_loss = torch.mean(
    #         torch.where(
    #             material_mean_loss > 0,
    #             material_mean_loss,
    #             torch.zeros_like(material_mean_loss),
    #         )
    #     )
    #     mean_bound_loss = torch.mean(
    #         torch.where(
    #             mean_bound_loss > 0, mean_bound_loss, torch.zeros_like(mean_bound_loss)
    #         )
    #     )
    #     print(
    #         f"Position mean loss: {position_mean_loss}, "
    #         f"Material mean loss: {material_mean_loss}, "
    #         f"Mean bound loss: {mean_bound_loss}"
    #     )
    #     return [
    #         loss + position_mean_loss + material_mean_loss + mean_bound_loss
    #         for loss in policy_loss
    #     ]


class CustomPPO(PPO):
    _allow_unknown_configs = True


ModelCatalog.register_custom_model("actor_model", Actor)
