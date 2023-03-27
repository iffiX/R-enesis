import os
import pickle
import graphviz
import gym.spaces
import math as m
import numpy as np
import torch as t
from torch import nn
from typing import List, Dict
from gym.spaces import Dict as DictSpace
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.rllib.models.torch.misc import (
    normc_initializer,
    SlimFC,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDiagGaussian,
    TorchDistributionWrapper,
)
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from renesis.env_model.cppn import CPPN
from renesis.utils.debug import enable_debugger
import renesis.utils.debug as debug
from renesis.utils.sys_debug import print_model_size
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer
from .gat import GAT, add_reflexive_edges
from ray.rllib.utils.torch_utils import FLOAT_MIN

t.set_printoptions(threshold=10000)


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


def make_mlp(sizes, squeeze_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if squeeze_last:
        layers.append(Squeeze())
    return nn.Sequential(*layers)


def safe_kl_for_categorical(cate1: TorchCategorical, cate2: TorchCategorical):
    probs1 = cate1.dist.probs
    probs2 = cate2.dist.probs

    # prevent generating nan values in probs1 * t.log(probs1 / probs2)
    # which will cause nan grads
    invalid_probs1 = probs1 < 1e-2
    probs1 = t.masked_fill(probs1, invalid_probs1, 1e-2)
    invalid_probs2 = probs2 < 1e-2
    probs2 = t.masked_fill(probs2, invalid_probs2, 1e-2)

    kl = t.masked_fill(
        probs1 * t.log(probs1 / probs2), t.logical_or(invalid_probs1, invalid_probs2), 0
    )
    return kl.sum(dim=1)


def compute_temperature(initial_temperature, timestep, exploration_timesteps):
    return max(
        1,
        initial_temperature
        * (
            0.99
            ** (timestep * m.log(1 / initial_temperature, 0.99) / exploration_timesteps)
        ),
    )


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


class Mean(nn.Module):
    def forward(self, x):
        return t.mean(x, dim=1)


# class ActorSampling(StochasticSampling):
#     def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
#         raise NotImplementedError("Not implemented for Tensorflow")
#
#     def _get_torch_exploration_action(
#         self, action_dist, timestep, explore,
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
#             logp = t.zeros_like(action_dist.sampled_action_logp())
#
#         return action, logp


class ActorDistribution(TorchDistributionWrapper):
    """
    Action distribution

    P(source_node, target_node, target_function, has_edge, weight | X)
    = P(source_node | X) * P(target_node | source_node, X)
      * P(target_function | source_node, target_node)
      * P(has_edge | source_node, target_node)
      * P(weight | source_node, target_node)
    """

    @staticmethod
    @override(TorchDistributionWrapper)
    def required_model_output_shape(
        action_space: gym.spaces.Space, model_config: ModelConfigDict
    ):
        node_num = int(action_space.high[0]) + 1

        # node ranks is concatenated before embeddings
        return np.array(
            [node_num, 1 + model_config["custom_model_config"]["output_feature_num"]]
        )

    @override(TorchDistributionWrapper)
    def deterministic_sample(self):
        src_dist = self.source_node_distribution()
        src_node = src_dist.deterministic_sample()
        tar_dist = self.target_node_distribution(src_node)
        tar_node = tar_dist.deterministic_sample()
        tar_func_dist = self.target_function_distribution(src_node, tar_node)
        tar_func = tar_func_dist.deterministic_sample()
        has_edge_dist = self.has_edge_distribution(src_node, tar_node)
        has_edge = has_edge_dist.deterministic_sample()
        weight_dist = self.weight_distribution(src_node, tar_node)
        weight = weight_dist.deterministic_sample().squeeze(-1)

        self._action_logp = (
            src_dist.logp(src_node)
            + tar_dist.logp(tar_node)
            + tar_func_dist.logp(tar_func)
            + has_edge_dist.logp(has_edge)
            + weight_dist.logp(weight)
        )

        # Return the actions. Match the order of flattened CPPN action space
        return t.stack((src_node, tar_node, tar_func, has_edge, weight), dim=1)

    @override(TorchDistributionWrapper)
    def sample(self, timestep=None):
        src_dist = self.source_node_distribution(timestep=timestep)
        src_node = src_dist.sample()
        tar_dist = self.target_node_distribution(src_node, timestep=timestep)
        tar_node = tar_dist.sample()
        tar_func_dist = self.target_function_distribution(
            src_node, tar_node, timestep=timestep
        )
        tar_func = tar_func_dist.sample()
        has_edge_dist = self.has_edge_distribution(
            src_node, tar_node, timestep=timestep
        )
        has_edge = has_edge_dist.sample()
        weight_dist = self.weight_distribution(src_node, tar_node)
        weight = weight_dist.sample().squeeze(-1)

        self._action_logp = (
            src_dist.logp(src_node)
            + tar_dist.logp(tar_node)
            + tar_func_dist.logp(tar_func)
            + has_edge_dist.logp(has_edge)
            + weight_dist.logp(weight)
        )

        # Return the action tuple.
        return t.stack((src_node, tar_node, tar_func, has_edge, weight), dim=1)

    @override(TorchDistributionWrapper)
    def logp(self, actions):
        src_node = actions[:, 0].to(dtype=t.int64)
        tar_node = actions[:, 1].to(dtype=t.int64)
        tar_func = actions[:, 2].to(dtype=t.int64)
        has_edge = actions[:, 3].to(dtype=t.int64)
        weight = actions[:, 4].float()

        src_dist = self.source_node_distribution()
        tar_dist = self.target_node_distribution(src_node)
        tar_func_dist = self.target_function_distribution(src_node, tar_node)
        has_edge_dist = self.has_edge_distribution(src_node, tar_node)
        weight_dist = self.weight_distribution(src_node, tar_node)

        return (
            src_dist.logp(src_node)
            + tar_dist.logp(tar_node)
            + tar_func_dist.logp(tar_func)
            + has_edge_dist.logp(has_edge)
            + weight_dist.logp(weight.unsqueeze(-1))
        )

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self):
        return t.exp(self._action_logp)

    @override(TorchDistributionWrapper)
    def entropy(self):
        # return t.tensor([0.0], device=self.inputs.device)
        src_dist = self.source_node_distribution()
        src_node = src_dist.sample()
        tar_dist = self.target_node_distribution(src_node)
        tar_node = tar_dist.sample()
        tar_func_dist = self.target_function_distribution(src_node, tar_node)
        has_edge_dist = self.has_edge_distribution(src_node, tar_node)
        weight_dist = self.weight_distribution(src_node, tar_node)
        return (
            src_dist.entropy()
            + tar_dist.entropy()
            + tar_func_dist.entropy()
            + has_edge_dist.entropy()
            + weight_dist.entropy()
        )

    @override(TorchDistributionWrapper)
    def kl(self, other: "ActorDistribution"):
        # return t.tensor([0.0], device=self.inputs.device)
        src_dist = self.source_node_distribution()
        src_terms = safe_kl_for_categorical(src_dist, other.source_node_distribution())

        src_node = src_dist.sample()
        tar_dist = self.target_node_distribution(src_node)
        tar_terms = safe_kl_for_categorical(
            tar_dist, other.target_node_distribution(src_node)
        )

        tar_node = tar_dist.sample()

        tar_func_terms = self.target_function_distribution(src_node, tar_node).kl(
            other.target_function_distribution(src_node, tar_node)
        )
        has_edge_terms = self.has_edge_distribution(src_node, tar_node).kl(
            other.has_edge_distribution(src_node, tar_node)
        )
        weight_terms = self.weight_distribution(src_node, tar_node).kl(
            other.weight_distribution(src_node, tar_node)
        )

        return src_terms + tar_terms + tar_func_terms + has_edge_terms + weight_terms

    def source_node_distribution(self, timestep=None):
        # inputs shape: [batch_size, node_num, output_feature_num]
        # logits shape: [batch_size, node_num]
        logits = self.model.source_node_module(self.inputs[:, :, 1:])
        node_ranks = self.inputs[:, :, 0].detach()

        # source_node_masks shape: [batch_size, node_num]
        source_node_masks = t.cat(
            [
                t.from_numpy(CPPN.get_source_node_mask(nr)).unsqueeze(0).bool()
                for nr in node_ranks.cpu().numpy()
            ]
        ).to(logits.device)

        # Prevent ray initialization error
        if not t.all(source_node_masks == False):
            logits = logits + t.clamp(t.log(source_node_masks), min=FLOAT_MIN)
            # logits = t.masked_fill(logits, ~source_node_masks, -t.inf)

        # if timestep is not None:
        #     temperature = compute_temperature(
        #         self.model.model_config["custom_model_config"]["initial_temperature"],
        #         timestep,
        #         self.model.model_config["custom_model_config"]["exploration_timesteps"],
        #     )
        # else:
        #     temperature = 1.0
        # print(f"Logits: {logits}, temp: {temperature}")
        # dist = TorchCategorical(logits, temperature=temperature)

        dist = TorchCategorical(logits)
        return dist

    def target_node_distribution(self, source_node, timestep=None):
        # src_node_embedding shape: [batch_size, 1, output_feature_num]
        batch_size = self.inputs.shape[0]
        node_num = self.inputs.shape[1]

        src_node_embedding = self.inputs[:, :, 1:][
            range(batch_size), source_node
        ].unsqueeze(1)

        node_ranks = self.inputs[:, :, 0].detach()

        # inputs shape: [batch_size, node_num, 2 * output_feature_num]
        inputs = t.cat(
            (src_node_embedding.repeat(1, node_num, 1), self.inputs[:, :, 1:]), dim=2
        )

        # logits shape: [batch_size, node_num]
        logits = self.model.target_node_module(inputs)

        # target_node_masks shape: [batch_size, node_num]
        target_node_masks = t.cat(
            [
                t.from_numpy(CPPN.get_target_node_mask(src, nr)).unsqueeze(0).bool()
                for src, nr in zip(source_node, node_ranks.cpu().numpy())
            ]
        ).to(logits.device)

        # Prevent ray initialization error
        if not t.all(target_node_masks == False):
            logits = logits + t.clamp(t.log(target_node_masks), min=FLOAT_MIN)
            # logits = t.masked_fill(logits, ~target_node_masks, -t.inf)

        # if timestep is not None:
        #     temperature = compute_temperature(
        #         self.model.model_config["custom_model_config"]["initial_temperature"],
        #         timestep,
        #         self.model.model_config["custom_model_config"]["exploration_timesteps"],
        #     )
        # else:
        #     temperature = 1.0

        # dist = TorchCategorical(logits, temperature=temperature)

        dist = TorchCategorical(logits)
        return dist

    def target_function_distribution(self, source_node, target_node, timestep=None):
        # tar_node_embedding shape: [batch_size, output_feature_num]
        batch_size = self.inputs.shape[0]
        src_node_embedding = self.inputs[:, :, 1:][range(batch_size), source_node]
        tar_node_embedding = self.inputs[:, :, 1:][range(batch_size), target_node]

        # logits shape: [batch_size, function_num]
        logits = self.model.target_function_module(
            t.cat((src_node_embedding, tar_node_embedding), dim=1)
        )

        # if timestep is not None:
        #     temperature = compute_temperature(
        #         self.model.model_config["custom_model_config"]["initial_temperature"],
        #         timestep,
        #         self.model.model_config["custom_model_config"]["exploration_timesteps"],
        #     )
        # else:
        #     temperature = 1.0

        # dist = TorchCategorical(logits, temperature=temperature)

        dist = TorchCategorical(logits)
        return dist

    def has_edge_distribution(self, source_node, target_node, timestep=None):
        batch_size = self.inputs.shape[0]

        # src_node_embedding and tar_node_embedding shape:
        # [batch_size, output_feature_num]
        src_node_embedding = self.inputs[:, :, 1:][range(batch_size), source_node]
        tar_node_embedding = self.inputs[:, :, 1:][range(batch_size), target_node]

        # logits shape: [batch_size, 2]
        # Or use bernoulli distribution here, but we need to implement it
        logits = self.model.has_edge_module(
            t.cat((src_node_embedding, tar_node_embedding), dim=1)
        )

        # if timestep is not None:
        #     temperature = compute_temperature(
        #         self.model.model_config["custom_model_config"]["initial_temperature"],
        #         timestep,
        #         self.model.model_config["custom_model_config"]["exploration_timesteps"],
        #     )
        # else:
        #     temperature = 1.0

        # dist = TorchCategorical(logits, temperature=temperature)

        dist = TorchCategorical(logits)
        return dist

    def weight_distribution(self, source_node, target_node):
        batch_size = self.inputs.shape[0]

        # src_node_embedding and tar_node_embedding shape:
        # [batch_size, output_feature_num]
        src_node_embedding = self.inputs[:, :, 1:][range(batch_size), source_node]
        tar_node_embedding = self.inputs[:, :, 1:][range(batch_size), target_node]

        # param shape: [batch_size, 2], each row is mean and log std of gaussian
        param = self.model.weight_module(
            t.cat((src_node_embedding, tar_node_embedding), dim=1)
        )

        dist = TorchDiagGaussian(param, model=None)
        return dist


class Actor(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: DictSpace,
        action_space: DictSpace,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # if model_config["custom_model_config"]["debug"]:
        #     enable_debugger(
        #         model_config["custom_model_config"]["debug_ip"],
        #         model_config["custom_model_config"]["debug_port"],
        #     )
        print(
            f"Initializing model with config: \n{model_config['custom_model_config']}"
        )
        self.gat = GAT(
            input_feature_num=model_config["custom_model_config"]["input_feature_num"],
            hidden_feature_num=model_config["custom_model_config"][
                "hidden_feature_num"
            ],
            output_feature_num=model_config["custom_model_config"][
                "output_feature_num"
            ],
            layer_num=model_config["custom_model_config"]["layer_num"],
            head_num=model_config["custom_model_config"]["head_num"],
            dropout_prob=model_config["custom_model_config"]["dropout_prob"],
        )

        output_feature_num = model_config["custom_model_config"]["output_feature_num"]

        self.value_net = nn.Sequential(
            Mean(),
            SlimFC(
                output_feature_num,
                1,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            ),
        )

        self.source_node_module = make_mlp((output_feature_num, 1), squeeze_last=True)
        self.target_node_module = make_mlp(
            (output_feature_num * 2, 1), squeeze_last=True,
        )
        self.target_function_module = make_mlp(
            (
                output_feature_num * 2,
                model_config["custom_model_config"]["target_function_num"],
            )
        )
        self.has_edge_module = make_mlp((output_feature_num * 2, 2))
        self.weight_module = make_mlp((output_feature_num * 2, 2))
        self._shared_base_output = None
        print_model_size(self)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # shape [batch_size, node_num]
        node_ranks = input_dict["obs"]["node_ranks"]
        # shape [batch_size, node_num, node_feature_num]
        nodes = input_dict["obs"]["nodes"]

        # shape [batch_size, max_edge_num, 3]
        # padded to max length

        edges = input_dict["obs"]["edges"].values
        edge_num = input_dict["obs"]["edges"].lengths

        new_edges, new_edge_num, new_edge_weight = add_reflexive_edges(
            nodes.shape[1],
            edges[:, :, :2].to(t.int64),
            edge_num.to(t.int64),
            edges[:, :, 2],
        )

        # shape [batch_size, node_num, output_feature_num]
        output = self.gat(nodes, new_edges, new_edge_num, new_edge_weight)

        # for debugging
        # if t.any(t.isnan(output)):
        #     enable_debugger()
        #     x = self.gat(nodes, new_edges, new_edge_num, new_edge_weight)

        self._shared_base_output = output
        # action shape [batch_size, node_num, 1 + output_feature_num]
        return t.cat([node_ranks.unsqueeze(-1), output], dim=2), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        # shape [batch_size]
        result = self.value_net(self._shared_base_output).squeeze(1)
        return result


class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.media["episode_data"] = {}

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = base_env.vector_env  # type: VoxcraftGrowthEnvironment
        episode.media["episode_data"]["rewards"] = env.all_rewards_history
        episode.media["episode_data"]["best_reward"] = env.best_reward
        episode.media["episode_data"]["best_robot"] = env.best_finished_robot
        episode.media["episode_data"][
            "best_robot_sim_history"
        ] = env.best_finished_robot_sim_history
        episode.media["episode_data"][
            "best_robot_cppn_graphs"
        ] = env.best_finished_robot_state_data

    def on_train_result(self, *, algorithm, result, trainer, **kwargs,) -> None:
        # Remove non-evaluation data
        result["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", [])

            if len(data) > 0:
                # Aggregate results
                rewards = []
                best_reward = -np.inf
                best_robot = None
                best_robot_sim_history = None
                best_robot_cppn_graphs = None
                for episode_data in data:
                    rewards += episode_data["rewards"]
                    if episode_data["best_reward"] > best_reward:
                        best_reward = episode_data["best_reward"]
                        best_robot = episode_data["best_robot"]
                        best_robot_sim_history = episode_data["best_robot_sim_history"]
                        best_robot_cppn_graphs = episode_data["best_robot_cppn_graphs"]

                result["evaluation"]["episode_media"] = {
                    "episode_data": {
                        "rewards": rewards,
                        "best_reward": best_reward,
                        "best_robot": best_robot,
                        "best_robot_sim_history": best_robot_sim_history,
                        "best_robot_cppn_graphs": best_robot_cppn_graphs,
                    }
                }


class DataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = os.path.join(trial.logdir, "data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        iteration = result[TRAINING_ITERATION]
        if "evaluation" in result:
            data = result["evaluation"]["episode_media"].get("episode_data", None)
            result["evaluation"]["episode_media"] = {}
            if data:
                log_file = os.path.join(self._trial_local_dir[trial], "metric.data")
                metrics = []
                if os.path.exists(log_file):
                    with open(log_file, "rb") as file:
                        metrics = pickle.load(file)
                with open(log_file, "wb") as file:
                    history_best_reward = -np.inf
                    for history_metric in metrics:
                        if history_metric[0] > history_best_reward:
                            history_best_reward = history_metric[0]
                    metrics += [
                        (
                            max(history_best_reward, np.max(data["rewards"])),
                            np.max(data["rewards"]),
                            np.mean(data["rewards"]),
                            np.min(data["rewards"]),
                        )
                    ]
                    pickle.dump(metrics, file)

                print(f"Saving cppn graph")
                unpruned_graph, pruned_graph = data["best_robot_cppn_graphs"]
                g1 = graphviz.Source(unpruned_graph)
                g1.render(
                    filename=f"unpruned_it_{iteration}_rew_{data['best_reward']}",
                    directory=self._trial_local_dir[trial],
                    format="png",
                )
                g2 = graphviz.Source(pruned_graph)
                g2.render(
                    filename=f"pruned_it_{iteration}",
                    directory=self._trial_local_dir[trial],
                    format="png",
                )

                robot = data["best_robot"]
                history = data["best_robot_sim_history"]
                frames = render(history)

                if frames is not None:
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"rendered_it_{iteration}_rew_{data['best_reward']}.gif",
                    )
                    print(f"Saving rendered results to {path}")
                    wait = create_video_subproc(
                        [f for f in frames],
                        path=self._trial_local_dir[trial],
                        filename=f"rendered_it_{iteration}",
                        extension=".gif",
                    )
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"robot_it_{iteration}_rew_{data['best_reward']}.vxd",
                    )
                    with open(path, "w") as file:
                        print(f"Saving robot to {path}")
                        file.write(robot)
                    path = os.path.join(
                        self._trial_local_dir[trial],
                        f"run_it_{iteration}_rew_{data['best_reward']}.history",
                    )
                    with open(path, "w") as file:
                        print(f"Saving history to {path}")
                        file.write(history)
                    wait()

            print("Saving completed")


ModelCatalog.register_custom_model("actor_model", Actor)
ModelCatalog.register_custom_action_dist("actor_dist", ActorDistribution)
