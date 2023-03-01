import os
import graphviz
import gym.spaces
import numpy as np
import ray
import torch as t
from torch import nn
from typing import List, Dict
from gym.spaces import Dict as DictSpace
from ray.tune import Callback
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


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


class Mean(nn.Module):
    def forward(self, x):
        return t.mean(x, dim=1)


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
    def sample(self):
        src_dist = self.source_node_distribution()
        src_node = src_dist.sample()
        tar_dist = self.target_node_distribution(src_node)
        tar_node = tar_dist.sample()
        tar_func_dist = self.target_function_distribution(src_node, tar_node)
        tar_func = tar_func_dist.sample()
        has_edge_dist = self.has_edge_distribution(src_node, tar_node)
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

    def source_node_distribution(self):
        # inputs shape: [batch_size, node_num, output_feature_num]
        # logits shape: [batch_size, node_num]
        logits = self.model.source_node_module(self.inputs[:, :, 1:])
        node_ranks = self.inputs[:, :, 0].detach()

        # source_node_masks shape: [batch_size, node_num]
        source_node_masks = t.cat(
            [
                t.from_numpy(
                    CPPN.get_source_node_mask(
                        self.model.model_config["custom_model_config"][
                            "cppn_input_node_num"
                        ],
                        self.model.model_config["custom_model_config"][
                            "cppn_output_node_num"
                        ],
                        nr,
                    )
                )
                .unsqueeze(0)
                .bool()
                for nr in node_ranks.cpu().numpy()
            ]
        ).to(logits.device)
        logits = t.masked_fill(logits, ~source_node_masks, -t.inf)
        dist = TorchCategorical(logits)
        return dist

    def target_node_distribution(self, source_node):
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
                t.from_numpy(
                    CPPN.get_target_node_mask(
                        src,
                        self.model.model_config["custom_model_config"][
                            "cppn_input_node_num"
                        ],
                        self.model.model_config["custom_model_config"][
                            "cppn_output_node_num"
                        ],
                        nr,
                    )
                )
                .unsqueeze(0)
                .bool()
                for src, nr in zip(source_node, node_ranks.cpu().numpy())
            ]
        ).to(logits.device)
        logits = t.masked_fill(logits, ~target_node_masks, -t.inf)
        dist = TorchCategorical(logits)
        return dist

    def target_function_distribution(self, source_node, target_node):
        # tar_node_embedding shape: [batch_size, output_feature_num]
        batch_size = self.inputs.shape[0]
        src_node_embedding = self.inputs[:, :, 1:][range(batch_size), source_node]
        tar_node_embedding = self.inputs[:, :, 1:][range(batch_size), target_node]

        # logits shape: [batch_size, function_num]
        logits = self.model.target_function_module(
            t.cat((src_node_embedding, tar_node_embedding), dim=1)
        )

        dist = TorchCategorical(logits)
        return dist

    def has_edge_distribution(self, source_node, target_node):
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
        )

        self.value_net = nn.Sequential(
            SlimFC(
                model_config["custom_model_config"]["output_feature_num"],
                256,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            ),
            Mean(),
            SlimFC(256, 128, initializer=normc_initializer(0.01), activation_fn=None),
            SlimFC(128, 1, initializer=normc_initializer(0.01), activation_fn=None),
        )

        output_feature_num = model_config["custom_model_config"]["output_feature_num"]
        self.source_node_module = make_mlp(
            (output_feature_num, 128, 1), squeeze_last=True
        )
        self.target_node_module = make_mlp(
            (output_feature_num * 2, 128, 1), squeeze_last=True
        )
        self.target_function_module = make_mlp(
            (
                output_feature_num * 2,
                128,
                model_config["custom_model_config"]["target_function_num"],
            )
        )
        self.has_edge_module = make_mlp((output_feature_num * 2, 128, 2))
        self.weight_module = make_mlp((output_feature_num * 2, 128, 2))
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
        # Render the robot simulation
        env = base_env.vector_env  # type: VoxcraftGrowthEnvironment
        episode.media["episode_data"]["robot"] = env.best_finished_robot
        episode.media["episode_data"][
            "robot_sim_history"
        ] = env.best_finished_robot_sim_history
        episode.media["episode_data"][
            "cppn_graphs"
        ] = env.best_finished_robot_state_data

    def on_train_result(self, *, algorithm, result, trainer, **kwargs,) -> None:
        num_episodes = result["episodes_this_iter"]
        data = result["episode_media"].get("episode_data", [])
        episode_data = data[-num_episodes:]

        # Only preserve 1 result to reduce load on checkpointing
        result["episode_media"] = {
            "episode_data": episode_data[-1] if len(episode_data) > 0 else []
        }

        if "evaluation" in result:
            result["evaluation"]["episode_media"] = {}
        if "sampler_results" in result:
            result["sampler_results"]["episode_media"] = {}
        print("Cleaning completed")


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
        data = result["episode_media"].get("episode_data", [])
        result["episode_media"] = {}
        if data and data["cppn_graphs"] is not None:
            print(f"Saving cppn graph")
            unpruned_graph, pruned_graph = data["cppn_graphs"]
            g1 = graphviz.Source(unpruned_graph)
            g1.render(
                filename=f"unpruned_{step:08d}",
                directory=self._trial_local_dir[trial],
                format="png",
            )
            g2 = graphviz.Source(pruned_graph)
            g2.render(
                filename=f"pruned_{step:08d}",
                directory=self._trial_local_dir[trial],
                format="png",
            )

            robot = data["robot"]
            history = data["robot_sim_history"]
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

        print("Saving completed")


ModelCatalog.register_custom_model("actor_model", Actor)
ModelCatalog.register_custom_action_dist("actor_dist", ActorDistribution)