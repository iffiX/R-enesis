import os
import gym
import numpy as np
import torch as t
import torch.nn as nn
from torch.distributions import MultivariateNormal
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from renesis.env.voxcraft import VoxcraftGrowthEnvironment
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer
from utils import Actor


# configurations
max_episodes = 1000
max_steps = 10
solved_reward = 20
solved_repeat = 5


# model definition
class ActorBranch(nn.Module):
    def __init__(self, actor_layers):
        super().__init__()
        self.actor_layers = actor_layers

    def forward(self, state, action=None):
        a = self.actor_layers(state.permute(0, 4, 1, 2, 3))
        length = a.shape[1] // 2
        dist = MultivariateNormal(
            loc=a[:, :length], covariance_matrix=t.diag_embed(a[:, length:] ** 2)
        )
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act)
        return act, act_log_prob, act_entropy


class CriticBranch(nn.Module):
    def __init__(self, actor_layers, value_branch):
        super().__init__()
        self.actor_layers = actor_layers
        self.value_branch = value_branch

    def forward(self, state):
        a = self.actor_layers(state.permute(0, 4, 1, 2, 3))
        return self.value_branch(a)


if __name__ == "__main__":
    env = VoxcraftGrowthEnvironment(
        {
            "materials": (0, 1),
            "max_dimension_size": 50,
            "max_view_size": 21,
            "amplitude_range": (0, 2),
            "frequency_range": (0, 4),
            "phase_shift_range": (0, 1),
            "max_steps": 10,
            "reward_interval": 1,
            "reward_type": "distance_traveled",
            "base_template_path": str(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa"
                )
            ),
            "voxel_size": 0.01,
            "fallen_threshold": 0.25,
        }
    )
    base_actor = Actor(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_outputs=2 * np.prod(env.action_space.shape),
        model_config={
            "conv_filters": [
                [16, (8, 8, 8), (4, 4, 4)],
                [32, (4, 4, 4), (2, 2, 2)],
                [256, None, None],
            ],
            "post_fcnet_hiddens": [128, None],
        },
        name="base_actor",
    )
    actor = ActorBranch(base_actor.layers).to("cuda:0")
    critic = CriticBranch(base_actor.layers, base_actor.value_branch).to("cuda:0")

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).unsqueeze(0)

        tmp_observations = []
        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = ppo.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.cpu().numpy())
                state = t.tensor(state, dtype=t.float32).unsqueeze(0)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        # update
        ppo.store_episode(tmp_observations)
        ppo.update()

        if episode % 10 == 0:
            renderer = VXHistoryRenderer(
                history=env.robot_sim_history, width=640, height=480
            )
            renderer.render()
            frames = renderer.get_frames()
            create_video_subproc([f for f in frames], "test", f"video-{episode}")
            t.save(
                actor.state_dict(), os.path.join("test", f"chk-actor-{episode}.ckpt"),
            )
            t.save(
                critic.state_dict(), os.path.join("test", f"chk-critic-{episode}.ckpt"),
            )
        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
