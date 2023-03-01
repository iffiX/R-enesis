import os
import copy
import graphviz
import numpy as np
from renesis.env_model.cppn import CPPN
from renesis.env.voxcraft import VoxcraftCPPNEnvironment
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer

MAX_ITERATIONS = 40
SELECTION_SIZE = 128 * 7
VARIATION_SIZE = 2
# OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_PATH = "/home/mlw0504/data/workspace/renesis"


def top_k_indicies(values, k):
    return np.argpartition(values, -k)[-k:]


def render(history):
    try:
        renderer = VXHistoryRenderer(history=history, width=640, height=480)
        renderer.render()
        frames = renderer.get_frames()
        if frames.ndim == 4:
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


if __name__ == "__main__":
    env = VoxcraftCPPNEnvironment(
        {
            "dimension_size": 6,
            "cppn_intermediate_node_num": 10,
            "max_steps": np.inf,
            "reward_type": "distance_traveled",
            "base_config_path": str(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "data", "base.vxa"
                )
            ),
            "voxel_size": 0.01,
            "fallen_threshold": 0.25,
            "num_envs": SELECTION_SIZE * VARIATION_SIZE,
        }
    )
    env.vector_reset()
    best_reward = -np.inf
    for step in range(MAX_ITERATIONS):
        actions = []
        for model in env.env_models:
            node_ranks = model.observe()["node_ranks"]
            source_node_mask = CPPN.get_source_node_mask(4, 3, node_ranks)
            # Randomly sample a valid source node
            source_node = np.random.choice(np.where(source_node_mask.astype(bool))[0])
            target_node_mask = CPPN.get_target_node_mask(source_node, 4, 3, node_ranks)
            # Randomly sample a valid target node
            target_node = np.random.choice(np.where(target_node_mask.astype(bool))[0])
            target_function = np.random.choice(list(range(len(model.cppn_functions))))
            has_edge = np.random.choice([0, 1])
            weight = float(np.random.normal(0, 1, 1))
            actions.append(
                np.array(
                    [source_node, target_node, target_function, has_edge, weight],
                    dtype=float,
                )
            )
        _, rewards, *__ = env.vector_step(actions)

        print(
            f"Step {step}: "
            f"\tbest: {env.best_reward:.4f} "
            f"\tmax: {np.max(rewards):.4f} "
            f"\tmean: {np.mean(rewards):.4f} "
            f"\tmin: {np.min(rewards):.4f}"
        )
        if env.best_reward > best_reward:
            best_reward = env.best_reward
            unpruned_graph, pruned_graph = env.best_finished_robot_state_data
            g1 = graphviz.Source(unpruned_graph)
            g1.render(
                filename=f"unpruned_{step:08d}", directory=OUTPUT_PATH, format="png",
            )
            g2 = graphviz.Source(pruned_graph)
            g2.render(
                filename=f"pruned_{step:08d}", directory=OUTPUT_PATH, format="png",
            )

            robot = env.best_finished_robot
            history = env.best_finished_robot_sim_history
            frames = render(history)

            if frames is not None:
                path = os.path.join(OUTPUT_PATH, f"rendered_{step:08d}.gif")
                wait = create_video_subproc(
                    [f for f in frames],
                    path=OUTPUT_PATH,
                    filename=f"rendered_{step:08d}",
                    extension=".gif",
                )
                with open(
                    os.path.join(OUTPUT_PATH, f"robot-{step:08d}.vxd"), "w"
                ) as file:
                    file.write(robot)
                with open(
                    os.path.join(OUTPUT_PATH, f"run-{step:08d}.history"), "w"
                ) as file:
                    file.write(history)
                wait()

        # Select population with highest rewards
        best_indicies = top_k_indicies(rewards, SELECTION_SIZE)
        best_cppns = []
        for idx in best_indicies:
            best_cppns.append(copy.deepcopy(env.env_models[idx]))

        env.vector_reset()
        for idx in range(len(env.env_models)):
            env.env_models[idx] = copy.deepcopy(best_cppns[idx % len(best_cppns)])
