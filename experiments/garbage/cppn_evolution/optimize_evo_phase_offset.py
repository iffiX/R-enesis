import os
import copy
import pickle
import graphviz
import numpy as np
from renesis.env_model.cppn import CPPN, CPPNBinaryTreeWithPhaseOffsetModel
from renesis.env.voxcraft import VoxcraftCPPNBinaryTreeWithPhaseOffsetEnvironment
from renesis.utils.media import create_video_subproc
from renesis.sim import VXHistoryRenderer

TRIALS = 10
GENOME_SIZE = 40
MAX_ITERATIONS = 100
SELECTION_SIZE = 8 * 40
VARIATION_SIZE = 2
ROOT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output2")
# ROOT_OUTPUT_PATH = "/home/mlw0504/data/workspace/renesis2"

LOG_PATH = os.path.join(ROOT_OUTPUT_PATH, "log")
RESULT_PATH = os.path.join(ROOT_OUTPUT_PATH, "result")
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


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


def select_action(model: CPPNBinaryTreeWithPhaseOffsetModel):
    node_ranks = model.observe()["node_ranks"]
    source_node_mask = CPPN.get_source_node_mask(node_ranks)
    # Randomly sample a valid source node
    source_node = np.random.choice(np.where(source_node_mask.astype(bool))[0])
    target_node_mask = CPPN.get_target_node_mask(source_node, node_ranks)
    # Randomly sample a valid target node
    target_node = np.random.choice(np.where(target_node_mask.astype(bool))[0])
    target_function = np.random.choice(list(range(len(model.cppn_functions))))
    has_edge = np.random.choice([0, 1])
    weight = float(np.random.normal(0, 1, 1))
    return np.array(
        [source_node, target_node, target_function, has_edge, weight], dtype=float,
    )


def mutate(genome, model):
    genome[np.random.randint(0, len(genome))] = select_action(model)
    return genome


if __name__ == "__main__":
    for trial in range(TRIALS):
        if os.path.exists(os.path.join(LOG_PATH, f"{trial}.data")):
            continue
        trial_log = []

        env = VoxcraftCPPNBinaryTreeWithPhaseOffsetEnvironment(
            {
                "dimension_size": 6,
                "cppn_hidden_node_num": 10,
                "cppn_require_dependency": False,
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
        best_reward = -np.inf
        best_finished_robot = None
        best_finished_robot_sim_history = None
        best_finished_robot_state_data = None

        # When action dependency is disabled, actions are disjoint
        model_genomes = [
            [select_action(env.env_models[0]) for _ in range(GENOME_SIZE)]
            for __ in range(len(env.env_models))
        ]

        for step in range(MAX_ITERATIONS):
            actions = []
            for model, genome in zip(env.env_models, model_genomes):
                for action in genome:
                    model.step(action)

            rewards = env.get_rewards([False for m in env.env_models])

            should_save = False
            # best_i = None
            for i in range(env.num_envs):
                if rewards[i] > best_reward:
                    # best_i = i
                    should_save = True
                    best_reward = rewards[i]
                    best_finished_robot = env.previous_best_robots[i]
                    best_finished_robot_sim_history = env.previous_best_sim_histories[i]
                    best_finished_robot_state_data = env.previous_best_state_data[i]
            # for y in model_genomes[best_i]:
            #     print(y)

            print(
                f"Step {step}: "
                f"\tbest: {best_reward:.4f} "
                f"\tmax: {np.max(rewards):.4f} "
                f"\tmean: {np.mean(rewards):.4f} "
                f"\tmin: {np.min(rewards):.4f}"
            )
            trial_log.append(
                (best_reward, np.max(rewards), np.mean(rewards), np.min(rewards))
            )

            if should_save:
                TRIAL_RESULT_PATH = os.path.join(RESULT_PATH, str(trial))
                unpruned_graph, pruned_graph = best_finished_robot_state_data
                g1 = graphviz.Source(unpruned_graph)
                g1.render(
                    filename=f"unpruned_{step:08d}",
                    directory=TRIAL_RESULT_PATH,
                    format="png",
                )
                g2 = graphviz.Source(pruned_graph)
                g2.render(
                    filename=f"pruned_{step:08d}",
                    directory=TRIAL_RESULT_PATH,
                    format="png",
                )

                robot = best_finished_robot
                history = best_finished_robot_sim_history
                frames = render(history)

                if frames is not None:
                    path = os.path.join(TRIAL_RESULT_PATH, f"rendered_{step:08d}.gif")
                    wait = create_video_subproc(
                        [f for f in frames],
                        path=TRIAL_RESULT_PATH,
                        filename=f"rendered_{step:08d}",
                        extension=".gif",
                    )
                    with open(
                        os.path.join(TRIAL_RESULT_PATH, f"robot-{step:08d}.vxd"), "w"
                    ) as file:
                        file.write(robot)
                    with open(
                        os.path.join(TRIAL_RESULT_PATH, f"run-{step:08d}.history"), "w"
                    ) as file:
                        file.write(history)
                    wait()

            # Select population with highest rewards
            best_indicies = top_k_indicies(rewards, SELECTION_SIZE)
            best_genomes = []
            for idx in best_indicies:
                best_genomes.append(model_genomes[idx])

            env.vector_reset()
            for idx in range(len(model_genomes)):
                model_genomes[idx] = mutate(
                    copy.deepcopy(best_genomes[idx % len(best_genomes)]),
                    env.env_models[0],
                )

        print(f"Saving trial {trial} log")
        with open(os.path.join(LOG_PATH, f"{trial}.data"), "wb") as file:
            pickle.dump(trial_log, file)
