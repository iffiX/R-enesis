import os
import copy
import pickle
import graphviz
import numpy as np
import pyvista as pv
from PIL import Image
from renesis.env_model.cppn import CPPN, CPPNBinaryTreeModel
from renesis.env.virtual_shape import VirtualShapeCPPNEnvironment
from renesis.utils.plotter import Plotter
from renesis.sim import VXHistoryRenderer
from experiments.cppn_virtual_shape_evolution.utils import generate_3d_shape

TRIALS = 10
GENOME_SIZE = 100
MAX_ITERATIONS = 100
SELECTION_SIZE = 8 * 40
VARIATION_SIZE = 2
ROOT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# ROOT_OUTPUT_PATH = "/home/mlw0504/data/workspace/renesis_shape"

LOG_PATH = os.path.join(ROOT_OUTPUT_PATH, "log")
RESULT_PATH = os.path.join(ROOT_OUTPUT_PATH, "result")
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


def top_k_indicies(values, k):
    return np.argpartition(values, -k)[-k:]


def select_action(model: CPPNBinaryTreeModel):
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
    pv.global_theme.window_size = [1536, 768]
    for trial in range(TRIALS):
        if os.path.exists(os.path.join(LOG_PATH, f"{trial}.data")):
            continue
        trial_log = []

        plotter = Plotter()
        reference_shape = generate_3d_shape(10, 100, material_num=3)

        with open(os.path.join(LOG_PATH, f"{trial}_shape.data"), "wb") as file:
            pickle.dump(reference_shape, file)

        img = plotter.plot_voxel(reference_shape, distance=30)
        Image.fromarray(img, mode="RGB").save(
            os.path.join(LOG_PATH, f"{trial}_shape.png")
        )

        env = [
            VirtualShapeCPPNEnvironment(
                {
                    "dimension_size": 10,
                    "cppn_hidden_node_num": 30,
                    "max_steps": np.inf,
                    "reference_shape": reference_shape,
                    "reward_type": "correct_rate",
                    "render_config": {"distance": 30},
                }
            )
            for _ in range(SELECTION_SIZE * VARIATION_SIZE)
        ]
        best_reward = -np.inf

        # When action dependency is disabled, actions are disjoint
        model_genomes = [
            [select_action(sub_env.env_model) for _ in range(GENOME_SIZE)]
            for sub_env in env
        ]

        for step in range(MAX_ITERATIONS):
            actions = []
            for sub_env, genome in zip(env, model_genomes):
                for action in genome:
                    sub_env.step(action)

            rewards = [sub_env.get_reward() for sub_env in env]

            should_save = False
            best_i = -1
            for i in range(len(env)):
                if rewards[i] > best_reward:
                    should_save = True
                    best_reward = rewards[i]
                    best_i = i

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
                unpruned_graph, pruned_graph = env[best_i].env_model.get_state_data()
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

                img = env[best_i].render()
                Image.fromarray(img, mode="RGB").save(
                    os.path.join(TRIAL_RESULT_PATH, f"generated_{step:08d}.png")
                )

                img = plotter.plot_voxel_error(
                    reference_shape, env[best_i].env_model.voxels, distance=30
                )
                Image.fromarray(img, mode="RGB").save(
                    os.path.join(TRIAL_RESULT_PATH, f"error_{step:08d}.png")
                )

            # Select population with highest rewards
            best_indicies = top_k_indicies(rewards, SELECTION_SIZE)
            best_genomes = []
            for idx in best_indicies:
                best_genomes.append(model_genomes[idx])

            for sub_env in env:
                sub_env.reset()
            for idx in range(len(model_genomes)):
                model_genomes[idx] = mutate(
                    copy.deepcopy(best_genomes[idx % len(best_genomes)]),
                    env[idx].env_model,
                )

        print(f"Saving trial {trial} log")
        with open(os.path.join(LOG_PATH, f"{trial}.data"), "wb") as file:
            pickle.dump(trial_log, file)
