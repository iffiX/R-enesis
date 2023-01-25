import numpy as np
import graphviz
import matplotlib.pyplot as plt
from renesis.env_model.cppn import CPPN, CPPNModel

if __name__ == "__main__":
    cppn = CPPN(
        input_node_num=2,
        output_node_num=2,
        intermediate_node_num=20,
        functions=CPPNModel.DEFAULT_CPPN_FUNCTIONS,
    )

    source_node_mask = cppn.get_source_node_mask(2, 2, cppn.node_ranks)
    target_node_mask = cppn.get_target_node_mask(1, 2, 2, cppn.node_ranks)
    cppn.step(0, 4, 3, 1, 1)
    cppn.step(1, 5, 3, 1, 1)
    cppn.step(4, 6, 5, 1, 1)
    cppn.step(5, 6, 5, 1, 1)
    cppn.step(6, 7, 0, 1, 1)
    cppn.step(7, 2, 0, 1, 1)
    cppn.step(7, 8, 7, 1, 1)
    cppn.step(8, 3, 0, 1, 1)
    # redundant edge
    cppn.step(0, 19, 0, 1, 1)
    features = cppn.get_node_features()
    edge_and_weights = cppn.get_edges_and_weights()
    indices = list(np.linspace(-1, 1, 20))
    coords = np.stack(np.meshgrid(indices, indices, indexing="ij"))
    coords = np.transpose(coords.reshape([coords.shape[0], -1]))
    inputs = [coords[:, 0], coords[:, 1]]
    outputs = list(cppn.eval(inputs))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(inputs[0], inputs[1], outputs[0], color="blue", marker="^")
    ax.scatter(inputs[0], inputs[1], outputs[1], color="red", marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    unpruned_graph, pruned_graph = cppn.get_graphs(
        ["x", "y"],
        ["z", "-z"],
        {
            "sin": None,
            "gaussian": None,
            "sigmoid": None,
            "power_square": None,
            "root_square": None,
            "agg_sum": "shape=doubleoctagon,color=lightgrey",
            "agg_mul": "shape=doubleoctagon,color=darkgrey",
            "negative": None,
        },
    )
    g1 = graphviz.Source(unpruned_graph)
    g1.render(filename="unpruned", directory=".", format="png")
    g2 = graphviz.Source(pruned_graph)
    g2.render(filename="pruned", directory=".", format="png")
    # plt.show()
