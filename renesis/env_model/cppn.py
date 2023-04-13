import cc3d
import numpy as np
from collections import OrderedDict
from typing import (
    List,
    OrderedDict as OrderedDictType,
    Dict as DictType,
    Set,
    Tuple,
    Callable,
    Any,
    Union,
)
from gym.spaces import Box, Dict, MultiDiscrete
from ray.rllib.utils.spaces.repeated import Repeated
from .base import BaseModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rescale(data: np.ndarray, rescale_range: Tuple[float, float]):
    return data * (rescale_range[1] - rescale_range[0]) + rescale_range[0]


def get_gaussian_pdf(mu=0, std=1):
    divider = std * np.sqrt(2 * np.pi)

    def gaussian_pdf(x):
        return np.exp(-((x - mu) ** 2) / (2 * std**2)) / divider

    return gaussian_pdf


def wrap_with_aggregator(
    func: Callable[[np.ndarray], np.ndarray],
    aggregator: Callable[[List[np.ndarray]], np.ndarray] = lambda x: np.mean(x, axis=0),
):
    return lambda x: func(aggregator(x))


def is_voxel_continuous(occupied: np.ndarray):
    _labels, label_num = cc3d.connected_components(
        occupied, connectivity=6, return_N=True, out_dtype=np.uint64
    )
    return label_num <= 1


class InputNode:
    pass


class OutputNode:
    pass


class CPPN:
    def __init__(
        self,
        input_node_num: int,
        output_node_num: int,
        hidden_node_num: int,
        functions: OrderedDictType[str, Callable[[List[np.ndarray]], np.ndarray]],
        output_aggregator: Callable[[List[np.ndarray]], np.ndarray] = lambda x: np.sum(
            x, axis=0
        ),
    ):
        """
        Create a compositional pattern producing network (CPPN) with
        several input nodes, output nodes and hidden nodes

        Args:
            input_node_num: Number of input nodes
            output_node_num: Number of output nodes
            hidden_node_num: Number of hidden nodes
            functions: A dictionary of functions where hidden nodes can choose from
            output_aggregator: Aggregator function for output nodes
        """
        for f in functions:
            if len(f) == 0:
                raise ValueError("Empty function name is not allowed")
        self.input_node_num = input_node_num
        self.output_node_num = output_node_num
        self.hidden_node_num = hidden_node_num
        self.functions = functions
        self.output_aggregator = output_aggregator

        self.nodes = []
        self.nodes += [InputNode() for _ in range(input_node_num)]
        self.nodes += [OutputNode() for _ in range(output_node_num)]
        self.nodes += [None for _ in range(hidden_node_num)]

        # Init
        # Input node always has a rank of 0, and can never be a target node
        # Output node always has a rank of inf, and can never be a source node
        # Hidden nodes have a initialized sequential rank
        # If all edges with a input node/ an output node as a vertex are
        # removed, the remaining graph of hidden nodes is a DAG, and with these
        # edges added back, it is still a DAG.
        self.node_ranks = np.array(
            [0] * input_node_num
            + [np.inf] * output_node_num
            + list(range(1, hidden_node_num + 1))
        )
        self.in_edges = {
            i: set() for i in range(len(self.nodes))
        }  # type: DictType[int, Set[int]]
        self.out_edges = {
            i: set() for i in range(len(self.nodes))
        }  # type: DictType[int, Set[int]]
        self.edges_weights = {}  # type: DictType[Tuple[int, int], float]

    @property
    def node_num(self):
        return len(self.nodes)

    @property
    def edge_num(self):
        return len(self.edges_weights)

    @property
    def function_num(self):
        return len(self.functions)

    def get_node_features(self):
        # row wise:
        # relative_index,
        # is_input_node, is_output_node, is_empty_hidden_node, function one-hot encodings
        features = np.zeros([len(self.nodes), 4 + len(self.functions)], dtype=float)
        features[: self.input_node_num, 0] = [
            x / self.input_node_num for x in range(self.input_node_num)
        ]
        features[
            self.input_node_num : self.input_node_num + self.output_node_num, 0
        ] = [x / self.output_node_num for x in range(self.output_node_num)]
        features[-self.hidden_node_num :, 0] = [
            x / self.hidden_node_num for x in range(self.hidden_node_num)
        ]

        features[: self.input_node_num, 1] = 1
        features[
            self.input_node_num : self.input_node_num + self.output_node_num,
            2,
        ] = 1
        function_names = list(self.functions.keys())
        for i in range(self.input_node_num + self.output_node_num, len(self.nodes)):
            if self.nodes[i] is None:
                features[i, 3] = 1
            else:
                features[i, 4 + function_names.index(self.nodes[i])] = 1
        return features

    def get_edges_and_weights(self):
        edges_and_weights = np.zeros([len(self.edges_weights), 3], dtype=float)
        for idx, (edge, weight) in enumerate(self.edges_weights.items()):
            edges_and_weights[idx, :2] = edge
            edges_and_weights[idx, 2] = weight
        return edges_and_weights

    def get_node_ranks(self):
        return self.node_ranks

    @classmethod
    def get_source_node_mask(cls, node_ranks: np.ndarray):
        """
        Returns:
            A mask of shape (node_num,) and values of {0, 1}.
            0 for not allowing creating/deleting the edge,
            1 for allowing editing.
        """
        # When action dependency is required:
        # The mask is always 0 for output nodes
        # The mask is 1 For input/hidden nodes

        mask = np.full_like(node_ranks, 1).astype(float)
        mask[node_ranks == np.inf] = 0
        return mask

    @classmethod
    def get_target_node_mask(
        cls,
        source_node: int,
        node_ranks: np.ndarray,
    ):
        """
        Args:
            source_node: Index of the source node

        Returns:
            A mask of shape (node_num,) and values of {0, 1}.
            0 for not allowing creating/deleting the edge,
            1 for allowing editing.
        """
        # The mask is always 1 for output nodes, and always 0 for input nodes
        # For hidden nodes, rank must be either higher than source_node

        mask = node_ranks > node_ranks[source_node]
        return mask

    def step(
        self,
        source_node: int,
        target_node: int,
        target_node_function: int,
        has_edge: int,
        weight: float,
    ):
        if (
            self.get_source_node_mask(self.node_ranks)[source_node] != 1
            or self.get_target_node_mask(source_node, self.node_ranks)[target_node] != 1
        ):
            raise ValueError("Invalid edge")

        if target_node >= self.input_node_num + self.output_node_num:
            # Only applies function name change if target node is an hidden node
            self.nodes[target_node] = list(self.functions.keys())[target_node_function]

        if target_node not in self.out_edges[source_node]:
            if has_edge:
                # Create new edge
                self.out_edges[source_node].add(target_node)
                self.in_edges[target_node].add(source_node)
                self.edges_weights[(source_node, target_node)] = weight
            else:
                # Do nothing
                pass
        else:
            if not has_edge:
                # Delete existing edge
                self.out_edges[source_node].remove(target_node)
                self.in_edges[target_node].remove(source_node)
            else:
                # Update existing edge
                self.edges_weights[(source_node, target_node)] = weight

    def eval(self, inputs: List[np.ndarray]):
        if len(inputs) != self.input_node_num:
            raise ValueError("Inputs must be the same size as input nodes")
        if len(set(inp.shape for inp in inputs)) != 1:
            raise ValueError("Inputs must have the same size")
        outputs = []
        cache = [None] * len(self.nodes)
        for output_node in range(
            self.input_node_num, self.input_node_num + self.output_node_num
        ):
            outputs.append(self.recursive_eval(output_node, inputs, cache))
        return outputs

    def recursive_eval(
        self,
        root_node: int,
        inputs: List[np.ndarray],
        cache: List[Union[None, np.ndarray]],
    ):
        source_inputs = []
        for source_node in self.in_edges[root_node]:
            if isinstance(self.nodes[source_node], InputNode):
                source_inputs.append(inputs[source_node])
            else:
                if cache[source_node] is None:
                    cache[source_node] = self.recursive_eval(source_node, inputs, cache)
                source_inputs.append(cache[source_node])

        if len(source_inputs) == 0:
            return np.zeros(inputs[0].shape)
        else:
            if isinstance(self.nodes[root_node], OutputNode):
                output = self.output_aggregator(source_inputs)
            else:
                output = self.functions[self.nodes[root_node]](source_inputs)
            return np.clip(
                np.nan_to_num(output, nan=0, posinf=1e4, neginf=-1e4), -1e4, 1e4
            )

    def get_graphs(
        self,
        input_names: List[str] = None,
        output_names: List[str] = None,
        function_styles: DictType[str, Union[str, None]] = None,
    ):
        """
        Args:
            input_names: A list of names corresponding to each input node.
                None for default naming.
            output_names: A list of names corresponding to each output node.
                None for default naming.
            function_styles: A dictionary of hidden node styles, corresponding
                to the function each hidden node has.
        Returns:
            Two graphs in graphviz representation.
            The first one is un-pruned graph with all edges,
            the second one is pruned graph where edges unrelated to output are removed.
        """
        input_names = input_names or [f"in_{i}" for i in range(self.input_node_num)]
        output_names = output_names or [f"out_{i}" for i in range(self.output_node_num)]
        hidden_nodes = [
            n + f"_{i + self.input_node_num + self.output_node_num}"
            if n is not None
            else None
            for i, n in enumerate(
                self.nodes[self.input_node_num + self.output_node_num :]
            )
        ]
        hidden_node_styles = (
            [None] * self.hidden_node_num
            if function_styles is None
            else [
                function_styles[n] if n is not None else None
                for n in self.nodes[self.input_node_num + self.output_node_num :]
            ]
        )

        unpruned_node_names = input_names + output_names + hidden_nodes
        unpruned_graph = self.render_graph(
            unpruned_node_names, self.edges_weights, hidden_node_styles
        )

        node_mask = [True] * (self.input_node_num + self.output_node_num) + [
            False
        ] * self.hidden_node_num
        self.recursive_find_dependent_nodes(
            list(
                range(self.input_node_num, self.input_node_num + self.output_node_num)
            ),
            node_mask,
        )
        pruned_node_names = [
            n if m else None for n, m in zip(unpruned_node_names, node_mask)
        ]
        pruned_graph = self.render_graph(
            pruned_node_names, self.edges_weights, hidden_node_styles
        )
        return unpruned_graph, pruned_graph

    def recursive_find_dependent_nodes(
        self,
        roots: List[int],
        mask: List[bool],
    ):
        """
        Find all nodes that are connected to an output node.

        Args:
            roots: A list of root nodes.
            mask: A binary mask the same length as total node num, the mask will be modified so that
                for every node that is depended on by an output the mask entry will be True.
        """
        for root in roots:
            mask[root] = True
            if not isinstance(self.nodes[root], InputNode):
                self.recursive_find_dependent_nodes(list(self.in_edges[root]), mask)

    def render_graph(
        self,
        node_names: List[Union[str, None]],
        edges: DictType[Tuple[int, int], float],
        hidden_node_styles: List[Union[str, None]] = None,
    ):
        """
        Args:
            node_names: Name for every node. If node is not appearing, use None for that node.
            edges: All edges. (Note that if the edge contains a node marked as None, it will
                not be displayed.)
            hidden_node_styles: Optional style configuration for hidden nodes
        Returns:
            Graphviz graph in string
        """
        if any(
            name is None
            for name in node_names[: self.input_node_num + self.output_node_num]
        ):
            raise ValueError("Input and output nodes must be present")

        graph = "digraph {\n"

        input_nodes = "; ".join([f'"{n}"' for n in node_names[: self.input_node_num]])
        graph += f"""
        subgraph cluster_inputs {{
            label = "inputs";
            node [shape=ellipse, style=filled];
            {input_nodes}
        }}
        
        """

        output_nodes = "; ".join(
            [
                f'"{n}"'
                for n in node_names[
                    self.input_node_num : self.input_node_num + self.output_node_num
                ]
            ]
        )
        graph += f"""
        subgraph cluster_outputs {{
            label = "outputs";
            node [shape=ellipse, style=filled];
            {output_nodes}
        }}
        
        """

        hidden_node_styles = hidden_node_styles or [None] * self.hidden_node_num

        hidden_node_names = node_names[self.input_node_num + self.output_node_num :]

        # Add styles for hidden nodes
        for style in set(hidden_node_styles):
            if style is None:
                # default style
                style_string = "shape=octagon,style=filled,color=lightgrey"
            else:
                style_string = style

            stylized_nodes = [
                f'"{n}"'
                for st, n in zip(hidden_node_styles, hidden_node_names)
                if st == style and n is not None
            ]
            if stylized_nodes:
                graph += f"node [{style_string}]; " + "; ".join(stylized_nodes) + "\n"

        # Add edges
        for edge, weight in edges.items():
            if node_names[edge[0]] is not None and node_names[edge[1]] is not None:
                graph += (
                    f'"{node_names[edge[0]]}" '
                    f"-> "
                    f'"{node_names[edge[1]]}" '
                    f'[label="{weight:.2f}"];\n'
                )
        graph += "}"
        return graph


class CPPNBaseModel(BaseModel):
    DEFAULT_CPPN_FUNCTIONS = OrderedDict(
        [
            ("sin", wrap_with_aggregator(np.sin)),
            ("gaussian", wrap_with_aggregator(get_gaussian_pdf(0, 1))),
            ("sigmoid", wrap_with_aggregator(lambda x: 1 / (1 + np.exp(-x)))),
            ("power_square", wrap_with_aggregator(lambda x: x**2)),
            ("root_square", wrap_with_aggregator(lambda x: np.sqrt(x))),
            ("agg_sum", wrap_with_aggregator(lambda x: x, lambda x: np.sum(x, axis=0))),
            (
                "agg_mul",
                wrap_with_aggregator(lambda x: x, lambda x: np.prod(x, axis=0)),
            ),
            ("negative", wrap_with_aggregator(lambda x: -x)),
        ]
    )

    def __init__(
        self,
        cppn_output_node_tags: List[str],
        dimension_size=20,
        cppn_hidden_node_num: int = 20,
        cppn_functions: OrderedDictType[str, Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__()
        self.dimension_size = dimension_size
        self.center_voxel_offset = self.dimension_size // 2
        self.cppn_output_node_tags = cppn_output_node_tags
        self.cppn_hidden_node_num = cppn_hidden_node_num

        # input: x, y, z, d
        # output: presence, likelihood * material num, actuation features
        self.cppn_functions = cppn_functions or self.DEFAULT_CPPN_FUNCTIONS

        self.cppn = CPPN(
            4,
            len(self.cppn_output_node_tags),
            self.cppn_hidden_node_num,
            functions=self.cppn_functions,
        )

        self.voxels = None
        self.occupied = None
        self.is_robot_valid = False
        self.update_voxels()

    @property
    def action_space(self):
        return Box(
            low=np.array([0, 0, 0, 0, -np.inf]),
            high=np.array(
                [
                    self.cppn.node_num - 1,
                    self.cppn.node_num - 1,
                    self.cppn.function_num - 1,
                    1,
                    np.inf,
                ]
            ),
        )

    @property
    def observation_space(self):
        # for node_ranks, the view is [ranks], shape [node_num]
        # For nodes, the view is [node features] * node_num, shape [node_num, node_feature_num]
        # For edges, the view is: [source node, target node, edge weight] * edge_num,
        # shape [edge_num, 3]
        return Dict(
            OrderedDict(
                [
                    (
                        "node_ranks",
                        Box(
                            low=-1,
                            high=np.inf,
                            shape=(self.cppn.node_num,),
                            dtype=np.float64,
                        ),
                    ),
                    (
                        "nodes",
                        Box(
                            low=0,
                            high=np.inf,
                            shape=(self.cppn.node_num, (4 + len(self.cppn_functions))),
                            dtype=np.float64,
                        ),
                    ),
                    (
                        "edges",
                        Repeated(
                            Box(
                                low=np.array([0, 0, -np.inf], dtype=float),
                                high=np.array([np.inf, np.inf, np.inf], dtype=float),
                                shape=(3,),
                                dtype=np.float64,
                            ),
                            max_len=2 * self.cppn.node_num * self.cppn.node_num,
                        ),
                    ),
                ]
            )
        )

    def reset(self, initial_steps=0):
        self.steps = 0
        self.cppn = CPPN(
            4,
            len(self.cppn_output_node_tags),
            self.cppn_hidden_node_num,
            functions=self.cppn_functions,
        )
        for i in range(initial_steps):
            self.step(self.select_action())

    def is_finished(self):
        return False

    def is_robot_invalid(self):
        return not self.is_robot_valid

    def step(self, action: np.ndarray):
        # print(action)
        self.cppn.step(
            int(action[0]),
            int(action[1]),
            int(action[2]),
            int(action[3]),
            action[4],
        )
        self.update_voxels()
        self.steps += 1

    def observe(self):
        result = OrderedDict(
            [
                ("node_ranks", self.cppn.get_node_ranks().astype(np.float64)),
                ("nodes", self.cppn.get_node_features().astype(np.float64)),
                ("edges", self.cppn.get_edges_and_weights().astype(np.float64)),
            ]
        )
        return result

    def get_voxels(self):
        raise NotImplementedError()

    def get_robot(self):
        raise NotImplementedError()

    def get_state_data(self):
        return self.cppn.get_graphs(
            ["x", "y", "z", "d"],
            self.cppn_output_node_tags,
            {
                "sin": None,
                "gaussian": None,
                "sigmoid": None,
                "power_square": None,
                "root_square": None,
                "agg_sum": "shape=doubleoctagon,color=lightgrey",
                "agg_mul": "shape=doubleoctagon,color=darkgrey",
                "negative": None,
            }
            if self.cppn_functions == self.DEFAULT_CPPN_FUNCTIONS
            else None,
        )

    ################################################################################
    # CPPN specific methods
    ################################################################################
    def select_action(self):
        node_ranks = self.observe()["node_ranks"]
        source_node_mask = CPPN.get_source_node_mask(node_ranks)
        # Randomly sample a valid source node
        source_node = np.random.choice(np.where(source_node_mask.astype(bool))[0])
        target_node_mask = CPPN.get_target_node_mask(source_node, node_ranks)
        # Randomly sample a valid target node
        target_node = np.random.choice(np.where(target_node_mask.astype(bool))[0])
        target_function = np.random.choice(list(range(len(self.cppn_functions))))
        has_edge = np.random.choice([0, 1])
        weight = float(np.random.normal(0, 1, 1))
        return np.array(
            [source_node, target_node, target_function, has_edge, weight],
            dtype=float,
        )

    def update_voxels(self):
        raise NotImplementedError()


class CPPNBinaryTreeModel(CPPNBaseModel):
    def __init__(
        self,
        dimension_size=20,
        cppn_hidden_node_num: int = 20,
        cppn_functions: OrderedDictType[str, Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(
            ["presence?", "passive?", "phase?"],
            dimension_size=dimension_size,
            cppn_hidden_node_num=cppn_hidden_node_num,
            cppn_functions=cppn_functions,
        )

    def get_robot(self):
        x_occupied = [
            x for x in range(self.occupied.shape[0]) if np.any(self.occupied[x])
        ]
        y_occupied = [
            y for y in range(self.occupied.shape[1]) if np.any(self.occupied[:, y])
        ]
        z_occupied = [
            z for z in range(self.occupied.shape[2]) if np.any(self.occupied[:, :, z])
        ]
        min_x = min(x_occupied)
        max_x = max(x_occupied) + 1
        min_y = min(y_occupied)
        max_y = max(y_occupied) + 1
        min_z = min(z_occupied)
        max_z = max(z_occupied) + 1
        representation = []

        for z in range(min_z, max_z):
            layer_representation = (
                self.voxels[min_x:max_x, min_y:max_y, z]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                None,
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_voxels(self):
        return (
            self.voxels
            if self.voxels is not None
            else np.zeros([self.dimension_size] * 3, dtype=np.float32)
        )

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = list(
            range(
                -self.center_voxel_offset,
                self.dimension_size - self.center_voxel_offset,
            )
        )
        coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
        coords = np.transpose(coords.reshape([coords.shape[0], -1]))
        distances = np.linalg.norm(coords, axis=1)
        # Inputs of shape (coord_num, 4), each row is x, y, z, d
        # splitted by column
        inputs = [coords[:, 0], coords[:, 1], coords[:, 2]] + [distances]

        # Outputs are List[array(coord_num,), array(coord_num,), array(coord_num,)]
        # first logit is used to control voxel presence
        # second logit is used to control voxel passiveness
        # third logit is used to control voxel phase
        outputs = self.cppn.eval(inputs)
        outputs = [sigmoid(outputs[0]), sigmoid(outputs[1]), sigmoid(outputs[2])]

        material = np.where(
            outputs[0] < 0.5,
            0,
            np.where(outputs[1] < 0.5, 1, np.where(outputs[2] < 0.5, 2, 3)),
        )
        self.voxels = np.zeros(
            [self.dimension_size] * 3,
            dtype=np.float32,
        )
        self.voxels[
            coords[:, 0] + self.center_voxel_offset,
            coords[:, 1] + self.center_voxel_offset,
            coords[:, 2] + self.center_voxel_offset,
        ] = material
        # print("outputs:")
        # print(outputs)
        # print("voxels:")
        # print(self.voxels)
        self.occupied = self.voxels[:, :, :] != 0
        self.is_robot_valid = is_voxel_continuous(self.occupied) and np.any(
            self.occupied
        )


class CPPNBinaryTreeWithPhaseOffsetModel(CPPNBaseModel):
    def __init__(
        self,
        dimension_size=20,
        cppn_hidden_node_num: int = 20,
        cppn_functions: OrderedDictType[str, Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(
            ["presence?", "passive?", "phase_offset"],
            dimension_size=dimension_size,
            cppn_hidden_node_num=cppn_hidden_node_num,
            cppn_functions=cppn_functions,
        )

    def get_robot(self):
        x_occupied = [
            x for x in range(self.occupied.shape[0]) if np.any(self.occupied[x])
        ]
        y_occupied = [
            y for y in range(self.occupied.shape[1]) if np.any(self.occupied[:, y])
        ]
        z_occupied = [
            z for z in range(self.occupied.shape[2]) if np.any(self.occupied[:, :, z])
        ]
        min_x = min(x_occupied)
        max_x = max(x_occupied) + 1
        min_y = min(y_occupied)
        max_y = max(y_occupied) + 1
        min_z = min(z_occupied)
        max_z = max(z_occupied) + 1
        representation = []

        for z in range(min_z, max_z):
            layer_representation = (
                self.voxels[min_x:max_x, min_y:max_y, z, 0]
                .astype(int)
                .flatten(order="F")
                .tolist(),
                None,
                None,
                self.voxels[min_x:max_x, min_y:max_y, z, 1]
                .astype(float)
                .flatten(order="F")
                .tolist(),
            )
            representation.append(layer_representation)
        return (max_x - min_x, max_y - min_y, max_z - min_z), representation

    def get_voxels(self):
        return (
            self.voxels[0]
            if self.voxels is not None
            else np.zeros([self.dimension_size] * 3, dtype=np.float32)
        )

    def update_voxels(self):
        # generate coordinates
        # Eg: if dimension size is 20, indices are [-10, ..., 9]
        # if dimension size if 21, indices are [-10, ..., 10]
        indices = list(
            range(
                -self.center_voxel_offset,
                self.dimension_size - self.center_voxel_offset,
            )
        )
        coords = np.stack(np.meshgrid(indices, indices, indices, indexing="ij"))
        coords = np.transpose(coords.reshape([coords.shape[0], -1]))
        distances = np.linalg.norm(coords, axis=1)
        # Inputs of shape (coord_num, 4), each row is x, y, z, d
        # splitted by column
        inputs = [coords[:, 0], coords[:, 1], coords[:, 2]] + [distances]

        # Outputs are List[array(coord_num,), array(coord_num,), array(coord_num,)]
        # first logit is used to control voxel presence
        # second logit is used to control voxel passiveness
        # third is phase offset output
        outputs = self.cppn.eval(inputs)
        outputs = [sigmoid(outputs[0]), sigmoid(outputs[1]), outputs[2]]

        material = np.where(
            outputs[0] < 0.5,
            0,
            np.where(outputs[1] < 0.5, 1, 2),
        )
        self.voxels = np.zeros(
            [self.dimension_size] * 3 + [2],
            dtype=np.float32,
        )
        self.voxels[
            coords[:, 0] + self.center_voxel_offset,
            coords[:, 1] + self.center_voxel_offset,
            coords[:, 2] + self.center_voxel_offset,
        ] = np.stack([material, outputs[2]], axis=1)
        # print("outputs:")
        # print(outputs)
        # print("voxels:")
        # print(self.voxels)
        self.occupied = self.voxels[:, :, :, 0] != 0
        self.is_robot_valid = is_voxel_continuous(self.occupied) and np.any(
            self.occupied
        )


class CPPNVirtualShapeBinaryTreeModel(CPPNBinaryTreeModel):
    def get_non_zero_voxel_positions(self):
        return np.stack(np.nonzero(self.occupied), axis=1)

    def get_non_zero_voxel_materials(self):
        positions = np.nonzero(self.occupied)
        return self.voxels[positions[0], positions[1], positions[2]]
