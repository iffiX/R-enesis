import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def add_reflexive_edges(
    node_num: int, edge: t.Tensor, edge_num: t.Tensor, edge_weight: t.Tensor
):
    # shape of edge: [B, E, 2]
    # shape of edge_num: [B]
    # shape of edge_weight: [B, E]
    B = edge.shape[0]
    edge_num = edge_num + node_num
    new_edge = t.Tensor(list(range(node_num))).to(dtype=edge.dtype, device=edge.device)
    new_edge = new_edge.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 2)
    edge = t.cat([new_edge, edge], dim=1)
    edge_weight = t.cat(
        [
            t.ones([B, node_num], dtype=edge_weight.dtype, device=edge_weight.device),
            edge_weight,
        ],
        dim=1,
    )
    return edge, edge_num, edge_weight


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self,
        input_feature_num: int,
        output_feature_num: int,
        dropout_prob: float = 0.1,
        lrelu_alpha: float = 1e-2,
        identity: bool = False,
        activation: Callable[[t.Tensor], t.Tensor] = F.relu,
        name: str = "GATLayer",
    ):
        """
        Args:
            input_feature_num: Input feature dimension.
            output_feature_num: Output feature dimension.
            dropout_prob: Probability of dropout.
            lrelu_alpha: Slope of the leaky relu function.
            identity: Perform no transformation on input hidden embeddings if set to True.
            activation: Activation function applied on the output h'.
            name: Name of this layer.
        """

        super(GraphAttentionLayer, self).__init__()
        self.dropout_prob = dropout_prob
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.lrelu_alpha = lrelu_alpha
        self.identity = identity
        self.activation = activation
        self.name = name

        if identity and input_feature_num != output_feature_num:
            raise ValueError(
                "In identity mode the input and output feature must be the same size"
            )
        else:
            self.W = nn.Parameter(t.empty(size=(output_feature_num, input_feature_num)))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(t.empty(size=(1, 2 * output_feature_num)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.lrelu_alpha)

    def forward(self, h, edge, edge_num, edge_weight=None):
        # Let B be batch size, N be the number of nodes,
        # E be the padded number of edges
        # shape of h: [B, N, input_feature_num]
        # shape of edge: [B, E, 2]
        # shape of edge_num: [B]
        # shape of edge_weight: [B, E]
        #
        # For the edge matrix, the first element of dim 2 is the "node
        # of attention" and the second element is one of the neighboring
        # nodes, an example of the edge matrix is like:
        # (0, 0) (0, 1)
        # (1, 0) (1, 1) (1, 2) (1, 3) (1, 4)
        # (2, 1) (2, 2) (2, 4)
        # (3, 3) (3, 4)
        # There should always be an reflexive edge for every node

        # Note: edge and edge_weight are padded with zeros up to the size
        # of E. To get the real size, use edge_num
        N = h.shape[1]

        # remove unnecessary padding
        max_edge_num = int(t.max(edge_num))
        edge = edge[:, :max_edge_num]
        edge_weight = edge_weight[:, :max_edge_num]

        # shape of Wh: [B, N, output_feature_num]
        if not self.identity:
            Wh = F.linear(h, self.W)
        else:
            Wh = h

        # a_input.shape: [B, E, 2 * output_feature_num]
        # where last dim is concatenation of (Wh_i, Wh_j),
        # and (i, j) is an edge
        a_input = self._get_attention_input(Wh, edge)

        if edge_weight is not None:
            a_input = a_input * edge_weight.unsqueeze(-1)

        # att1.shape: [B, E, 1]
        att1 = self.leakyrelu(F.linear(a_input, self.a))

        att2 = self.indexed_softmax(N, att1, edge, edge_num)
        if self.dropout_prob > 0:
            att3 = F.dropout(att2, self.dropout_prob, training=self.training)
        else:
            att3 = att2
        h_prime = self.indexed_multiply_and_gather(att3, Wh, edge, edge_num)

        return self.activation(h_prime)

    def _get_attention_input(self, Wh, edge):
        # center_index and neighbor_index shape: [B, E, output_feature_num]
        center_index = edge[:, :, 0:1].repeat(1, 1, Wh.shape[-1])
        neighbor_index = edge[:, :, 1:2].repeat(1, 1, Wh.shape[-1])

        # output shape: [B, E, 2 * output_feature_num]
        return t.cat(
            [t.gather(Wh, 1, center_index), t.gather(Wh, 1, neighbor_index)], dim=-1
        )

    @staticmethod
    def indexed_softmax(N, x, edge, edge_num):
        # x shape: [B, E, 1]
        # edge shape: [B, E, 2]
        # edge_num shape: [B]

        B = x.shape[0]
        # prevent infinity caused nan
        x_exp = x.exp().clamp(0, 1e6)

        # set x value to 0 if index beyond edge num
        for batch, e_num in enumerate(edge_num):
            x_exp[batch, e_num:] = 0

        # denom shape: [B, N]
        denom = t.scatter_add(
            t.full([B, N], 1e-10, dtype=t.float32, device=x.device),
            dim=1,
            index=edge[:, :, 0],
            src=x_exp.squeeze(-1),
        )

        # denom_per_edge_attention shape: [B, E]
        denom_per_edge_attention = t.gather(denom, 1, edge[:, :, 0])
        return x_exp / denom_per_edge_attention.unsqueeze(-1)

    @staticmethod
    def indexed_multiply_and_gather(x, Wh, edge, edge_num):
        # center_index and neighbor_index shape: [B, E, output_feature_num]
        center_index = edge[:, :, 0:1].repeat(1, 1, Wh.shape[-1])
        neighbor_index = edge[:, :, 1:2].repeat(1, 1, Wh.shape[-1])

        # att_Wh shape: [B, E, output_feature_num]
        att_Wh = t.gather(Wh, 1, neighbor_index) * x

        # set att_Wh value to 0 if index beyond edge num
        for batch, e_num in enumerate(edge_num):
            att_Wh[batch, e_num:] = 0
        return t.scatter_add(t.zeros_like(Wh), dim=1, index=center_index, src=att_Wh)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(In: {str(self.in_features)} -> Out: {str(self.out_features)})"
        )


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(
        self,
        input_feature_num: int,
        output_feature_num: int,
        head_num: int,
        dropout_prob: float = 0.1,
        lrelu_alpha: float = 1e-2,
        identity: bool = False,
        activation: Callable[[t.Tensor], t.Tensor] = F.relu,
        name: str = "GATLayer",
    ):
        super().__init__()
        self.att_heads = [
            GraphAttentionLayer(
                input_feature_num,
                output_feature_num,
                dropout_prob=dropout_prob,
                lrelu_alpha=lrelu_alpha,
                identity=identity,
                activation=activation,
                name=f"{name}_head_{i}",
            )
            for i in range(head_num)
        ]
        for att_head in self.att_heads:
            self.add_module(att_head.name, att_head)

    def forward(self, h, edge, edge_num, edge_weight=None):
        return t.cat(
            [att_head(h, edge, edge_num, edge_weight) for att_head in self.att_heads],
            dim=-1,
        )


class GAT(nn.Module):
    def __init__(
        self,
        input_feature_num: int,
        hidden_feature_num: int,
        output_feature_num: int,
        layer_num: int = 4,
        head_num: int = 3,
        dropout_prob: float = 0.1,
        lrelu_alpha: float = 1e-2,
        activation: Callable[[t.Tensor], t.Tensor] = F.relu,
    ):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout_prob = dropout_prob
        self.layers = []

        for i in range(layer_num):
            in_size = hidden_feature_num * head_num
            out_size = hidden_feature_num
            if i == 0:
                in_size = input_feature_num
            elif i == layer_num - 1:
                out_size = output_feature_num

            if i < layer_num - 1:
                self.layers.append(
                    MultiHeadGraphAttentionLayer(
                        in_size,
                        out_size,
                        head_num,
                        dropout_prob=dropout_prob,
                        lrelu_alpha=lrelu_alpha,
                        identity=False,
                        activation=activation,
                        name="layer_{}".format(i),
                    )
                )
            else:
                self.layers.append(
                    GraphAttentionLayer(
                        in_size,
                        out_size,
                        dropout_prob=dropout_prob,
                        lrelu_alpha=lrelu_alpha,
                        identity=False,
                        activation=lambda x: x,
                        name="layer_out",
                    )
                )
            self.add_module("layer_{}".format(i), self.layers[-1])

    def forward(self, x, edge, edge_num, edge_weight):
        for layer in self.layers:
            x = layer(x, edge, edge_num, edge_weight)
        return x
