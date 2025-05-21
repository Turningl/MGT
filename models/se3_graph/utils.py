# -*- coding: utf-8 -*-
# @Author : liang
# @File : utils.py


import math
import numpy as np
import torch
from torch import nn, optim
from typing import Optional
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor


class TransformerConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_dim: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.0,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None


        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)

        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn = nn.BatchNorm1d(out_channels)
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)

        return self.softplus(x[1] + self.bn(out))

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
        return out


class UpdateEdge(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_dim: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.0,
            bias: bool = True,
            root_weight: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lemb = nn.Embedding(num_embeddings=3, embedding_dim=32)

        self.embedding_dim = 32
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)

        # for test
        self.lin_key_e1 = nn.Linear(in_channels, heads * out_channels)
        self.lin_value_e1 = nn.Linear(in_channels, heads * out_channels)
        self.lin_key_e2 = nn.Linear(in_channels, heads * out_channels)
        self.lin_value_e2 = nn.Linear(in_channels, heads * out_channels)
        self.lin_key_e3 = nn.Linear(in_channels, heads * out_channels)
        self.lin_value_e3 = nn.Linear(in_channels, heads * out_channels)

        # for test ends
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge_len = nn.Linear(in_channels + self.embedding_dim, in_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                            nn.SiLU(),
                                            nn.Linear(out_channels, out_channels))
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
                                        nn.SiLU(),
                                        nn.Linear(out_channels, out_channels))
        self.bn_att = nn.BatchNorm1d(out_channels)

        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge, edge_nei_len, edge_nei_angle: OptTensor = None):
        # preprocess for edge of shape [num_edges, hidden_dim]

        H, C = self.heads, self.out_channels
        if isinstance(edge, Tensor):
            edge: PairTensor = (edge, edge)

        query_x = self.lin_query(edge[1]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        key_x = self.lin_key(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)
        value_x = self.lin_value(edge[0]).view(-1, H, C).unsqueeze(1).repeat(1, 3, 1, 1)

        key_y = torch.cat((self.lin_key_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                           self.lin_key_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                           self.lin_key_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C)), dim=1)

        value_y = torch.cat((self.lin_value_e1(edge_nei_len[:, 0, :]).view(-1, 1, H, C),
                             self.lin_value_e2(edge_nei_len[:, 1, :]).view(-1, 1, H, C),
                             self.lin_value_e3(edge_nei_len[:, 2, :]).view(-1, 1, H, C)), dim=1)

        # preprocess for interaction of shape [num_edges, 3, hidden_dim]
        edge_xy = self.lin_edge(edge_nei_angle).view(-1, 3, H, C)

        key = self.key_update(torch.cat((key_x, key_y, edge_xy), dim=-1))
        alpha = (query_x * key) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_x, value_y, edge_xy), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(-1, 3, self.heads, self.out_channels))

        out = out.view(-1, 3, self.heads * self.out_channels)
        out = self.lin_concate(out)
        # aggregate the msg
        out = out.sum(dim=1)

        return self.softplus(edge[1] + self.bn(out))


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(vmin, vmax, bins)
        )
        self.type = type

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance.unsqueeze(-1) - self.centers)
        if self.type == 'gaussian':
            return (-base ** 2).exp()
        elif self.type == 'quadratic':
            return base ** 2
        elif self.type == 'linear':
            return base
        elif self.type == 'inverse_quadratic':
            return 1.0 / (1.0 + base ** 2)
        elif self.type == 'multiquadric':
            return (1.0 + base ** 2).sqrt()
        elif self.type == 'inverse_multiquadric':
            return 1.0 / (1.0 + base ** 2).sqrt()
        elif self.type == 'spline':
            return base ** 2 * (base + 1.0).log()
        elif self.type == 'poisson_one':
            return (base - 1.0) * (-base).exp()
        elif self.type == 'poisson_two':
            return (base - 2.0) / 2.0 * base * (-base).exp()
        elif self.type == 'matern32':
            return (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp()
        elif self.type == 'matern52':
            return (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp()
        else:
            raise Exception("No Implemented Radial Basis Method")