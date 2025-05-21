# -*- coding: utf-8 -*-
# @Author : liang
# @File : utils.py


import math
import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple, Union, Dict
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor, PairTensor
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn import o3


class FFNConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.0,
            edge_dim: Optional[int] = None,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(FFNConv, self).__init__(node_dim=0, **kwargs)

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


class UpdateConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        ns: int = 64,
        nv: int = 8,
        residual: bool = True,
    ):
        super().__init__()

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e'
        ]
        self.ns, self.nv = ns, nv
        self.node_linear = nn.Linear(in_channels, ns)
        # self.skip_linear = nn.Linear(in_channels, out_channels)

        self.sh = '1x0e + 1x1o + 1x2e'

        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=edge_dim,
            residual=residual
        )

        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=edge_dim,
            residual=False
        )

        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(ns)
        self.node_linear_2 = nn.Linear(ns, out_channels)

    def forward(self, data, node_feature, edge_index, edge_feature):

        edge_vec = data.edge_attr

        edge_irr = o3.spherical_harmonics(self.sh, edge_vec, normalize=True, normalization='component')

        skip_connect = node_feature
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.softplus(self.node_linear_2(self.softplus(self.bn(node_feature))))
        node_feature += skip_connect

        return node_feature


class TensorProductConvLayer(torch.nn.Module):
    # from Torsional diffusion
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, self.tp.weight_numel)
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out


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