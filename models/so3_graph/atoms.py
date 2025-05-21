# -*- coding: utf-8 -*-
# @Author : liang
# @File : atoms.py


import torch
from torch import nn, optim
import torch.nn.init as init
from torch_geometric.nn import global_mean_pool
from models.so3_graph.utils import (RBFExpansion,
                                    FFNConv,
                                    UpdateConvEqui,
                                    TensorProductConvLayer)


class SO3_GraphEncoder(nn.Module):
    """
    A graph encoder module designed for processing SO(3) symmetric graph-structured data,
    typically used in molecular or geometric deep learning tasks.


    Attributes:
        euclidean (bool): Whether to use Euclidean distance for edge features.
        eno3 (bool): Whether to use E(3)-equivariant graph convolution layers.
        conv_num_layers (int): Number of convolutional layers.
        atom_embedding (nn.Linear): Linear layer for embedding atom features.
        edge_embedding (nn.Sequential): Sequential module for embedding edge features using RBF expansion and linear transformations.
        equi_update (nn.ModuleList): List of equivariant graph convolution layers.
        ffn_layers (FFNConv): Network for final node feature processing.

    Methods:
        forward(data): Processes the input graph data through the encoder pipeline.

    Example:
        >>> encoder = SO3_GraphEncoder(conv_layers=3, rbf_min=-4.0, rbf_max=4.0, atom_input_features=92, fc_features=256)
        >>> data = ...  # Example graph data with node features and edge attributes
        >>> output = encoder(data)
        >>> print(output.shape)  # Output shape should be (batch_size, fc_features)

    Note:
        The input graph data is expected to be in the format of PyTorch Geometric Data objects,
         containing node features (`data.x`), edge indices (`data.edge_index`), and edge attributes (`data.edge_attr`).
    """

    def __init__(self,
                 conv_layers: int = 3,
                 rbf_min: float = -4.0,
                 rbf_max: float = 4.0,
                 atom_input_features: int = 92,
                 fc_features: int = 256,
                 euclidean: bool = False,
                 ns: int = 64,
                 nv: int = 8,
                 eno3: bool = False,
                 ):
        """
        Args:
            conv_layers (int): Number of convolutional layers.
            rbf_min (float): Minimum value for RBF expansion.
            rbf_max (float): Maximum value for RBF expansion.
            atom_input_features (int): Number of input features for atoms.
            fc_features (int): Number of features for fully connected layers.
            euclidean (bool): Whether to use Euclidean distance for edge features.
            ns (int): Number of scalar features for E(3)-equivariant layers.
            nv (int): Number of vector features for E(3)-equivariant layers.
            eno3 (bool): Whether to use equivariant graph convolution layers.
        """

        super().__init__()
        self.euclidean = euclidean
        self.eno3 = eno3
        self.conv_num_layers = conv_layers

        # Embedding layer for atom features
        self.atom_embedding = nn.Linear(atom_input_features, fc_features)

        # Sequential module for edge feature embedding using RBF expansion and linear transformations
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=rbf_min, vmax=rbf_max, bins=fc_features),
            nn.Linear(fc_features, fc_features),
            nn.SiLU()
        )

        # E(3)-equivariant graph convolution layers
        if self.eno3:
            self.equi_update = nn.ModuleList(
                [
                    UpdateConvEqui(fc_features, fc_features, fc_features,
                                   ns=ns,
                                   nv=nv,
                                   residual=True)
                    for _ in range(conv_layers - 1)
                ]
            )

        self.transconv = nn.ModuleList(
            [
                FFNConv(fc_features,
                     fc_features,
                     edge_dim=fc_features)
                for _ in range(conv_layers - 1)
             ]
        )


    def forward(self, data):
        """
        Forward pass through the graph encoder.

        Args:
            data: Input graph data containing node features (`data.x`), edge indices (`data.edge_index`), and edge attributes (`data.edge_attr`).

        Returns:
            torch.Tensor: Encoded graph features pooled over all nodes.
        """

        edge_index = data.edge_index

        # Use Euclidean distance for edge features or not
        if self.euclidean:
            edge_features = self.edge_embedding(torch.norm(data.edge_attr, dim=1))
        else:
            edge_features = self.edge_embedding(-0.75 / torch.norm(data.edge_attr, dim=1))

        # Embed initial node features using the atom embedding layer
        node_features = self.atom_embedding(data.x)

        # Apply SO(3)-equivariant graph convolution layers if enabled
        if self.eno3:
            for i in range(self.conv_num_layers - 1):
                node_features = self.equi_update[i](data, node_features, edge_index, edge_features)

                # Atom layer for node feature processing
                node_features = self.transconv[i](node_features, edge_index, edge_features)

        # Pool output
        features_pool = global_mean_pool(node_features, data.batch)

        return features_pool


class SO3_ProjectionHead(nn.Module):
    """
    A projection head module designed for SO(3) tasks,
    typically used in neural networks for dimensionality reduction and feature transformation.

    Attributes:
        projection (nn.Linear): The initial linear projection layer that reduces or transforms the input embedding dimension to the desired projection dimension.
        act (nn.SiLU): The non-linear activation function (Sigmoid Linear Unit) applied after the first linear transformation.
        fc (nn.Linear): The second linear layer that further processes the projected features.
        dropout (nn.Dropout): Dropout layer for regularization to prevent overfitting.
        layer_norm (nn.LayerNorm): Layer normalization applied to the projected features before adding the residual connection.

    Methods:
        forward(x): Processes the input tensor through the projection head pipeline.
        _init_weights(): Initializes the weights of the linear layers using Xavier uniform initialization and sets biases to zero.

    Example:
        >>> proj_head = SO3_ProjectionHead(embedding_dim=128, projection_dim=64, dropout=0.1)
        >>> input_tensor = torch.randn(10, 128)  # Example input tensor with batch size 10 and embedding dim 128
        >>> output = proj_head(input_tensor)
        >>> print(output.shape)  # Output shape should be (10, 64)

    Note:
        The input to this module is expected to be a tensor of shape (batch_size, embedding_dim).
        The output will have shape (batch_size, projection_dim).
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        """
        Args:
            embedding_dim (int): The dimension of the input embedding.
            projection_dim (int): The desired dimension of the projected output.
            dropout (float): Dropout probability for regularization.
        """

        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)  # Initial linear projection layer
        self.act = nn.SiLU()  # Non-linear activation function (SiLU)
        self.fc = nn.Linear(projection_dim, projection_dim)  # Second linear layer for further processing
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        self.layer_norm = nn.LayerNorm(projection_dim)  # Layer normalization

        self._init_weights()  # Initialize weights using Xavier uniform and set biases to zero

    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization and set biases to zero.
        """
        init.xavier_uniform_(self.projection.weight)  # Xavier uniform initialization for projection layer weights
        init.zeros_(self.projection.bias)  # Set projection layer bias to zero
        init.xavier_uniform_(self.fc.weight)  # Xavier uniform initialization for fc layer weights
        init.zeros_(self.fc.bias)  # Set fc layer bias to zero


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim).
        """

        projected = self.projection(x)  # Apply initial linear projection
        projected = self.dropout(self.act(self.fc(projected)))  # Apply non-linear activation, second linear layer, and dropout
        x = x + self.layer_norm(projected)  # Add residual connection after layer normalization

        return x
