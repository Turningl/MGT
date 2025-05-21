# -*- coding: utf-8 -*-
# @Author : liang
# @File : experts.py


import torch
import torch.nn as nn
# import torch.nn.functional as F
# from matplotlib import pyplot as plt
# import seaborn as sns
from torch.nn import init


class Expert(nn.Module):
    """
    Attributes:
        expert1 (nn.Linear): First linear layer.
        expert2 (nn.Linear): Second linear layer.
        act (nn.GELU): Activation function.

    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(Expert, self).__init__()

        self.expert1 = nn.Linear(input_dim, hidden_dim)
        self.expert2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization and set biases to zero.
        """
        init.xavier_uniform_(self.expert1.weight)
        init.zeros_(self.expert1.bias)
        init.xavier_uniform_(self.expert2.weight)
        init.zeros_(self.expert2.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """

        x = self.act(self.expert1(x))
        x = self.dropout(x)
        x = self.expert2(x)

        return x


class AttentionGate(nn.Module):
    """
    Attributes:
        attention (nn.MultiheadAttention): Attention mechanism to compute expert weights.

    """
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super(AttentionGate, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, expert_outputs):
        """
        Args:
            expert_outputs (torch.Tensor): Expert outputs of shape (batch_size, num_experts, output_dim).

        Returns:
            torch.Tensor: Attention weights of shape (batch_size, num_experts, output_dim).
        """
        # Reshape expert outputs for attention input
        expert_outputs_t = expert_outputs.permute(1, 0, 2)  # (num_experts, batch_size, output_dim)
        attention_weights, _ = self.attention(expert_outputs_t, expert_outputs_t, expert_outputs_t)
        attention_weights = attention_weights.permute(1, 0, 2)  # (batch_size, num_experts, output_dim)
        return attention_weights


class MixOfExperts(nn.Module):
    """
    A Mixture of Experts (MoE) module that combines the outputs of multiple expert modules using attention-based weighting.

    Attributes:
        num_experts (int): Number of expert modules.
        experts (nn.ModuleList): List of expert modules.
        attention (nn.MultiheadAttention): Attention mechanism to compute expert weights.
        projection (nn.Linear): Final linear layer to project the combined expert outputs.

    """
    def __init__(self, num_experts, se3_embed_dim, so3_embed_dim, hidden_dim=128, output_dim=1, dropout=0.1):
        super(MixOfExperts, self).__init__()
        self.num_experts = num_experts

        # Create expert modules with specified input and hidden dimensions
        self.experts = nn.ModuleList([
            Expert(se3_embed_dim, hidden_dim, output_dim, dropout=dropout),
            Expert(so3_embed_dim, hidden_dim, output_dim, dropout=dropout),
        ])

        # Attention mechanism to compute expert weights
        self.attention_gate = AttentionGate(embed_dim=output_dim, num_heads=1, dropout=dropout)

        # Final linear layer to project the combined expert outputs
        self.projection = nn.Linear(output_dim * num_experts, output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the linear layers using Xavier uniform initialization and set biases to zero.
        """
        init.xavier_uniform_(self.projection.weight)
        init.zeros_(self.projection.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (list of torch.Tensor): List of input tensors for each expert.

        Returns:
            torch.Tensor: Final output tensor of shape (batch_size, output_dim).
            torch.Tensor: Combined expert embeddings of shape (batch_size, output_dim * num_experts).
        """

        expert_outputs = torch.stack([
            expert(input_feature) for expert, input_feature in zip(self.experts, inputs)
        ], dim=1)  # (batch_size, num_experts, output_dim)

        # Compute expert weights using attention gate
        attention_weights = self.attention_gate(expert_outputs)

        # Apply attention weights to the expert outputs
        weighted_expert_outputs = expert_outputs * attention_weights  # (batch_size, num_experts, output_dim)

        # Combine weighted expert outputs
        expert_embeddings = weighted_expert_outputs.view(weighted_expert_outputs.size(0), -1)  # (batch_size, output_dim * num_experts)

        # Output
        output = self.projection(expert_embeddings)

        # return output, expert_embeddings
        return output
