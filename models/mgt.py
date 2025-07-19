# -*- coding: utf-8 -*-
# @Author : liang
# @File : mgt.py


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.so3.atoms import SO3_GraphEncoder, SO3_ProjectionHead
from models.se3.layers import SE3_GraphEncoder, SE3_ProjectionHead
from models.nnutils import NTXentLoss, AnglePredictionLayer, EdgePredictionLayer, scatter_
from models.moe.experts import MixOfExperts


class MGTransformer(nn.Module):

    """
    Multi-view Graph Transformer (MGT)

    This model combines SE(3) and SO(3) graph encoders with projection heads and a mixture of experts model.
    It supports two processes: pretraining (contrastive learning and position prediction) and finetuning (downstream tasks).
    """


    def __init__(self, config, config_model):

        super().__init__()

        # se3 invariant graph and se3 projection layer
        self.se3_graph_encoder = SE3_GraphEncoder(config_model['conv_layers'],
                                                  config_model['rbf_min'],
                                                  config_model['rbf_max'],
                                                  config_model['transformer'],
                                                  config_model['atom_input_features'],
                                                  config_model['fc_features'],
                                                  config_model['euclidean'],
                                                  config_model['num_heads'])


        self.se3_projection_encoder = SE3_ProjectionHead(config_model['graph_embed_dim'],
                                                        config_model['projection_dim'],
                                                        config_model['dropout'])

        # so3 equivariant graph and so3 projection layer
        self.so3_graph_encoder = SO3_GraphEncoder(config_model['conv_layers'],
                                                  config_model['rbf_min'],
                                                  config_model['rbf_max'],
                                                  config_model['atom_input_features'],
                                                  config_model['fc_features'],
                                                  config_model['euclidean'],
                                                  config_model['ns'],
                                                  config_model['nv'],
                                                  config_model['eno3'])

        self.so3_projection_encoder = SO3_ProjectionHead(config_model['graph_embed_dim'],
                                                         config_model['projection_dim'],
                                                         config_model['dropout'])

        # pretraining or finetune
        self.task = config['task']

        if self.task == 'pretraining':
            self.ntxentloss = NTXentLoss(config['device'],
                                         config['batch_size'],
                                         config_model['temperature'],
                                         config_model['use_cosine_similarity'])

            self.lambda_ = config_model['lambda_']
            self.use_cosine_similarity = config_model['use_cosine_similarity']  # whether to use cosine similarity in NT-Xent loss (i.e. True/False)
            self.temperature = config_model['temperature']
            self.batch = config['batch_size']
            self.device = config['device']

            # Angle prediction layer
            self.se3_angle_layer = AnglePredictionLayer(
                config_model['projection_dim'], config_model['position_dim']
            )

            # Edge prediction layer
            self.so3_edge_layer = EdgePredictionLayer(
                config_model['projection_dim'], config_model['position_dim']
            )


        elif self.task == 'finetune':

            # self.se3_embeddings = []
            # self.so3_embeddings = []
            # self.expert_embeddings = []

            # Mix of expert (MOE) model
            self.mixture_of_experts = MixOfExperts(config_model['num_experts'],
                                                   config_model['projection_dim'],
                                                   config_model['projection_dim'],
                                                   config_model['hidden_dim'],
                                                   config_model['output_dim'],
                                                   config_model['dropout'],)


        else:
            raise ValueError("Invalid task. Task must be either 'pretraining' or 'finetune'.")

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * config_model['text_encoder_num_layers']))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model. Subtract the position embeddings
        by default. The token embeddings are always included, since they are used in the
        final layer due to weight tying.

        :param non_embedding: whether to subtract the position embeddings (default is True)
        :returns: the number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.Encoder.model.wpe.weight.numel()
        return n_params


    def forward(self, se3_graph_batch, so3_graph_batch):
        """
        Args:
            se3_graph_batch: SE(3) graph batch
            so3_graph_batch: SO(3) graph batch

        Returns:
            Depending on the task, returns either the loss (pretraining) or the predicted output (finetune).
        """

        # Extract SE(3) and SO(3) graph features
        se3_graph_pool = self.se3_graph_encoder(se3_graph_batch)
        se3_graph_embeddings = self.se3_projection_encoder(se3_graph_pool)

        # graph_embeddings = prompt['graph_embed']
        so3_graph_pool = self.so3_graph_encoder(so3_graph_batch)
        so3_graph_embeddings = self.so3_projection_encoder(so3_graph_pool)

        # cross embeddings
        # cross_embeddings = self.cross_mixture_of_experts(text_embeddings * graph_embeddings)

        if self.task == 'pretraining':
            # Calculate pretraining loss

            loss = self.get_pretrained_loss(se3_graph_embeddings, so3_graph_embeddings,
                                            se3_graph_batch, so3_graph_batch)

            return loss

        elif self.task == 'finetune':
            # Combine SE(3) and SO(3) embeddings using the mixture of experts model

            y_pred = (
                self.mixture_of_experts((se3_graph_embeddings, so3_graph_embeddings)))

            y_pred = torch.squeeze(y_pred,
                                   dim=-1)

            return y_pred

        else:
            return None

    def get_pretrained_loss(self, se3_graph_embeddings, so3_graph_embeddings, se3_graph_batch, so3_graph_batch):
        """
        Calculate the pretraining loss, including position prediction and contrastive learning losses.

        Args:
            se3_graph_batch: SE(3) graph batch
            so3_graph_batch: SO(3) graph batch
            se3_graph_embeddings: SE(3) graph embeddings
            so3_graph_embeddings: SO(3) graph embeddings

        Returns:
            The total pretraining loss.
        """

        (angle,
         angle_noise) = se3_graph_batch.origin_angle, se3_graph_batch.edge_nei_angle

        (edge,
         edge_noise) =  so3_graph_batch.origin_edge_attr, so3_graph_batch.edge_attr

        # SE3 denoising loss
        se3_pred_noise = self.se3_angle_layer(se3_graph_embeddings)
        se3_noise = scatter_(se3_graph_batch, angle, angle_noise)
        se3_loss = F.mse_loss(se3_pred_noise, se3_noise, reduction='sum')

        # SO3 denoising loss
        so3_pred_noise = self.so3_edge_layer(so3_graph_embeddings)
        so3_noise =  scatter_(so3_graph_batch, edge, edge_noise)
        so3_loss = F.mse_loss(so3_pred_noise, so3_noise, reduction='sum')

        # Contrastive learning loss to learn the correlations between SE(3) and SO(3) embeddings
        contrast_loss = self.ntxentloss(se3_graph_embeddings, so3_graph_embeddings)

        # Combine the losses with respective weights (lambda_)
        loss = (self.lambda_[0] * contrast_loss +  # Weighted contrastive loss
                self.lambda_[1] * se3_loss +  # Weighted SE(3) angle prediction loss
                self.lambda_[2] * so3_loss)  # Weighted SO(3) position prediction loss

        return loss
