#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import make_encoder
from utils import weight_init


class CURL(nn.Module): # Nawid - Module for the contrastive loss
    """
    CURL
    """
    def __init__(self,encoder_type, obs_shape, z_dim, encoder_feature_dim, num_layers=4, num_filters=32, downsample = True):
        super(CURL, self).__init__()

        # Need to fix the encoders since I do not plan to use the critics
        self.encoder = make_encoder( # Nawid - Encoder of critic which is also used for the contrastive loss
            encoder_type,obs_shape, encoder_feature_dim, num_layers=num_layers,
            num_filters=num_filters, downsample = downsample, output_logits=True)

        self.encoder_target = make_encoder( # Nawid - Encoder of critic which is also used for the contrastive loss
            encoder_type,obs_shape, encoder_feature_dim, num_layers=num_layers,
            num_filters=num_filters, downsample = downsample, output_logits=True)

        self.encoder_target.load_state_dict(self.encoder.state_dict()) # copies the parameters of the encoder into the target encoder which is changing slowly
        self.W = nn.Parameter(torch.rand(z_dim, z_dim)) # Nawid - weight vector for the bilinear product

        self.apply(weight_init)

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos): # Nawid -  computes logits for contrastive loss
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class Dynamics_model(nn.Module):
    ''' MLP network'''
    def __init__(self, obs_shape,hidden_dim,
                 output_dim , encoder_feature_dim, num_layers, num_filters):
        super(Dynamics_model,self).__init__()

        self.encoder = make_encoder(obs_shape, encoder_feature_dim,num_layers,num_filters,output_logits=True)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + n_actions, hidden_dim),nn.ReLU(), # Size of the input is related to the encoder output as well as the concatenated one hot vector for the actions
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W = nn.Parameter(torch.rand(encoder_feature_dim, hidden_dim)) # Nawid - weight vector for the bilinear product

        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.apply(weight_init)

    def forward(self, obs,aux, detach_encoder = False):
        embedding = self.encoder(obs, detach = detach_encoder)
        concat_embed = torch.cat((embedding, aux), 1) # Join vectors along this dimension
        next_embed = self.trunk(concat_embed)
        skip_connection = torch.matmul(embedding,self.W) # linear mapping of the current embedding to make it the same shape so that the skip connection can be used
        next_embed = next_embed + skip_connection
        next_embed = F.ReLU(next_embed) #  Relu after skip connection is added to add non-linear behaviour- may need to remove if I want it to be controllable by linear dynamics
        next_embed = self.output_linear(next_embed)
        return next_embed
