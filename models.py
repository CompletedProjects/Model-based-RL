#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import make_encoder
from utils import weight_init

n_actions = 4
class CURL(nn.Module): # Nawid - Module for the contrastive loss
    """
    CURL
    """
    def __init__(self,obs_shape, z_dim, encoder_feature_dim,hidden_dim, downsample = True):
        super(CURL, self).__init__()

        # Need to fix the encoders since I do not plan to use the critics
        self.encoder = make_encoder( # Nawid - Encoder of critic which is also used for the contrastive loss
            obs_shape, encoder_feature_dim, downsample = downsample)

        # Encoder target required for momentum encoding
        self.encoder_target = make_encoder( # Nawid - Encoder of critic which is also used for the contrastive loss -  Momentum encoder
            obs_shape, encoder_feature_dim, downsample = downsample)

        self.encoder_target.load_state_dict(self.encoder.state_dict()) # copies the parameters of the encoder into the target encoder which is changing slowly
        self.W = nn.Parameter(torch.rand(z_dim, z_dim)) # Nawid - weight vector for the bilinear product

        # Part related to the dynamics
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + n_actions, hidden_dim),nn.ReLU(), # Size of the input is related to the encoder output as well as the concatenated one hot vector for the actions
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_skip = nn.Parameter(torch.rand(self.encoder.feature_dim, hidden_dim)) # Nawid - weight vector for the skip connection
        self.output_linear = nn.Linear(hidden_dim, z_dim)

        self.apply(weight_init)

    def encode(self, x, detach_map=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x,detach_feature_map = detach_map) # Nawid - This is used to detach the feature map

        '''
        if detach:
            print('detach being used')
            z_out = z_out.detach()
        else:
            print('detach not used')
        '''

        return z_out

    def encode_predicted(self, x, aux, detach_map = False, ema = False):
        '''
        Encoder: z_t+1 = e(x_t,action)
        :param x: x_t, image at current timestep, aux -  action taken
        :return : predicted zt+1
        '''
        '''
        if ema:
            with torch.no_grad():
                z_t = self.encoder_target(x) # Says whether to change the encoder value
        else:
            z_t =self.encoder(x)
        if detach:
        '''

        z_t = self.encode(x,detach_map = detach_map, ema=ema) #  obtains the embedding of the current state and I can choose to do this from the momentum encoder
        concat_zt = torch.cat((z_t, aux), 1) # Join vectors along this dimension
        z_next_t = self.trunk(concat_zt)
        skip_connection = torch.matmul(z_t,self.W_skip) # linear mapping of the current embedding to make it the same shape so that the skip connection can be used
        z_next_t = z_next_t + skip_connection
        z_next_t = F.relu(z_next_t)
        z_next_t = self.output_linear(z_next_t)
        return z_next_t

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
    def __init__(self,encoder,
                 z_dim ,hidden_dim):
        '''
        args
        encoder: encoder for the convolutional layers and fc layers to make the initial embedding ( can stop gradient from flowing past feature map layers initially)
        z_dim: latent space dimension (currently set to be the same as the dimensionality of the embedding)
        hidden_dim: dimensionality of the hidden states in the fc layers
        '''
        super(Dynamics_model,self).__init__()

        self.encoder = encoder

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + n_actions, hidden_dim),nn.ReLU(), # Size of the input is related to the encoder output as well as the concatenated one hot vector for the actions
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W = nn.Parameter(torch.rand(self.encoder.feature_dim, hidden_dim)) # Nawid - weight vector for the skip connection

        self.output_linear = nn.Linear(hidden_dim, z_dim)
        self.apply(weight_init)

    def forward(self, obs, aux, detach_encoder = False):
        embedding = self.encoder(obs, detach = detach_encoder)
        concat_embed = torch.cat((embedding, aux), 1) # Join vectors along this dimension
        next_embed = self.trunk(concat_embed)
        skip_connection = torch.matmul(embedding,self.W) # linear mapping of the current embedding to make it the same shape so that the skip connection can be used
        next_embed = next_embed + skip_connection
        next_embed = F.relu(next_embed) #  Relu after skip connection is added to add non-linear behaviour- may need to remove if I want it to be controllable by linear dynamics
        next_embed = self.output_linear(next_embed)
        return next_embed
