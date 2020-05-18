#!/usr/bin/python

from __future__ import print_function
import os
import pickle
import sys

import wandb

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .custom_wrappers import custom_wrapper
from .encoder import make_encoder
from .EarlyStopping import EarlyStopping_loss
from .GeneralFunctions import General_functions
from .utils import make_dir, random_crop,center_crop_image, soft_update_params, weight_init
from torch.autograd import Variable
from .DataCollection import Data_collection
from .models import CURL, Dynamics_model
import gym
import time

# Needed to create dataloaders
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CurlAgent(object):
    ''' CURL representation learning'''
    def __init__(
        self,
        obs_shape,
        device,
        hidden_dim = 256,
        output_dim = 2,
        encoder_feature_dim = 50,
        encoder_lr = 1e-3,
        encoder_tau = 0.005,
        num_layers=4,
        num_filters = 32,
        cpc_update_freq=1,
        dyn_update_freq= 2,
        encoder_update_freq = 2,
        detach_encoder=True,
    ):
        self.device = device
        self.cpc_update_freq = cpc_update_freq
        self.dyn_update_freq = dyn_update_freq
        self.image_size = obs_shape[-2] # Changed this to the numpy dimension
        self.detach_encoder = detach_encoder
        self.encoder_update_freq = encoder_update_freq
        self.encoder_tau = encoder_tau
        self.epoch_step = 0
        self.pretrain_epoch_step = 0

        self.CURL = CURL(obs_shape, encoder_feature_dim,
                         encoder_feature_dim, num_layers, num_filters).to(self.device)

        self.Model = Dynamics_model(obs_shape,hidden_dim,output_dim,encoder_feature_dim,num_layers,num_filters).to(self.device)

        # tie encoders between and CURL and dynamics predictor - VERY IMPORTANT-  CAN REMOVE IF I WANT SEPARATE NETWORKS
        if self.detach_encoder: # If the encoder for the dynamics network is not being updated , then we only want to use the contrastive loss to update the network
            self.Model.encoder.copy_conv_weights_from(self.CURL.encoder)

        self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )

        self.dynamics_optimizer = torch.optim.Adam(self.Model.parameters(), lr= encoder_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss() # Nawid - Added this loss for the prediction
        self.train()

    def train(self, training = True):
        self.training = training
        self.CURL.train(training)
        self.Model.train(training)

    def update(self, train_dataloader,val_dataloader,early_stopper):
        #torch.cuda.empty_cache() # Releases cache so the GPU has more memory
        if early_stopper.early_stop:
            print('early stopping')
            return

        for step, (obs, actions, state_change, cpc_kwargs) in enumerate(train_dataloader):

            if step % self.encoder_update_freq == 0:
                soft_update_params(
                    self.CURL.encoder, self.CURL.encoder_target,
                    self.encoder_tau
                )

            if step % self.cpc_update_freq == 0:
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                self.update_cpc(obs_anchor, obs_pos,cpc_kwargs) # Nawid -  Performs the contrastive loss I believe

            if step % self.dyn_update_freq ==0:
                self.update_dynamics(obs, actions,state_change)


        self.validation(val_dataloader,early_stopper)


    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs):
        obs_anchor, obs_pos = obs_anchor.float().to(self.device), obs_pos.float().to(self.device)
        z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
        z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder

        logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        wandb.log({'Contrastive Training loss':loss.item()})

        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.cpc_optimizer.step()  # Nawid - Used to update the cpc

    def update_dynamics(self, obs,actions, labels):
        obs, actions, labels = obs.float().to(self.device), actions.float().to(self.device), labels.float().to(self.device)

        prediction = self.Model(obs,actions,detach_encoder=self.detach_encoder) # gradient not backpropagated to the encoder
        prediction_loss = self.MSE_loss(prediction,labels)
        wandb.log({'Dynamics Training loss':prediction_loss.item()}) #  Need to use .item otherwise the loss will still be kept which will reduce the memory on the GPU

        self.dynamics_optimizer.zero_grad()
        prediction_loss.backward()
        self.dynamics_optimizer.step()

    def validation(self, dataloader,early_stopper):
        epoch_contrastive_loss = 0
        epoch_dynamics_loss = 0
        self.Model.eval()
        self.CURL.eval()
        with torch.no_grad():
            for i, (obses, actions, state_change, cpc_kwargs) in enumerate(dataloader):
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obses, obs_anchor,obs_pos = obses.float().to(self.device), obs_anchor.float().to(self.device), obs_pos.float().to(self.device)
                actions, state_change = actions.to(self.device), state_change.to(self.device)

                z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
                z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder
                logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels)
                epoch_contrastive_loss += loss.item()

                val_prediction = self.Model(obs_anchor,actions,detach_encoder=self.detach_encoder) # gradient not backpropagated to the encoder
                val_prediction_loss = self.MSE_loss(val_prediction,state_change)
                epoch_dynamics_loss += val_prediction_loss.item()

            average_epoch_contrastive_loss = epoch_contrastive_loss/(i+1)
            average_epoch_dynamics_loss = epoch_dynamics_loss/(i+1)

            self.epoch_step += 1 # increase epoch counter
            wandb.log({'Contrastive Validation loss':average_epoch_contrastive_loss,'Dynamics Validation loss':average_epoch_dynamics_loss,'epoch': self.epoch_step})
            print('epoch:', self.epoch_step)
            print('val prediction:', val_prediction[0:20]) #  batch of the val prediction is the size of the last batch so it will be what is leftover till the set is complete
            print('state change:',state_change[0:20])

            early_stopper(average_epoch_dynamics_loss,self.Model,self.dynamics_optimizer)

        self.train()


    def pretrain(self,train_dataloader,val_dataloader, pretrain_early_stopper):
        if pretrain_early_stopper.early_stop:
            print('early stopping pretrained model')
            return

        for pretrain_step, (_, _, state_change, cpc_kwargs) in enumerate(train_dataloader):
            if pretrain_step % self.encoder_update_freq == 0:
                soft_update_params(
                    self.CURL.encoder, self.CURL.encoder_target,
                    self.encoder_tau
                )

            if pretrain_step % self.cpc_update_freq == 0:
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                self.update_cpc(obs_anchor, obs_pos,cpc_kwargs) # Nawid -  Performs the contrastive loss I believe


        self.pretrain_val(val_dataloader,pretrain_early_stopper)

    def pretrain_val(self, dataloader,pretrain_early_stopper):
        epoch_pretrain_contrastive_loss = 0
        self.CURL.eval()
        with torch.no_grad():
            for i, (_, _, _, cpc_kwargs) in enumerate(dataloader):
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obs_anchor,obs_pos = obs_anchor.float().to(self.device), obs_pos.float().to(self.device)

                z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
                z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder
                logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels)
                epoch_pretrain_contrastive_loss += loss.item()

            self.pretrain_epoch_step += 1
            average_epoch_pretrain_contrastive_loss = epoch_pretrain_contrastive_loss/(i+1)
            self.pretrain_epoch_step += 1 # increase epoch counter
            wandb.log({'Pretrain Contrastive Validation loss':average_epoch_pretrain_contrastive_loss,'pretrain epoch': self.pretrain_epoch_step})
            pretrain_early_stopper(average_epoch_pretrain_contrastive_loss,self.CURL,self.cpc_optimizer)

def make_agent(obs_shape, device, dict_info):
    return CurlAgent(
        obs_shape = obs_shape,
        device = device,
        output_dim = dict_info['state_space'],
        encoder_update_freq =dict_info['encoder_update_freq'],
        dyn_update_freq =dict_info['dynamics_update_freq'],
        encoder_feature_dim = dict_info['encoder_feature_dim'],
        encoder_lr = dict_info['encoder_lr'],
        encoder_tau = dict_info['encoder_tau'],
        num_layers = dict_info['num_layers'],
        num_filters = dict_info['num_filters'],
        detach_encoder =dict_info['detach_dyn_encoder']
    )

ENV_NAME = 'MsPacmanDeterministic-v4'
n_actions = 4 #9 - Nawid - Change to 5 actions as the 4 other actions are simply copies of the other actions, therefore 5 actions should lower the amount of data needed.
no_agents = 2
output_dim = no_agents*2
parse_dict= {'pre_transform_image_size':100,
             'image_size':84,
             'frame_stack':4,
             'state_space':output_dim,
             'train_capacity':100000,
             'val_capacity':20000,
             'num_train_epochs':30,
             'batch_size':256,
             'random_crop': True,
             'encoder_update_freq':2,
             'encoder_feature_dim':50,
             'encoder_lr':1e-3,
             'encoder_tau':0.05,
             'num_layers':4,
             'num_filters':32,
             'load_trajectories':False,
             'grayscale': False,
             'load_pretrain_model': False,
             'walls_present':False,
             'dynamics_update_freq': 2,
             'detach_dyn_encoder':True,
             'pretrain_model':False,
             'save_data':False,
             'num_pretrain_epochs':50
            }

custom_name = str(no_agents)+ '_agents' + '_rand_crop-' +str(parse_dict['random_crop'])  + '_gray-' + str(parse_dict['grayscale']) + '_walls-' +str(parse_dict['walls_present']) + '_detach_encoder-' + str(parse_dict['detach_dyn_encoder']) + '_pretrain-' + str(parse_dict['pretrain_model'])
wandb.init(entity="nerdk312",name=custom_name, project="Contrastive_learning",config = parse_dict)

possible_positions = np.load('/possible_pacman_positions.npy',allow_pickle=True)
preloaded_train_data_1 = '/content/drive/My Drive/MsPacman-data/12-05_18:55_capacity-25000_grayscale-False_walls_present-False'
preloaded_train_data_2 = '/content/drive/My Drive/MsPacman-data/12-05_18:58_capacity-25000_grayscale-False_walls_present-False'
preloaded_train_data_3 = '/content/drive/My Drive/MsPacman-data/12-05_19:02_capacity-25000_grayscale-False_walls_present-False'
preloaded_val_data = '/content/drive/My Drive/MsPacman-data/12-05_19:05_capacity-20000_grayscale-False_walls_present-False'

config = wandb.config
if parse_dict['load_trajectories']:
    config.loaded_train_trajectories_1 = preloaded_train_data_1
    config.loaded_train_trajectories_2 = preloaded_train_data_2
    config.loaded_train_trajectories_3 = preloaded_train_data_3
    config.loaded_val_trajectories = preloaded_val_data


if parse_dict['load_pretrain_model']:
    config.pretrained_model = pretrain_model_dir


# Data collection
data_object = Data_collection(ENV_NAME,n_actions,possible_positions, parse_dict,parse_dict['train_capacity'])
val_data_object = Data_collection(ENV_NAME,n_actions,possible_positions, parse_dict, parse_dict['val_capacity'])
if parse_dict['load_trajectories']:
    data_object.replay_buffer.load(preloaded_train_data_1)
    val_data_object.replay_buffer.load(preloaded_val_data)
else:
    data_object.gather_random_trajectories(5000)
    val_data_object.gather_random_trajectories(5000)

dyn_model_name = 'Dynamics'+ '_' + data_object.ts
pretrain_model_name = 'Contrastive' +'_' + data_object.ts

# Order is pretrain, random_crop, detach_encoder
test_info = (False, True, False), (True, True, True), (False, False, True),(False, False, False), (True, False, True)
tests = len(test_info) + 1

for i in range(tests):
    print(i)
    if i >0:
        parse_dict['pretrain_model'], parse_dict['random_crop'], parse_dict['detach_dyn_encoder'] = test_info[i-1][0], test_info[i-1][1], test_info[i-1][2]
        custom_name = str(no_agents)+ '_agents' + '_rand_crop-' +str(parse_dict['random_crop'])  + '_gray-' + str(parse_dict['grayscale']) + '_walls-' +str(parse_dict['walls_present']) + '_detach_encoder-' + str(parse_dict['detach_dyn_encoder']) + '_pretrain-' + str(parse_dict['pretrain_model'])
        wandb.init(entity="nerdk312",name=custom_name, project="Contrastive_learning",config = parse_dict)

    data_object.replay_buffer.crop_control(parse_dict['random_crop'])
    val_data_object.replay_buffer.crop_control(parse_dict['random_crop'])

    train_dataloader = DataLoader(data_object.replay_buffer, batch_size = 256, shuffle = True)
    val_dataloader = DataLoader(val_data_object.replay_buffer, batch_size = 256, shuffle = True)

    agent = make_agent(
            obs_shape = data_object.obs_shape,
            device =data_object.device,
            dict_info = parse_dict
        )

    early_stopping_dynamics = EarlyStopping_loss(patience=3, verbose=True, wandb=wandb, name=dyn_model_name)
    early_stopping_contrastive = EarlyStopping_loss(patience=3, verbose=True, wandb=wandb, name=pretrain_model_name)
    env = gym.make(ENV_NAME)
    env = custom_wrapper(env, grayscale = parse_dict['grayscale'])
    obs = env.reset()
    info_labels = env.labels()
    state = data_object.state_conversion(info_labels)

    if parse_dict['pretrain_model']:
        for pretrain_step in range(parse_dict['num_pretrain_epochs']):
            if early_stopping_contrastive.early_stop: #  Stops the training if early stopping counter is hit
                break
            agent.pretrain(train_dataloader,val_dataloader, early_stopping_contrastive)

    for step in range(parse_dict['num_train_epochs']):
        if early_stopping_dynamics.early_stop: #  Stops the training if early stopping counter is hit
            break
        agent.update(train_dataloader,val_dataloader,early_stopping_dynamics)

    wandb.join()
