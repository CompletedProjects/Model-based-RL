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

from custom_wrappers import custom_wrapper
from encoder import make_encoder
from EarlyStopping import EarlyStopping_loss
from GeneralFunctions import General_functions
from utils import make_dir, random_crop,center_crop_image, soft_update_params, weight_init
from torch.autograd import Variable
from DataCollection import Data_collection
from models import CURL, Dynamics_model
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
        frames,
        encoder_lr = 1e-4,
        encoder_tau = 0.001,
        encoder_feature_dim = 50, # This is the size of the embedding used for the
        num_layers=4,
        num_filters = 32,
        downsample = True,
        cpc_update_freq=1,
        encoder_type = 'pixel',
        encoder_update_freq = 1,
        random_jitter = True,

    ):
        self.device = device
        self.cpc_update_freq = cpc_update_freq
        self.image_size = obs_shape[-2] # Changed this to the numpy dimension
        self.frames = frames

        self.encoder_tau = encoder_tau
        self.epoch_step = 0
        self.encoder_update_freq = encoder_update_freq
        self.random_jitter = random_jitter

        self.CURL = CURL(encoder_type, obs_shape, encoder_feature_dim,
                         encoder_feature_dim,downsample = downsample, num_layers=num_layers, num_filters=num_filters).to(self.device)


        self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()

    def train(self, training = True):
        self.training = training
        self.CURL.train(training)

    def update(self, train_dataloader,val_dataloader,early_stopper):
        #torch.cuda.empty_cache() # Releases cache so the GPU has more memory
        if early_stopper.early_stop:
            print('early stopping')
            return

        for step, (obs, actions, next_obs, cpc_kwargs) in enumerate(train_dataloader):

            if step % self.encoder_update_freq == 0:
                soft_update_params(
                    self.CURL.encoder, self.CURL.encoder_target,
                    self.encoder_tau
                )
            if step % self.cpc_update_freq == 0:
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                self.update_cpc(obs_anchor, obs_pos) # Nawid -  Performs the contrastive loss I believe

        self.validation(val_dataloader,early_stopper)


    def update_cpc(self, obs_anchor, obs_pos):
        obs_anchor, obs_pos = obs_anchor.to(self.device), obs_pos.to(self.device)
        if self.random_jitter:
            obs_anchor, obs_pos = random_color_jitter(obs_anchor,batch_size = obs_anchor.shape[0],frames = self.frames), random_color_jitter(obs_pos,batch_size = obs_pos.shape[0],frames= self.frames)

        z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
        z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder

        logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        wandb.log({'Contrastive Training loss':loss.item()})

        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.cpc_optimizer.step()  # Nawid - Used to update the cpc

    def validation(self, dataloader,early_stopper):
        epoch_contrastive_loss = 0
        self.CURL.eval()
        with torch.no_grad():
            for i, (obses, actions, next_obses, cpc_kwargs) in enumerate(dataloader):
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obses, obs_anchor,obs_pos = obses.to(self.device), obs_anchor.to(self.device), obs_pos.to(self.device)
                if self.random_jitter:
                    obs_anchor, obs_pos =  random_color_jitter(obs_anchor,batch_size = obs_anchor.shape[0],frames = self.frames), random_color_jitter(obs_pos,batch_size = obs_pos.shape[0],frames= self.frames)

                ''' Code to check the appearance of the image
                image = obs_pos[0]
                image = image.permute(1, 2, 0)
                plt.imshow(image)
                plt.figure()
                plt.show()
                return
                '''
                actions, next_obses = actions.to(self.device), next_obses.to(self.device)
                z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
                z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder
                logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels)
                epoch_contrastive_loss += loss.item()

            average_epoch_contrastive_loss = epoch_contrastive_loss/(i+1)
            self.epoch_step += 1 # increase epoch counter
            wandb.log({'Contrastive Validation loss':average_epoch_contrastive_loss,'epoch': self.epoch_step})

            print('epoch:', self.epoch_step)
            early_stopper(average_epoch_contrastive_loss,self.CURL,self.cpc_optimizer)

        self.train()

def make_agent(obs_shape, device, dict_info):
    return CurlAgent(
        obs_shape = obs_shape,
        device = device,
        frames = dict_info['frames'],
        random_jitter = dict_info['random_jitter'],
        encoder_update_freq =dict_info['encoder_update_freq'],
        encoder_feature_dim = dict_info['encoder_feature_dim'], #  size of the embedding from the projection head
        encoder_lr = dict_info['encoder_lr'],
        encoder_tau = dict_info['encoder_tau'],
        encoder_type = dict_info['encoder_type'],
        num_layers = dict_info['num_layers'],
        num_filters = dict_info['num_filters'], # num of conv filters
        downsample = dict_info['downsample']
    )


ENV_NAME = 'MsPacmanDeterministic-v4'
n_actions = 4 #9 - Nawid - Change to 5 actions as the 4 other actions are simply copies of the other actions, therefore 5 actions should lower the amount of data needed.
'''
data_transform =  transforms.Compose([
        transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1),
        transforms.ToTensor()])
'''
data_transform = transforms.Compose([
                                    transforms.ToTensor()])

no_agents = 5
state_space = no_agents*2

parse_dict= {'pre_transform_image_size':100,
             'image_size':84,
             'frame_stack':False,
             'frames': 1,
             'state_space':state_space,
             'train_capacity':50000,
             'val_capacity':20000,
             'num_train_epochs':20,
             'batch_size':512,
             'random_crop': True,
             'encoder_update_freq':1,
             'encoder_feature_dim':50,
             'encoder_lr':1e-3,
             'encoder_tau':0.05, # value used for atari experiments in curl
             'num_layers':4,
             'num_filters':32,
             'downsample':True,
             'encoder_type':'Impala',
             'grayscale': False,
             'load_pretrain_model': False,
             'walls_present':True,
             'pretrain_model':False,
             'save_data':False,
             'num_pretrain_epochs':25,
             'transform': data_transform,
             'random_jitter':True
            }

#custom_name = 'rand_crop-' +str(parse_dict['random_crop'])  + '_gray-' + str(parse_dict['grayscale']) + '_walls-' +str(parse_dict['walls_present'])  + '_pretrain-' + str(parse_dict['pretrain_model'])
custom_name = 'Contrastive_hp_testing_random_jitter-'+str(parse_dict['random_jitter']) + 'frames-' +str(parse_dict['frames'])
wandb.init(entity="nerdk312",name=custom_name, project="Embed2Contrast",config = parse_dict)

possible_positions = np.load('possible_pacman_positions.npy',allow_pickle=True)

config = wandb.config

if parse_dict['load_pretrain_model']:
    config.pretrained_model = pretrain_model_dir

# Data collection
data_object = Data_collection(ENV_NAME,n_actions,possible_positions, parse_dict,parse_dict['train_capacity'])
val_data_object = Data_collection(ENV_NAME,n_actions,possible_positions, parse_dict, parse_dict['val_capacity'])

data_object.gather_random_trajectories(5000)
val_data_object.gather_random_trajectories(5000)

data_object.replay_buffer.crop_control(parse_dict['random_crop'])
val_data_object.replay_buffer.crop_control(parse_dict['random_crop'])

train_dataloader = DataLoader(data_object.replay_buffer, batch_size = parse_dict['batch_size'], shuffle = True)
val_dataloader = DataLoader(val_data_object.replay_buffer, batch_size = parse_dict['batch_size'], shuffle = True)



#test_info = [0.001,0.005,0.01,0.05,0.1,0.5,1]
#tests = len(test_info) + 1
tests = 1

for i in range(tests):
    print(i)
    if i >0:
        #parse_dict['encoder_tau'] = np.random.uniform(1e-4,1e-2)
        #parse_dict['encoder_lr'] = np.random.uniform(1e-3,1e-2)
        parse_dict['random_jitter'] = True
        parse_dict['encoder_tau'] = test_info[i-1]
        custom_name = 'Contrastive_hp_testing_random_jitter-'+str(parse_dict['random_jitter']) + '_encoder_tau-' +str(parse_dict['encoder_tau'])
        wandb.init(entity="nerdk312",name=custom_name, project="Contrastive_learning",config = parse_dict)

    agent = make_agent(
    obs_shape = data_object.obs_shape,
    device =data_object.device,
    dict_info = parse_dict
    )

    pretrain_model_name = 'Contrastive' +'_' + data_object.ts
    early_stopping_contrastive = EarlyStopping_loss(patience=3, verbose=True, wandb=wandb, name=pretrain_model_name)

    for step in range(parse_dict['num_train_epochs']):
        if early_stopping_contrastive.early_stop: #  Stops the training if early stopping counter is hit
            break
        agent.update(train_dataloader,val_dataloader,early_stopping_contrastive)

    wandb.join()
