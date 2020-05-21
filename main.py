#!/usr/bin/python

from __future__ import print_function
import os
import pickle
import sys

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True

if in_notebook(): # Checks whether I am in a jupyter notebook
    sys.path.append('/content/drive/My Drive/Embed_2_Contrast')

import wandb
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gym
import time
import matplotlib.pyplot as plt

from custom_wrappers import custom_wrapper
from encoder import make_encoder
from earlystopping import EarlyStopping_loss
from generalfunctions import General_functions
from utils import make_dir, random_crop,center_crop_image, soft_update_params, weight_init, random_color_jitter
from torch.autograd import Variable
from datacollection import Data_collection
from models import CURL, Dynamics_model
from replay_buffer import ReplayBuffer

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
        dynamics_hidden_dim = 256,
        downsample = True,
        cpc_update_freq=1,
        encoder_update_freq = 1,
        random_jitter = True,
        detach_encoder=True,
        dynamics_update_freq= 1
    ):
        self.device = device
        self.cpc_update_freq = cpc_update_freq
        self.dynamics_update_freq = dynamics_update_freq
        self.image_size = obs_shape[-2] # Changed this to the numpy dimension
        self.frames = frames
        self.detach_encoder =  detach_encoder

        self.encoder_tau = encoder_tau
        self.epoch_step = 0
        self.encoder_update_freq = encoder_update_freq
        self.random_jitter = random_jitter

        self.CURL = CURL(obs_shape, encoder_feature_dim,
                         encoder_feature_dim,hidden_dim=dynamics_hidden_dim,downsample = downsample).to(self.device)


        self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        '''
        self.Model = Dynamics_model(self.CURL.encoder, encoder_feature_dim,
                                    hidden_dim=dynamics_hidden_dim).to(self.device)


        self.dynamics_optimizer = torch.optim.Adam(
            self.Model.parameters(), lr =encoder_lr
        )
        '''
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss() # Nawid - Added this loss for the prediction
        self.train()

    def train(self, training = True):
        self.training = training
        self.CURL.train(training)
        #self.Model.train(training)

    def update(self, train_dataloader,val_dataloader,early_stopper_contrastive, early_stopper_dynamics):
        #torch.cuda.empty_cache() # Releases cache so the GPU has more memory
        if early_stopper_contrastive.early_stop or early_stopper_dynamics.early_stop:
            print('early stopping-Early stopping contrastive, Early stopping dynamics :',early_stopper_contrastive.early_stop, early_stopper_dynamics.early_stop)
            return

        for step, (obs, actions, next_obs, cpc_kwargs) in enumerate(train_dataloader):
            obs, actions, next_obs = obs.to(self.device),actions.to(self.device), next_obs.to(self.device)

            if step % self.encoder_update_freq == 0:
                soft_update_params(
                    self.CURL.encoder, self.CURL.encoder_target,
                    self.encoder_tau
                )
            if step % self.cpc_update_freq == 0:
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obs_anchor, obs_pos = obs_anchor.to(self.device), obs_pos.to(self.device)
                self.update_cpc(obs_anchor, obs_pos) # Nawid -  Performs the contrastive loss I believe


            if step % self.dynamics_update_freq ==0:
                self.update_dynamics(obs,actions,next_obs)

        self.validation(val_dataloader,early_stopper_contrastive, early_stopper_dynamics)

    def update_cpc(self, obs_anchor, obs_pos):
        if self.random_jitter:
            obs_anchor, obs_pos = random_color_jitter(obs_anchor,batch_size = obs_anchor.shape[0],frames = self.frames), random_color_jitter(obs_pos,batch_size = obs_pos.shape[0],frames= self.frames)

        z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
        z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder

        logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        wandb.log({'Contrastive Training loss':loss.item()})

        #frozen_weights = self.CURL.encoder.final_linear.weight.detach().clone()
        #frozen_weights = self.CURL.encoder.layer3[0].net[1].weight.detach().clone()

        self.cpc_optimizer.zero_grad()
        loss.backward()
        self.cpc_optimizer.step()  # Nawid - Used to update the cpc

    def update_dynamics(self,obs,action,next_obs):
        if self.random_jitter:
            obs, next_obs = random_color_jitter(obs,batch_size = obs.shape[0],frames = self.frames), random_color_jitter(next_obs,batch_size = next_obs.shape[0],frames= self.frames)

        next_zt = self.CURL.encode(next_obs)
        predicted_next_zt = self.CURL.encode_predicted(obs,action,ema=True) # only the embedding of the current state is made using the exponential moving average, the next latent state is obtained from the mapping

        logits = self.CURL.compute_logits(next_zt, predicted_next_zt)# next_zt is the anchor and predicted_next_zt is the positive example used
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits,labels)
        wandb.log({'Dynamics Training loss': loss.item()})

        self.cpc_optimizer.zero_grad()
        loss.backward()
        self.cpc_optimizer.step()


    '''
    def update_dynamics(self, obs,actions, next_obs):
        predicted_next_latent = self.Model(obs,actions,detach_encoder = self.detach_encoder) # only trains the fully connected part of the output, features from the encoder are not trained
        next_latent = self.CURL.encode(next_obs,detach=True) # no gradients will flow from this output
        prediction_loss = self.MSE_loss(predicted_next_latent,next_latent)
        wandb.log({'Dynamics Training loss':prediction_loss.item()}) #  Need to use .item otherwise the loss will still be kept which will reduce the memory on the GPU

        self.dynamics_optimizer.zero_grad()
        prediction_loss.backward()
        self.dynamics_optimizer.step()
    '''

    def validation(self, dataloader,early_stopper_contrastive, early_stopper_dynamics):
        epoch_contrastive_loss = 0
        epoch_dynamics_loss = 0
        self.CURL.eval()
        #self.Model.eval()
        with torch.no_grad():
            for i, (obs, actions, next_obs, cpc_kwargs) in enumerate(dataloader):
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obs, obs_anchor,obs_pos = obs.to(self.device), obs_anchor.to(self.device), obs_pos.to(self.device)
                actions, next_obs = actions.to(self.device), next_obs.to(self.device)
                if self.random_jitter:
                    obs_anchor, obs_pos = random_color_jitter(obs_anchor,batch_size = obs_anchor.shape[0],frames = self.frames), random_color_jitter(obs_pos,batch_size = obs_pos.shape[0],frames= self.frames)
                    obs, next_obs = random_color_jitter(obs,batch_size = obs.shape[0],frames = self.frames), random_color_jitter(next_obs,batch_size = next_obs.shape[0],frames= self.frames)

                ''' Code to check the appearance of the image
                image = obs_pos[0]
                image = image.permute(1, 2, 0)
                plt.imshow(image)
                plt.figure()
                plt.show()
                return
                '''
                actions, next_obs = actions.to(self.device), next_obs.to(self.device)
                z_a = self.CURL.encode(obs_anchor) # Nawid -  Encode the anchor
                z_pos = self.CURL.encode(obs_pos, ema=True) # Nawid- Encode the positive with the momentum encoder
                logits = self.CURL.compute_logits(z_a, z_pos) #  Nawid- Compute the logits between them
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                loss = self.cross_entropy_loss(logits, labels)
                epoch_contrastive_loss += loss.item()


                next_zt = self.CURL.encode(next_obs)
                predicted_next_zt = self.CURL.encode_predicted(obs,actions,ema=True)
                dynamics_logits = self.CURL.compute_logits(next_zt, predicted_next_zt)# next_zt is the anchor and predicted_next_zt is the positive example used
                dynamics_labels = torch.arange(logits.shape[0]).long().to(self.device)
                prediction_loss = self.cross_entropy_loss(dynamics_logits,dynamics_labels)
                epoch_dynamics_loss += prediction_loss.item()
                '''
                prediced_next_latent = self.Model(obs,actions,detach_encoder = self.detach_encoder) # only trains the fully connected part of the output, features from the encoder are not trained
                next_latent = self.CURL.encode(next_obs,detach=True) # no gradients will flow from this output
                prediction_loss = self.MSE_loss(prediced_next_latent,next_latent)
                epoch_dynamics_loss += prediction_loss.item()
                '''

            average_epoch_contrastive_loss = epoch_contrastive_loss/(i+1)
            average_epoch_dynamics_loss = epoch_dynamics_loss/(i+1)

            self.epoch_step += 1 # increase epoch counter
            #wandb.log({'Contrastive Validation loss':average_epoch_contrastive_loss, 'epoch': self.epoch_step})
            wandb.log({'Contrastive Validation loss':average_epoch_contrastive_loss, 'Dynamics Validation loss':average_epoch_dynamics_loss,'epoch': self.epoch_step})

            print('epoch:', self.epoch_step)
            early_stopper_contrastive(average_epoch_contrastive_loss,self.CURL,self.cpc_optimizer)
            early_stopper_dynamics(average_epoch_dynamics_loss, self.CURL,self.cpc_optimizer)
            #early_stopper_dynamics(average_epoch_dynamics_loss, self.Model, self.dynamics_optimizer)

        self.train()

def make_agent(obs_shape, device, dict_info):
    return CurlAgent(
        obs_shape = obs_shape,
        device = device,
        frames = dict_info['frames'],
        random_jitter = dict_info['random_jitter'],
        encoder_update_freq =dict_info['encoder_update_freq'],
        dynamics_update_freq =dict_info['dynamics_update_freq'],
        encoder_feature_dim = dict_info['encoder_feature_dim'], #  size of the embedding from the projection head
        encoder_lr = dict_info['encoder_lr'],
        encoder_tau = dict_info['encoder_tau'],
        downsample = dict_info['downsample'],
        dynamics_hidden_dim = dict_info['dynamics_hidden_dim'],
        detach_encoder = dict_info['detach_encoder']
    )
ENV_NAME = 'MsPacmanDeterministic-v4'
n_actions = 4

data_transform = transforms.Compose([
                                    transforms.ToTensor()])

no_agents = 5
state_space = no_agents*2
parse_dict= {'pre_transform_image_size':100,
             'image_size':84,
             'frame_stack':False,
             'frames': 1,
             'state_space':state_space,
             'train_capacity':100000,
             'val_capacity':20000,
             'num_train_epochs':20,
             'batch_size':128,
             'random_crop': True,
             'encoder_update_freq':1,
             'dynamics_update_freq':1,
             'encoder_feature_dim':128,
             'dynamics_hidden_dim': 256,
             'encoder_lr':1e-3,
             'encoder_tau':0.05, # value used for atari experiments in curl
             'downsample':True,
             'encoder_type':'Impala',
             'grayscale': False,
             'load_pretrain_model': False,
             'walls_present':True,
             'pretrain_model':False,
             'save_data':False,
             'num_pretrain_epochs':25,
             'transform': data_transform,
             'random_jitter':True,
             'detach_encoder':True
            }

#custom_name = 'rand_crop-' +str(parse_dict['random_crop'])  + '_gray-' + str(parse_dict['grayscale']) + '_walls-' +str(parse_dict['walls_present'])  + '_pretrain-' + str(parse_dict['pretrain_model'])
custom_name = 'Batch_size-' +str(parse_dict['batch_size'])
wandb.init(entity="nerdk312",name=custom_name, project="Embed2Contrast_Dynamics_Contrastive",config = parse_dict)

if in_notebook():
    possible_positions = np.load('/content/drive/My Drive/Embed_2_Contrast/possible_pacman_positions.npy',allow_pickle=True)
else:
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

# dataloader
train_dataloader = DataLoader(data_object.replay_buffer, batch_size = parse_dict['batch_size'], shuffle = True)
val_dataloader = DataLoader(val_data_object.replay_buffer, batch_size = parse_dict['batch_size'], shuffle = True)


test_info = [256,512,1024,2048]
tests = len(test_info) + 1
#tests = 6

#Training loop

for i in range(tests):
    print(i)
    if i >0:

        #parse_dict['encoder_tau'] = np.random.uniform(1e-3,1e-2)
        #parse_dict['encoder_lr'] = np.random.uniform(1e-4,1e-2)
        custom_name = 'Batch_size-' +str(parse_dict['batch_size'])
        wandb.init(entity="nerdk312",name=custom_name, project="Embed2Contrast_contrastive_dynamics",config = parse_dict)

    agent = make_agent(
    obs_shape = data_object.obs_shape,
    device =data_object.device,
    dict_info = parse_dict
    )

    pretrain_model_name = 'Contrastive' +'_' + data_object.ts
    dynamics_model_name = 'Dynamics' +'_' + data_object.ts

    early_stopping_contrastive = EarlyStopping_loss(patience=3, verbose=True, wandb=wandb, name=pretrain_model_name)
    early_stopping_dynamics = EarlyStopping_loss(patience=3, verbose=True, wandb=wandb, name=dynamics_model_name)

    for step in range(parse_dict['num_train_epochs']):
        if early_stopping_contrastive.early_stop or early_stopping_dynamics.early_stop: #  Stops the training if early stopping counter is hit
            break
        agent.update(train_dataloader,val_dataloader,early_stopping_contrastive,early_stopping_dynamics)

    wandb.join()
