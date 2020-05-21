#!/usr/bin/python
import numpy as np
import pickle
import os
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import random_crop, center_crop_image, random_color_jitter


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape,action_shape,capacity, batch_size, device,image_size=84,frames = 4,transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.rand_crop = True
        self.transform = transform
        self.frames = frames


        obs_dtype = np.uint8 # Need to consider the sign of this
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity,*action_shape),dtype = np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def crop_control(self,rand_crop):
        if rand_crop:
            self.rand_crop = True
        else:
            self.rand_crop = False


    def add(self, obs,action,next_obs): # Nawid- Add information to replay buffer
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_obses[self.idx],next_obs)

        self.idx = (self.idx + 1) % self.capacity # This makes the data gets replaced in a recursive manner when the capacity is full
        self.full = self.full or self.idx == 0

    def sample_cpc(self): # Nawid - samples images I believe

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        ) # Used to randomly sample indices

        obses = self.obses[idxs] # Nawid - Samples observation
        pos = obses.copy() # Nawid -
        next_obses = self.next_obses[idxs]

        # Random crop or centre crops the image
        if self.rand_crop:
            obses_input = random_crop(obses,self.image_size)
            next_obses_input = random_crop(next_obses,self.image_size)
        else:
            obses_input = center_crop_image(obses, self.image_size)
            next_obses_input = centre_crop_image(next_obses,self.image_size)

        # Nawid - Crop images randomly
        obses_anc = random_crop(obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses_input, next_obses_input = np.transpose(obses_input, (0, 3, 1, 2)), np.transpose(next_obses_input, (0, 3, 1, 2))
        obses_anc, pos =  np.transpose(obses_anc, (0, 3, 1, 2)), np.transpose(pos, (0, 3, 1, 2))


        obses_input = torch.tensor(obses_input, device= self.device).float()/255
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        next_obses_input = torch.tensor(next_obses_input, device= self.device).float()/255
        obses_anc = torch.as_tensor(obses_anc, device=self.device).float()/255 # Random color jitter turns the values already into torch tenros
        pos = torch.as_tensor(pos, device=self.device).float()/255


        obses_anc = random_color_jitter(obses_anc,batch_size = self.batch_size,frames = self.frames)
        pos = random_color_jitter(pos,batch_size = self.batch_size,frames = self.frames)

        cpc_kwargs = dict(obs_anchor=obses_anc, obs_pos=pos,
                          time_anchor=None, time_pos=None) # Nawid  Postitive example is pos whilst anchor is obses

        return obses_input, actions, next_obses_input, cpc_kwargs

    def save(self, save_dir):
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.capacity))
        payload = [
            self.obses[0:self.capacity], #  Changed the payload compared to their training as I intend to use a offline training at the moment
            self.actions[0:self.capacity],
            self.next_obses[0:self.capacity]
        ]
        torch.save(payload, path)

    def load(self, save_dir): # Nawid - Loads the data into the replay buffer
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.actions[start:end] = payload[1]
            self.next_obses[start:end] = payload[2]
            self.idx = end

    def __getitem__(self, idx): # Nawid - Obtains item from replay buffer
        ''' Remove the randomness in the dataloading of each sample as the dataloader itself should be able to find the different values
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        '''

        obses = np.expand_dims(self.obses[idx],0) # Need to expand dim to allow it to be the shape for cropping, then need to squeeze so its a 4d tensor rather than 5d with an extra dim so it can be used with the dataloader
        next_obses = np.expand_dims(self.next_obses[idx],0)
        pos = obses.copy()

        #obs and next_obs
        if self.rand_crop:
            obses_input = random_crop(obses,self.image_size) #center_crop_image(obses,self.image_size) #
            next_obses_input = random_crop(next_obses,self.image_size)
        else:
            obses_input = center_crop_image(obses, self.image_size)
            next_obses_input = center_crop_image(next_obses,self.image_size)

        # random crop images
        obses_anc = random_crop(obses, self.image_size)
        pos = random_crop(pos, self.image_size)
        next_obses_anc = random_crop(next_obses,self.image_size) # Set anchor for the next observation in order to contrast with the contrastive loss

        # Squeeze shape
        obses_input = np.squeeze(obses_input)
        next_obses_input = np.squeeze(next_obses_input)
        obses_anc = np.squeeze(obses_anc)
        pos = np.squeeze(pos)
        next_obses_anc = np.squeeze(next_obses_anc)

        action = self.actions[idx]

        if self.transform:
            obses_input = self.transform(obses_input)
            next_obses_input = self.transform(next_obses_input)
            obses_anc = self.transform(obses_anc)
            pos = self.transform(pos)
            next_obses_anc = self.transform(next_obses_anc)


        cpc_kwargs = dict(obs_anchor=obses_anc, obs_pos=pos,next_obs_anchor= next_obses_anc) # Nawid  Postitive example is pos whilst anchor is obses
        return obses_input, action, next_obses_input, cpc_kwargs

    def __len__(self):
        return self.capacity
