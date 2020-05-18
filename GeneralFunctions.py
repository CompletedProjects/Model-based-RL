#!/usr/bin/python

import torch
import gym
import numpy as np
from .custom_wrappers import custom_wrapper

class General_functions():
    def __init__(self, ENV_NAME, n_actions,possible_positions,info_dict):
        self.ENV_NAME = ENV_NAME
        self.env = gym.make(self.ENV_NAME)
        self.env = custom_wrapper(self.env)
        self.initial_info_labels = self.env.labels()
        self.state_shape = (info_dict['state_space'],)
        #self.key_list = ['enemy_blinky_x', 'enemy_blinky_y','player_x','player_y']
        self.key_list = ['enemy_blinky_x', 'enemy_blinky_y','enemy_pinky_x','enemy_pinky_y','enemy_inky_x','enemy_inky_y', 'enemy_sue_x', 'enemy_sue_y', 'player_x', 'player_y']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.prev_action_counter = False
        self.repeated_end = False
        self.n_actions = n_actions
        # possible positions is numpy array with most of the possible positions that the agent can go to
        self.possible_positions = possible_positions
        self.possible_positions_list = self.possible_positions.tolist()

    def one_hot(self,i):
        a = np.zeros(self.n_actions, 'uint8')
        a[i-1] = 1
        return a

    def state_conversion(self,info_labels):
        state = [info_labels[word] for word in self.key_list if word in info_labels]
        state = np.array(state).astype(np.float32) # changes to float 32
        return state

    def next_position(self,state, action):
        next_position = state[-2:].copy()
        if action == 1:
            next_position[1] = next_position[1] - 2
        elif action == 2:
            next_position[0] = next_position[0] + 2
        elif action == 3:
            next_position[0] = next_position[0] - 2
        elif action == 4:
            next_position[1] = next_position[1] + 2
        if next_position.tolist() in self.possible_positions_list: # possible positions will be a list which is fed into the network
            return True
        else:
            return False

    def check_all_agents(self,info_label,next_info_label):
        repeated = np.equal(info_label,next_info_label).all()
        if repeated:
            self.repeated_end= True

    def check_state(self,state, next_state):
        repeated_state = np.equal(state[-2:], next_state[-2:]).all()
        if repeated_state:
            self.prev_action_counter = False
