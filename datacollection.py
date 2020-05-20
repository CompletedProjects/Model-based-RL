#!/usr/bin/python

import numpy as np
from utils import make_dir
from generalfunctions import General_functions
from custom_wrappers import custom_wrapper
from replay_buffer import ReplayBuffer
import gym
import time

class Data_collection(General_functions):
    def __init__(self, ENV_NAME, n_actions, possible_positions,info_dict,buffer_capacity):
        super(Data_collection, self).__init__(ENV_NAME, n_actions, possible_positions,info_dict)
        self.grayscale = info_dict['grayscale']
        self.channels = 1 if self.grayscale else 3
        self.frame_stack = info_dict['frame_stack']
        self.frames = info_dict['frames']

        self.obs_shape = (info_dict['image_size'], info_dict['image_size'],self.channels*self.frames) # Nawid - Stack together images (multiply of 3 present due to 3 channels (RGB))
        self.pre_aug_obs_shape = (info_dict['pre_transform_image_size'],info_dict['pre_transform_image_size'],self.channels*self.frames)
        self.action_shape = (n_actions,)
        self.buffer_capacity = buffer_capacity
        self.replay_buffer  = ReplayBuffer(self.pre_aug_obs_shape,self.action_shape,self.buffer_capacity, info_dict['batch_size'],self.device, info_dict['image_size'],info_dict['frames'],info_dict['transform'])
        self.ts = time.gmtime()
        self.ts = time.strftime('%d-%m_%H:%M')
        self.walls_present = info_dict['walls_present']
        self.save_data = info_dict['save_data']


    def random_action_selection(self, state,prev_action = None):
        while True:
            action = np.random.randint(1,5)
            feasible_action = self.next_position(state,action)
            if feasible_action:
                self.prev_action_counter = True
                return action, None
            else:
                infeasible_action_one_hot = self.one_hot(action)
                if self.prev_action_counter and prev_action !=None:
                    return prev_action, infeasible_action_one_hot

    def gather_random_trajectories(self,num_traj):
        if self.save_data:
            work_dir = '/content/drive/My Drive/MsPacman-data' + '/' + self.ts +'_capacity-' + str(self.buffer_capacity) +'_grayscale-'+ str(self.grayscale) + '_walls_present-'+ str(self.walls_present)
            work_dir = make_dir(work_dir)

        for n in range(num_traj):
            if n % 10 ==0:
                print('trajectory number:',n)
                # Initial set up
            #self.env.seed(0)

            self.env = gym.make(self.ENV_NAME) # Due to error in code, I reinstantiate the env each time
            self.env = custom_wrapper(self.env,grayscale = self.grayscale,frame_stack=self.frame_stack,frames = self.frames)
            obs = self.env.reset()

            self.repeated_end = False
            info_labels = self.env.labels() # Nawid - Used to get the current state
            state = self.state_conversion(info_labels) # Used to get the initial state
            prev_action = None # Initialise prev action has having no action

            while True:
                sampled_action, infeasible_action_one_hot = self.random_action_selection(state,prev_action)
                sampled_action_one_hot = self.one_hot(sampled_action)

                next_obs, reward, done, next_info = self.env.step(sampled_action)
                next_info_labels = next_info['labels']

                next_state = self.state_conversion(next_info_labels)
                state_change = next_state -  state

                self.check_state(state,next_state)
                self.check_all_agents(info_labels, next_info_labels) # need to use the info labels to predict the state as the info labels have all the informaiton

                if not self.repeated_end:
                    if infeasible_action_one_hot is not None and self.walls_present:
                        fake_next_state = np.zeros_like(state) #  Need to instantiate a new version each time to prevent updating a single variable which will affect all places(eg lists) where the variable is added
                        fake_next_state[0:-2] = next_state[0:-2].copy() # the enemy position of the fake next state is the current enemy position
                        fake_next_state[-2:] = state[-2:].copy() # The agent position for the fake next state is the state before any action was taken
                        fake_state_change = fake_next_state - state
                        self.replay_buffer.add(obs,infeasible_action_one_hot, next_obs) # THERE IS NOTHING SUCH AS A FAKE NEXT_OBS SINCE IT IS A IMAGE - THE CLOSEST THING WOULD BE AN ACTION WHERE NOTHING OCCURS

                    self.replay_buffer.add(obs,sampled_action_one_hot, next_obs)
                else:
                    done = True

                obs = next_obs # do not need to copy as a new variable of obs is instantiated at each time step.
                state = next_state.copy()
                info_labels = next_info_labels.copy()
                prev_action = sampled_action

                if done:
                    break
                if self.replay_buffer.full:
                    if self.save_data:
                        self.replay_buffer.save(work_dir)
                    return
