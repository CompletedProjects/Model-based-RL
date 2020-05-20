#!/usr/bin/python

import gym
from ram_annotations import atari_dict


class InfoWrapper(gym.Wrapper): # Nawid - General wrapper for gym
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info(info)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def info(self, info):
        raise NotImplementedError

    def labels(self):
        raise NotImplementedError


class AtariARIWrapper(InfoWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env_name = self.env.spec.id
        game_name = self.env_name.split("-")[0].split("No")[0].split("Deterministic")[0]
        assert game_name.lower() in atari_dict, "{} is not currently supported by AARI. It's either not an Atari game or we don't have the ram annotations yet!".format(game_name)

    def info(self, info): # Nawid - Obtains dictionary of labels
        label_dict = self.labels()
        info["labels"] = label_dict # Nawid -places the labels in the info area
        return info

    def labels(self):
        ram = self.env.unwrapped.ale.getRAM() # Nawid-  Obtains the RAM interface which is the source code with all the different labels
        label_dict = ram2label(self.env_name, ram) # Nawid - Obtains the labels for a particular game
        return label_dict


def ram2label(env_name, ram):
    game_name = env_name.split("-")[0].split("No")[0].split("Deterministic")[0] # Nawid- Obtains name of game
    if game_name.lower() in atari_dict: # Nawid-  If name in actari dictionary
        label_dict = {k: ram[ind] for k, ind in atari_dict[game_name.lower()].items()} # Nawid - Looks at each k associates it with a Ram[ind] value
    else:
        assert False, "{} is not currently supported by AARI. It's either not an Atari game or we don't have the ram annotations yet!".format(game_name)
    return label_dict
