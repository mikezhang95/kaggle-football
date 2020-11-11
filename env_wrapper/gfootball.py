
import numpy as np
from collections import deque

import gfootball.env as football_env
from env_wrapper import BaseEnv

"""
Wrap google football environment. Set parameters from: 
    - level: e.x. 11_vs_11_easy_stochastic.py (full list: https://github.com/google-research/football/tree/master/gfootball/scenarios)
    - state: ['pixels', 'pixels_gray', 'extracted', 'simple115v2','raw']
    - reward_experiment: ['scoring', 'scoring,checkpoints']
"""

class GFootballEnv(BaseEnv):
    
    def __init__(self, args):
        super(GFootballEnv, self).__init__()

        self.representation = args.state.split('_')[0]
        self.stacked = 'stacked' in args.state

        # wrap the original environment
        # https://github.com/google-research/football/blob/master/gfootball/env/__init__.py
        # this is same as online environment
        self.raw_env = football_env.create_environment(
            env_name=args.level,
            stacked=False, # set to False to align with online evaluation
            representation="raw", # set to raw to align with online evaluation
            rewards=args.reward_experiment)
        self.obs_stack = deque([], maxlen=4)

        # this is for training environment
        self.real_env = football_env.create_environment(
            env_name=args.level,
            stacked=self.stacked, # set to False to align with online evaluation
            representation=self.representation,
            rewards=args.reward_experiment)

        self.action_space = self.real_env.action_space
        self.observation_space = self.real_env.observation_space

    def _transform_obs(self, raw_obs):

        # TODO: raw_obs how many
        if self.representation == "raw":
            return raw_obs

        if "simple115" in self.representation:
            raise NotImplementedError

        if "pixels" in self.representation:
            raise NotImplementedError

        if "extracted" in self.representation:
            obs = raw_obs[0]
            obs = football_env.observation_preprocessing.generate_smm([obs])
            if not self.obs_stack:
                self.obs_stack.extend([obs] * 4)
            else:
                self.obs_stack.append(obs)

        # stack observation to add time dependencies, only for pixels and extracted
        if self.stacked:
            obs = np.concatenate(list(self.obs_stack), axis=-1) # [72,96,4*4]
            obs = np.squeeze(obs)
        else:
            obs = np.array(list(self.obs_stack)[-1])
        return obs


    def step(self, action):
        obs, reward, done, info = self.raw_env.step([action])
        obs = self._transform_obs(obs)
        return obs, float(reward), done, info


    def reset(self):
        self.obs_stack.clear()
        obs = self.raw_env.reset()
        obs = self._transform_obs(obs)
        return obs





