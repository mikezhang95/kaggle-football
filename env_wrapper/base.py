

import gym

class BaseEnv(gym.Env):

    def __init__(self, config=None):
        super(BaseEnv, self).__init__()
        return

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
