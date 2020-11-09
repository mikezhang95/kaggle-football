
import gfootball.env as football_env
from env_wrapper import BaseEnv

"""
Wrap google football environment. Set parameters from: 
    - env_name: 
    - stacked:
    - representation: 
    - reward_experiment: []
"""

class GFootballEnv(BaseEnv):
    
    def __init__(self, args):
        super(GFootballEnv, self).__init__()

        # wrap the original environment
        self.env = football_env.reate_environment(
            env_name=args.level,
            stacked=False, # set to False to align with online evaluation
            representation="raw", # set to raw to align with online evaluation
            rewards=args.reward_experiment,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None, # set to train against other players
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0)  

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.obs_stack = deque([], maxlen=4)

    def _transform_obs(self, raw_obs):
        obs = raw_obs[0]
        obs = observation_preprocessing.generate_smm([obs])
        if not self.obs_stack:
            self.obs_stack.extend([obs] * 4)
        else:
            self.obs_stack.append(obs)
        obs = np.concatenate(list(self.obs_stack), axis=-1)
        obs = np.squeeze(obs)
        return obs


    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        obs = self._transform_obs(obs)
        return obs, float(reward), done, info


    def reset(self):
        self.obs_stack.clear()
        obs = self.env.reset()
        obs = self._transform_obs(obs)
        return obs





