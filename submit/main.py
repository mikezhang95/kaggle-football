
import sys
sys.path.append('/kaggle_simulations/agent/')
# sys.path.append('./')
import os
os.environ['CUDA_DEVCIE_ORDER'] = "PCR_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from env_wrapper import GFootballEnv

load_dir = "/kaggle_simulations/agent/ppo_gfootball.pt"
# load_dir = "ppo_gfootball.pt"
class EnvArgs(object):
    level = '11_vs_11_easy_stochastic'
    state = 'extracted_stacked'
    reward_experiment = 'scoring,checkpoints'

# environment
env_args = EnvArgs()
eval_env = GFootballEnv(env_args) # for evaluation
print("Environment created.")

# policy
policy = ActorCriticPolicy

# agent
model = PPO(policy, eval_env)
model.load(load_dir)
print("Agent loaded.")

# almost same as env_wrapper/gfootball.py when state="extracted_stacked"
def transform_obs(raw_obs):
    obs = raw_obs['players_raw']
    obs = eval_env._transform_obs(obs)
    return obs


# main function for agent
def agent(raw_obs):
    obs = transform_obs(raw_obs)
    action = model.predict(obs, deterministic=True)
    return [action] 

