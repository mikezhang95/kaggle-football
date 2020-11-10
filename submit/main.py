
import sys

sys.path.append('/kaggle_simulations/agent/')
import os
os.environ['CUDA_DEVCIE_ORDER'] = "PCR_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from env_wrapper import GFootballEnv

load_dir = "/kaggle_simulations/agent/ppo_gfootball.pt"
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
model.load(load_dir,device="cpu")
print("Agent loaded.")

# almost same as env_wrapper/gfootball.py when state="extracted_stacked"
def transform_obs(raw_obs):
    obs = raw_obs['players_raw']
    obs = eval_env._transform_obs(obs)
    return obs

# main function for agent
def agent(raw_obs):
    obs = transform_obs(raw_obs)
    action, state = model.predict(obs, deterministic=True)
    return [int(action)]


if __name__ == "__main__":

    # test
    obs = eval_env.reset()
    for i in range(5):
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step([int(action)])
        print(i, action, obs.shape, reward)
    print("Done")






