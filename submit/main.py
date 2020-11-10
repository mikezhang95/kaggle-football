
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from env_wrapper import GFootballEnv


load_dir = "/kaggle_simulations/agent/ppo_gfootball.pt"
class EnvArgs(object):
    level = 'academy_empty_goal_clos'
    state = 'extracted_stacekd'
    reward_experiment = 'scoring,checkpoints'
env_args = EnvArgs()


eval_env = GFootballEnv(env_args) # for evaluation
policy = ActorCriticPolicy

model = PPO(policy, eval_env)
model.load(load_dir)


# same as env_wrapper/gfootball.py when state="extracted_stacked"
def transform_obs(raw_obs):

    global obs_stack
    obs = raw_obs['players_raw'][0]
    obs = football_env.observation_preprocessing.generate_smm([obs])
    if not obs_stack:
        obs_stack.extend([obs] * 4)
    else:
        obs_stack.append(obs)

    # stack observation to add time dependencies, only for pixels and extracted
    obs = np.concatenate(list(obs_stack), axis=-1) # [72,96,4*4]
    obs = np.squeeze(obs)
    return obs


# main function for agent
def agent(obs):
    obs = transform_obs(raw_obs)
    action = model.predict(obs, deterministic=True)
    return [action] 

