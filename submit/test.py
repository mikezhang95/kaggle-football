
import sys, os

sys.path.append('./submit/')
os.environ['CUDA_DEVCIE_ORDER'] = "PCR_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from stable_baselines3 import PPO
from football_env import GFootballEnv


# LEVEL = "academy_empty_goal_close"
# LEVEL = "academy_pass_and_shoot_with_keeper"
# LEVEL = "academy_counterattack_easy"
LEVEL = "academy_shoot_with_keeper"

load_dir = "./outputs/{}/ppo_gfootball".format(LEVEL)

class EnvArgs(object):
    level = LEVEL
    state = 'extracted_stacked'
    reward_experiment = 'scoring'

# environment
env_args = EnvArgs()
eval_env = GFootballEnv(env_args) # for evaluation
print("Environment created.")

# agent
model = PPO.load(load_dir,device="cpu")
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
    done, i= False, 0
    while not done:
        action, state = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(int(action))
        print(i, action, obs.shape, reward)
        i += 1
    print("Done")
