

import os, sys
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# environment
# import customized environment wrapper
from env_wrapper import GFootballEnv 

# policy/value networks
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
# import customized policy/value network


parser = argparse.ArgumentParser(description='test parameters')
# environment parameters
parser.add_argument('--level', default='academy_empty_goal_close',
        help='Defines type of problem being solved')
parser.add_argument('--state', default='extracted_stacked', 
        help='Observation to be used for training.')
parser.add_argument('--reward_experiment', default='scoring',
        help='Reward to be used for training.')

# policy parameters
parser.add_argument('--policy', default='ActorCriticPolicy',
        help='Policy architecture')
parser.add_argument('--load_path', default=None,
        help='Path to load initial checkpoint from.')

# eval parameters
parser.add_argument('--n_epochs', default=10,
        help='Num episodes for test.')
parser.add_argument('--render', default=False,
        help='Shows the simulations.')

def make_env(args, save_path, rank=0):
    def _init():
        env = GFootballEnv(args)
        log_file = os.path.join(save_path, "env_"+str(rank)+".log")
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init


def test():
    """Trains a PPO2 policy."""

    args = parser.parse_args()

    # create environment
    # train_env = GFootballEnv(env_args) # for evaluation
    test_env = GFootballEnv(args) # for evaluation
    check_env(env=test_env, warn=True)

    # define rl policy/value network
    policy = getattr(sys.modules[__name__], policy_args.policy)
    
    # initialize ppo
    model = PPO(policy, eval_env)

    # load initial checkpoint
    if opt_args.load_path:
        ppo.load(os.path.join(opt_args.load_path, "ppo_gfootball.pt"))

    # test
    episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.n_epochs,
            render=args.render,
            deterministic=True,
            return_episode_rewards=True
    )

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
    print(f"Eval epochs={args.n_Eval_episodes}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

    print("\nDone")

if __name__ == '__main__':
    test()

