

import os, sys
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# environment
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# import customized environment wrapper
from env_wrapper import GFootballEnv 

# policy/value networks
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
# import customized policy/value network


# environment parameters
env_parser = argparse.ArgumentParser(description='enviorment parameters')
env_parser.add_argument('--level', default='academy_empty_goal_close',
        help='Defines type of problem being solved')
env_parser.add_argument('--state', default='extracted_stacked', 
        help='Observation to be used for training.')
env_parser.add_argument('--reward_experiment', default='scoring',
        help='Reward to be used for training.')

# policy parameters
policy_parser = argparse.ArgumentParser(description='policy parameters')
policy_parser.add_argument('--policy', default='ActorCriticPolicy',
        help='Policy architecture')

# training parameters
opt_parser = argparse.ArgumentParser(description='training parameters')
opt_parser.add_argument('--num_timesteps', default=3e6, type=int,
        help='Number of timesteps to run for.')
opt_parser.add_argument('--num_envs', default=4, type=int,
        help='Number of environments to run in parallel.')
opt_parser.add_argument('--seed', default=0, type=int,
        help='Random seed.')
opt_parser.add_argument('--lr', default=0.00008, type=float,
        help='Learning rate')
opt_parser.add_argument('--ent_coef', default=0.01, type=float,
        help='Entropy coeficient')
opt_parser.add_argument('--vf_coef', default=0.5, type=float,
        help='Value loss coeficient')
opt_parser.add_argument('--gamma', default=0.993,  type=float,
        help='Discount factor')
opt_parser.add_argument('--clip_range', default=0.27,  type=float,
        help='Clip range')
opt_parser.add_argument('--max_grad_norm', default=0.5,  type=float,
        help='Max gradient norm (clipping)')
opt_parser.add_argument('--n_steps', default=128, type=int, 
        help='Number of environment steps per epoch; ''batch size is nsteps * nenv')
opt_parser.add_argument('--n_epochs', default=10,  type=int,
        help='Number of updates per epoch.')
opt_parser.add_argument('--save_interval', default=100, type=int,
        help='How frequently checkpoints are saved.')

opt_parser.add_argument('--load_path', default=None,
        help='Path to load initial checkpoint from.')
opt_parser.add_argument('--save_path', default='./outputs', 
        help='Path to save checkpoints.')


def make_env(args, save_path, rank=0):
    def _init():
        env = GFootballEnv(args)
        log_file = os.path.join(save_path, "env_"+str(rank)+".log")
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init


def train():
    """Trains a PPO2 policy."""

    env_args = env_parser.parse_known_args()[0]
    policy_args = policy_parser.parse_known_args()[0]
    opt_args = opt_parser.parse_known_args()[0]

    os.makedirs(opt_args.save_path, exist_ok=True)

    # create environment
    # train_env = GFootballEnv(env_args) # for evaluation
    train_env = DummyVecEnv([make_env(env_args, opt_args.save_path, rank=i) for i in range(opt_args.num_envs)])
    eval_env = GFootballEnv(env_args) # for evaluation
    check_env(env=eval_env, warn=True)

    # define rl policy/value network
    policy = getattr(sys.modules[__name__], policy_args.policy)
    
    # initialize ppo
    tb_dir = os.path.join(opt_args.save_path, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    verbose = 1
    ppo = PPO(policy, train_env, learning_rate=opt_args.lr, n_steps=opt_args.n_steps, n_epochs=opt_args.n_epochs, 
            gamma=opt_args.gamma, gae_lambda=0.95, clip_range=opt_args.clip_range, clip_range_vf=None, 
            ent_coef=opt_args.ent_coef, vf_coef=opt_args.vf_coef, max_grad_norm=opt_args.max_grad_norm, 
            tensorboard_log=tb_dir, verbose=verbose, seed=opt_args.seed)

    # load initial checkpoint
    if opt_args.load_path:
        ppo.load(os.path.join(opt_args.load_path, "ppo_gfootball.pt"))

    # start training ppo
    eval_dir = os.path.join(opt_args.save_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ppo.learn(opt_args.num_timesteps, log_interval=1, 
            eval_env=eval_env, eval_freq=opt_args.save_interval, n_eval_episodes=10, eval_log_path=eval_dir)

    # save final checkpoint
    ppo.save(os.path.join(opt_args.save_path, "ppo_gfootball"))


if __name__ == '__main__':
    train()

