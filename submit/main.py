

# customized policy networks here: cnn, impala_cnn
from gfootball.examples import models  

# customized env here
import gfootball.env as football_env

# ppo dependencies
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables

# parameters
level = "11_vs_11_easy_stocahstic"
state_representation = "extracted_stacked"
reward_experiment = "scoring,checkpoints"
network = "impala_cnn"
load_path = "checkpoint"


# initialize an environment to create agent
env = football_env.create_environment(
    env_name=level, stacked=('stacked' in state_representation),
    rewards=reward_experiment,
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    )


# get state_space and action_space
ob_space = env.observation_space
ac_space = env.action_space


# define actor policy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy_fn = build_policy(env, network)
policy = policy_fn(nbatch=1, sess=sess)

# load checkpoints from file
load_variables(load_path, sess)


# main function for agent
def agent(obs):
    actions, values, _, _ = model.step(obs)
    return actions



