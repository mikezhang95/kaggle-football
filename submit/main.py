
import tensorflow.compat.v1 as tf

# customized policy networks here: cnn, impala_cnn
from gfootball.examples import models  

# customized env here
import gfootball.env as football_env

# ppo dependencies
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session, save_variables, load_variables

# parameters
level = "11_vs_11_easy_stochastic"
state_representation = "extracted_stacked"
reward_experiment = "scoring,checkpoints"
network = "impala_cnn"
load_path = "/kaggle_simulations/agent/checkpoint"


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
print("state space, action space")
print(ob_space, ac_space)


# define actor policy
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy_fn = build_policy(env, network)
policy = policy_fn(nbatch=1, sess=sess)

prefix = "ppo2_model/"
# load checkpoints from file
import joblib, os
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

loaded_params = joblib.load(os.path.expanduser(load_path))

restores = []
if isinstance(loaded_params, list):
    assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
    for d, v in zip(loaded_params, variables):
        restores.append(v.assign(d))
else:
    for v in variables:
        v_name = prefix + v.name 
        restores.append(v.assign(loaded_params[v_name]))
sess.run(restores)
print("policy loaded")



# main function for agent
def agent(obs):
    actions, values, _, _ = policy.step(obs)
    return actions
