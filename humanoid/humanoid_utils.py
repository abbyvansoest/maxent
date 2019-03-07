
import gym
import time
import numpy as np

import utils
args = utils.get_args()

env = gym.make(args.env)

qpos = env.env.init_qpos
qvel = env.env.init_qvel

# state_dim = int(env.env.state_vector().shape[0])
# action_dim = int(env.action_space.sample().shape[0])
# print("state_dim = %d" % state_dim)
# print("action_dim = %d" % action_dim)

start = 0
stop = 2

min_bin_2d = -20
max_bin_2d = 20
num_bins_2d = 20

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins_2d_state():
    state_bins = []
    for i in range(start, stop):
        state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins_2d))
    
    return state_bins


state_bins_2d = get_state_bins_2d_state()
num_states_2d = tuple([num_bins_2d for i in range(start, stop)])

# Discretize the observation features and reduce them to a single list.
def discretize_state_2d(obs, norm=[], env=None):
    state = []
    for i in range(start, stop):
        feature = obs[i]
        state.append(discretize_value(feature, state_bins_2d[i - start]))
    return state


