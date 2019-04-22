# State-Space (name/joint/parameter):
# 0 : rootx     slider      position (m)
# 1 : rootz     slider      position (m)
# 2 : rooty     hinge       angle (rad)
# 3 : bthigh    hinge       angle (rad)
# 4 : bshin     hinge       angle (rad)
# 5 : bfoot     hinge       angle (rad)
# 6 : fthigh    hinge       angle (rad)
# 7 : fshin     hinge       angle (rad)
# 8 : ffoot     hinge       angle (rad)
# 9 : rootx     slider      velocity (m/s)
# 10 : rootz     slider      velocity (m/s)
# 11 : rooty     hinge       angular velocity (rad/s)
# 12 : bthigh    hinge       angular velocity (rad/s)
# 13 : bshin     hinge       angular velocity (rad/s)
# 14 : bfoot     hinge       angular velocity (rad/s)
# 15 : fthigh    hinge       angular velocity (rad/s)
# 16 : fshin     hinge       angular velocity (rad/s)
# 17 : ffoot     hinge       angular velocity (rad/s)


import gym
import time
import numpy as np

import utils
args = utils.get_args()

env = gym.make(args.env)

dim_dict = {
    0:"x",
    1:"y",
    2:"z",      # have as special coordinates that you do not project. bin at appropriate value for these coordinates
}

qpos = env.env.init_qpos
qvel = env.env.init_qvel

full_obs_dim = int(len(env.reset()))
state_dim = int(env.env.state_vector().shape[0])
action_dim = int(env.action_space.sample().shape[0])

features = [0,1,3,4,5,6]
height_bins = 20

min_bin = -10
max_bin = 10
num_bins = 15

special = [0, 9]
mins = [-20, -8]
maxs = [20, 8]
bins = [20, 15]

plot_2d = [0,9]
min_bin_2d_0, min_bin_2d_1 = -20, -8
max_bin_2d_0, max_bin_2d_1  = 20, 8
num_bins_2d_0, num_bins_2d_1 = 20, 10

reduce_dim = args.reduce_dim
expected_state_dim = len(special) + reduce_dim
G = np.transpose(np.random.normal(0, 1, (state_dim - len(special), reduce_dim)))

total_state_space = np.prod(np.array(bins)) * (num_bins**reduce_dim)

print("full_obs_dim = %d" % full_obs_dim)
print("total_state_space = %d" % total_state_space)
print("expected_state_dim = %d" % expected_state_dim)
print("action_dim = %d" % action_dim)

norm_factors = []

def convert_obs(observation):
    new_obs = []
    for i in special:
        new_obs.append(observation[i])
    observation = [i for j, i in enumerate(observation) if j not in special]
    new_obs = np.concatenate((new_obs, np.dot(G, observation)))
    return new_obs 

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    
    state_bins = [
          # position
          discretize_range(-10, 10, 10),
          discretize_range(-10, 10, 10),
#           discretize_range(0, 2*np.pi, 4),
          discretize_range(0, 2*np.pi, 4),
          discretize_range(0, 2*np.pi, 4),
#           # velocity
          discretize_range(-5, 5, 5),
          discretize_range(-5, 5, 5)]
#           discretize_range(-5, 5, 5)]
#           discretize_range(-5, 5, 3),
#           discretize_range(-5, 5, 3)]
    
    return state_bins

def get_state_bins_reduced():
    state_bins = []
    
    for idx, i in enumerate(special):
        state_bins.append(discretize_range(mins[idx], maxs[idx], bins[idx]))
    
    for i in range(reduce_dim):
        state_bins.append(discretize_range(min_bin, max_bin, num_bins))
    return state_bins

def get_state_bins_2d_state():
    state_bins = []
    state_bins.append(discretize_range(min_bin_2d_0, max_bin_2d_0, num_bins_2d_0))
    state_bins.append(discretize_range(min_bin_2d_1, max_bin_2d_1, num_bins_2d_1))
    return state_bins

def get_num_states(state_bins):
    num_states = []
    for i in range(len(state_bins)):
        num_states.append(len(state_bins[i]) + 1)
    return num_states

state_bins = []
if args.gaussian:
    state_bins = get_state_bins_reduced()
else:
    state_bins = get_state_bins()
num_states = get_num_states(state_bins)

state_bins_2d = get_state_bins_2d_state()
num_states_2d = tuple([num_bins_2d_0, num_bins_2d_1])

# Discretize the observation features and reduce them to a single list.
def discretize_state_2d(obs, norm=[], env=None):
    
    state = []
    for idx, i in enumerate(plot_2d):
        feature = obs[i]
        state.append(discretize_value(feature, state_bins_2d[idx]))
    return state

def discretize_state_normal(observation):
    state = []
    for i, idx in enumerate(features):
        state.append(discretize_value(observation[idx], state_bins[i]))
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state_reduced(observation, norm=[]):
    
    if (len(observation) != expected_state_dim):
        observation = convert_obs(observation)

    if len(norm) > 0:
        for i in range(len(observation)):
            observation[i] = observation[i] / norm[i]

    state = []
    for i, feature in enumerate(observation):
        state.append(discretize_value(feature, state_bins[i]))
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state(observation, norm=[], env=None):
    if args.gaussian:
        state = discretize_state_reduced(observation, norm)
    else:
        state = discretize_state_normal(observation)

    return state

def get_height_dimension(arr):
    return np.array([np.sum(arr[i]) for i in range(arr.shape[0])])

def get_ith_dimension(arr, i):
    return np.array([np.sum(arr[j]) for j in range(arr.shape[i])])

def get_state(env, obs, wrapped=False):
    if wrapped:
        state = env.unwrapped.state_vector()
    else:
        state = env.env.state_vector()
    
    if not np.array_equal(np.clip(obs[:len(state) - 1], -10, 10), np.clip(state[1:], -10, 10)):
        utils.log_statement(obs)
        utils.log_statement(state)
        raise ValueError("state and observation are not equal")

#     print(state[special])
    return state
