# self.sim.data.qpos are the positions, with the first 7 element the 
# 3D position (x,y,z) and orientation (quaternion x,y,z,w) of the torso, 
# and the remaining 8 positions are the joint angles.

# The [2:], operation removes the first 2 elements from the position, 
# which is the X and Y position of the agent's torso.

# self.sim.data.qvel are the velocities, with the first 6 elements 
# the 3D velocity (x,y,z) and 3D angular velocity (x,y,z) and the 
# remaining 8 are the joint velocities.

# 0 - x position
# 1 - y position
# 2 - z position
# 3 - x torso orientation
# 4 - y torso orientation
# 5 - z torso orientation
# 6 - w torso orientation
# 7-14 - joint angles

# 15-21 - 3d velocity/angular velocity
# 23-29 - joint velocities

import gym
import time
import numpy as np
from autoencoder.custom import CustomAutoencoder

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

features = [2,7,8,9,10]
height_bins = 20

min_bin = -1
max_bin = 1
num_bins = 15

start = 0
stop = 2

special = [0,1]
min_x, min_y = -12, -12
max_x, max_y = 12, 12
x_bins, y_bins = 16, 16

min_bin_2d = -20
max_bin_2d = 20
num_bins_2d = 20

n_bins_autoencoder = 10

reduce_dim = args.reduce_dim
expected_state_dim = len(special) + reduce_dim
G = np.transpose(np.random.normal(0, 1, (state_dim - len(special), reduce_dim)))

total_state_space = x_bins*y_bins* (num_bins**reduce_dim)

print("full_obs_dim = %d" % full_obs_dim)
print("total_state_space = %d" % total_state_space)
print("expected_state_dim = %d" % expected_state_dim)
print("action_dim = %d" % action_dim)

autoencoders = []
norm_factors = []

# TODO: is any of this being set globally?????
def learn_encoding(train, test):
    print("Custom....")
    if not args.reuse_net:
        autoencoder = CustomAutoencoder(num_input=29, 
                                num_hid1=24, num_hid2=16,
                                reduce_dim=args.autoencoder_reduce_dim, 
                                printfn=utils.log_statement)
    autoencoder.set_data(train)
    autoencoder.set_test_data(test)
    autoencoder.train()
    
    # set normalization factors
    norm_factors.append(autoencoder.test(iterations=5000))
    
    autoencoders.append(autoencoder)

def convert_obs(observation):
    new_obs = []
    for i in special:
        new_obs.append(observation[i])
    new_obs = np.concatenate((new_obs, np.dot(G, observation[2:])))
    return new_obs 

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = [
        # height
        discretize_range(0.2, 1.0, height_bins),
        # other fields
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins)
    ]
    return state_bins

def get_state_bins_reduced():
    state_bins = []
    state_bins.append(discretize_range(min_x, max_x, x_bins))
    state_bins.append(discretize_range(min_y, max_y, y_bins))
    
    for i in range(reduce_dim):
        state_bins.append(discretize_range(min_bin, max_bin, num_bins))
    return state_bins

def get_state_bins_autoencoder():
    state_bins = []
    for i in range(args.autoencoder_reduce_dim):
        state_bins.append(discretize_range(-1, 1, n_bins_autoencoder)) 
    return state_bins

def get_state_bins_2d_state():
    state_bins = []
    for i in range(start, stop):
        state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins_2d))
        
    if args.autoencode:
        state_bins = []
        for i in range(start, stop):
            state_bins.append(discretize_range(-1, 1, num_bins_2d))
    
    return state_bins

def get_num_states(state_bins):
    num_states = []
    for i in range(len(state_bins)):
        num_states.append(len(state_bins[i]) + 1)
    return num_states

state_bins = []
if args.gaussian:
    state_bins = get_state_bins_reduced()
elif args.autoencode:
    state_bins = get_state_bins_autoencoder()
else:
    state_bins = get_state_bins()
num_states = get_num_states(state_bins)

state_bins_2d = get_state_bins_2d_state()
num_states_2d = tuple([num_bins_2d for i in range(start, stop)])

# Discretize the observation features and reduce them to a single list.
def discretize_state_2d(obs, norm=[], env=None):
    
    # DO THIS if you want to examine the distribution over the 
    # autoencoded dimensions. Otherwise, it'll examine xy still
    if args.autoencode2d and env is not None:
        obs = env.env._get_obs()[:29]
        obs = autoencoders[-1].encode(obs).flatten()
        obs = np.divide(obs, norm_factors[-1])
    
    state = []
    for i in range(start, stop):
        feature = obs[i]
        state.append(discretize_value(feature, state_bins_2d[i - start]))
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
def discretize_state_autoencoder(env):
    
    obs = env.env._get_obs()[:29]
    obs = autoencoders[-1].encode(obs).flatten()
    obs = np.divide(obs, norm_factors[-1])

    # log encoded data to file.
    encodedfile = 'logs/encoded/' + args.exp_name + '.txt'
    with open(encodedfile, 'a') as f:
        f.write(str(obs) + '\n')
        
    # todo: discretize from here....
    state = []
    for i, feature in enumerate(obs):
        state.append(discretize_value(feature, state_bins[i]))
#     print(obs)
#     print(state)
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state(observation, norm=[], env=None):
    if args.gaussian:
        state = discretize_state_reduced(observation, norm)
    elif args.autoencode:
        state = discretize_state_autoencoder(env)
    else:
        state = discretize_state_normal(observation)

    return state

def get_height_dimension(arr):
    return np.array([np.sum(arr[i]) for i in range(arr.shape[0])])

def get_ith_dimension(arr, i):
    return np.array([np.sum(arr[j]) for j in range(arr.shape[i])])


