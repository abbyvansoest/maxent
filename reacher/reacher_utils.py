# TODO: reacher should explore the theta/velocity of each joint angle (produce charts for each, eventually)
# TODO: cap speed/torque of the model somehow 0 -- getting into really fast spin loops
    # create xml file with "limited" joints

import gym
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

joint0th = 0
joint0v = 2

joint1th = 1
joint1v = 3

def get_state(env, obs):
    svec = env.env.state_vector()
    deg = np.degrees(svec[:2]) % 360

    if np.radians(deg).any() > 2*np.pi or np.radians(deg).any() < 0:
        print(obs)
        print(svec)
        raise ValueError("invalid degree")

    if not (np.isclose(np.cos(svec[:2]), np.cos(np.radians(deg)))).all() or \
    not (np.isclose(np.sin(svec[:2]), np.sin(np.radians(deg)))).all():
        print(obs)
        print(svec)
        raise ValueError("state and observation are not equal")

    if not np.array_equal(svec[:2], env.env.sim.data.qpos.flat[:2]) or \
    not np.array_equal(svec[4:6], env.env.sim.data.qvel.flat[:2]):
        print(obs)
        print(svec)
        raise ValueError("state and observation are not equal")

    # state = [f1 f2 f3 theta1 theta2 vel1 vel2]
    state = np.concatenate([
        svec[0:2], 
        svec[4:6]])
    if args.wrap:
        state = np.concatenate([
            np.radians(deg), 
            svec[4:6]])
    
    if args.fingertip:
        state = np.concatenate([state, env.env.get_body_com("fingertip")[:2]])
#     print("----")
#     print(svec)
#     print(state)
    return state

o = env.reset()
state_dim = len(get_state(env, o))
env_state_dim = int(len(o))
action_dim = int(env.action_space.sample().shape[0])

min_bin = -1
max_bin = 1
num_bins = 10

start = 0
stop = 2

special = [0,1]

min_bin_2d = -.27
max_bin_2d = .27
num_bins_2d = 15

theta_bin = 30
vel_bin = 30
if args.wrap:
    theta_bin = 16
    vel_bin = 30

reduce_dim = args.reduce_dim
expected_state_dim = len(special) + reduce_dim
G = np.transpose(np.random.normal(0, 1, (state_dim - len(special), reduce_dim)))

total_state_space = num_bins_2d*num_bins_2d* (num_bins**reduce_dim)

print("total_state_space = %d" % total_state_space)
print("gaussian_state_dim = %d" % expected_state_dim)
print("action_dim = %d" % action_dim)

def convert_obs(observation):
    if not args.gaussian:
        return observation
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
    state_bins = []

    # position - angular
    if args.wrap:
        state_bins.append(discretize_range(0, 2*np.pi, 6))
        state_bins.append(discretize_range(0, 2*np.pi, 6))
    else:
        state_bins.append(discretize_range(-10, 10, 15))
        state_bins.append(discretize_range(-10, 10, 15))
        
    # velocity
    state_bins.append(discretize_range(-30, 30, 15))
    state_bins.append(discretize_range(-30, 30, 15))
    
    # Bins for fingertip
    if args.fingertip:
        state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins))
        state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins))

    return state_bins

def get_state_bins_reduced():
    state_bins = []
    state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins))
    state_bins.append(discretize_range(min_bin_2d, max_bin_2d, num_bins))
    
    for i in range(reduce_dim):
        state_bins.append(discretize_range(min_bin, max_bin, num_bins))
    return state_bins

def get_state_bins_2d_state():
    state_bins = []
    # theta
    if args.wrap:
        state_bins.append(discretize_range(0, 2*np.pi, theta_bin))
    else:
        state_bins.append(discretize_range(-20, 20, theta_bin))
    # velocity
    state_bins.append(discretize_range(-50, 50, vel_bin))
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
num_states_2d = tuple([theta_bin, vel_bin])

# Discretize the observation features and reduce them to a single list.
def discretize_state_2d_idx(observation, idx1, idx2, norm=[]):
    state = []
    state.append(discretize_value(observation[idx1], state_bins_2d[0]))
    state.append(discretize_value(observation[idx2], state_bins_2d[1]))
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state_2d(observation, idx1=3, idx2=5, norm=[]):
    return discretize_state_2d_idx(observation, idx1, idx2, norm)
#     state = []
#     for i in range(start, stop):
#         feature = observation[i]
#         state.append(discretize_value(feature, state_bins_2d[i - start]))
#     return state

def discretize_state_normal(observation):
    state = []
    for i, feature in enumerate(observation):
        state.append(discretize_value(feature, state_bins[i]))
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
def discretize_state(observation, norm=[]):
    if args.gaussian:
        state = discretize_state_reduced(observation, norm)
    else:
        state = discretize_state_normal(observation)
    return state

def get_height_dimension(arr):
    return np.array([np.sum(arr[i]) for i in range(arr.shape[0])])

def get_ith_dimension(arr, i):
    return np.array([np.sum(arr[j]) for j in range(arr.shape[i])])


