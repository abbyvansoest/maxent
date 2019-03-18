import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.05, metavar='eps',
                    help='exploration rate')
parser.add_argument('--train_steps', type=int, default=500, metavar='ts',
                    help='number of steps per episodes')
parser.add_argument('--episodes', type=int, default=100, metavar='ep',
                    help='number of episodes per agent')
parser.add_argument('--epochs', type=int, default=50, metavar='epo',
                    help='number of models to train on entropy rewards')
parser.add_argument('--T', type=int, default=1000, metavar='T',
                    help='number of steps to roll out entropy policy')
parser.add_argument('--env', type=str, default='fake', metavar='env',
                    help='the env to learn')
parser.add_argument('--models_dir', type=str, default='/home/abby/entropy/data', metavar='N',
                    help='directory from which to load model policies')

parser.add_argument('--grad_ent', action='store_true',
                    help='use original gradient of entropy rewards')

parser.add_argument('--hid', type=int, default=300)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--exp_name', type=str, default='ant_sac')

parser.add_argument('--save_models', action='store_true',
                    help='collect a video of the final policy')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--online', action='store_true',
                    help='use online reward fn')

parser.add_argument('--gaussian', action='store_true',
                    help='reduce dimension with random gaussian')
parser.add_argument('--reduce_dim', type=int, default=5, metavar='rd',
                    help='dimension reduction parameter')

args = parser.parse_args()

def get_args():
    return copy.deepcopy(args)

# Env variables for MountainCarContinuous
nx = 12
nv = 11
mc_obs_dim = 2
mc_action_dim = 3
 
# env variables for Pendulum
pend_na = 8
pend_nv = 8
pendulum_obs_dim = 3
pendulum_action_dim = 3

# Env variables for HalfCheetah
cheetah_num_bins = 10
cheetah_obs_dim = 5 #len(env.observation_space.high)
cheetah_action_dim = 6
cheetah_space_dim = cheetah_num_bins**cheetah_obs_dim


def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = []
    if args.env == "HalfCheetah-v2":
        for i in range(cheetah_obs_dim):
            state_bins.append(discretize_range(-2, 2, cheetah_num_bins))
    elif args.env == "MountainCarContinuous-v0":
        state_bins = [
            # Cart position.
            discretize_range(-1.2, 0.6, nx), 
            # Cart velocity.
            discretize_range(-0.07, 0.07, nv)
        ]
    elif args.env == "Pendulum-v0":
        state_bins = [ # TODOTODO
            # Angles -- from 0 to 2pi?
            discretize_range(0, np.pi, pend_na), 
            discretize_range(0, np.pi, pend_na), 
            # Velocity
            discretize_range(-8, 8, pend_nv) # todo: need to limit velocity/acceleration?
        ]
    return state_bins

def get_obs_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_obs_dim
    elif args.env == "MountainCarContinuous-v0":
        return mc_obs_dim
    elif args.env == "Pendulum-v0":
        return pendulum_obs_dim

def get_action_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_action_dim
    elif args.env == "MountainCarContinuous-v0":
        return mc_action_dim
    elif args.env == "Pendulum-v0":
        return pendulum_action_dim

def get_space_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_space_dim
    elif args.env == "MountainCarContinuous-v0":
        return (nx, nv)
    elif args.env == "Pendulum-v0":
        return (pend_na, pend_nv)

def get_num_states(obs_dim, state_bins):
    num_states = []
    for i in range(obs_dim):
        num_states.append(len(state_bins[i]) + 1)

    if args.env == "HalfCheetah-v2":
        return num_states
    elif args.env == "MountainCarContinuous-v0":
        return num_states
    elif args.env == "Pendulum-v0":
        return (pend_na, pend_nv)

if args.env != 'Ant-v2' and args.env != 'Humanoid-v2':
    action_dim = get_action_dim()
    obs_dim = get_obs_dim()
    state_bins = get_state_bins()
    space_dim = get_space_dim()

    num_states = get_num_states(obs_dim, state_bins)

    print(state_bins)

total_state_space = nv*nx

# to discretize observation from pendulum, first figure out what theta is
# from the observation
def pendulum_discretize_state(observation):
    theta = np.arccos(observation[0])
    vel = observation[2]
  
    state = [discretize_value(theta, state_bins[0]), discretize_value(vel, state_bins[1])]
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state(observation):

    if (args.env == "Pendulum-v0"):
        return pendulum_discretize_state(observation)

    state = []
    for i, feature in enumerate(observation):
        if i >= obs_dim:
            break
        state.append(discretize_value(feature, state_bins[i]))
    return state

