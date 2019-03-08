import numpy as np
import argparse
import copy
import sys
import os

parser = argparse.ArgumentParser(description='Ant Entropy')

# learning and Frank Wolfe args
parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.05, metavar='eps',
                    help='exploration rate')
parser.add_argument('--episodes', type=int, default=15, metavar='ep',
                    help='number of episodes per agent')
parser.add_argument('--epochs', type=int, default=15, metavar='epo',
                    help='number of models to train on entropy rewards')
parser.add_argument('--T', type=int, default=10000, metavar='T',
                    help='number of steps to roll out entropy policy')
parser.add_argument('--n', type=int, default=20, metavar='n',
                    help='number of rollouts to average over')
parser.add_argument('--env', type=str, default='Humanoid-v2', metavar='env',
                    help='the env to learn')


# policy architecture args
parser.add_argument('--hid', type=int, default=300)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--seed', '-s', type=int, default=-1)
parser.add_argument('--exp_name', type=str, default='test')

# saving args
parser.add_argument('--models_dir', type=str, default='logs/file.out', metavar='N',
                    help='directory from which to load model policies')
parser.add_argument('--save_models', action='store_true',
                    help='collect a video of the final policy')
parser.add_argument('--render', action='store_true',
                    help='render the environment')


# run config
parser.add_argument('--start_steps', type=int, default=10000, metavar='ss',
                    help='start steps parameter')

# experimental args
parser.add_argument('--deterministic', action='store_true',
                    help='act deterministically in mixed policy')
parser.add_argument('--cumulative', action='store_true',
                    help='use cumulative reward_fn')
parser.add_argument('--grad_ent', action='store_true',
                    help='use original gradient of entropy rewards')
parser.add_argument('--initial_state', action='store_true',
                    help='seed learning policies with initial state')

args = parser.parse_args()

def get_args():
    return copy.deepcopy(args)

if not os.path.exists('logs/encoded'):
    os.makedirs('logs/encoded')

logfile = 'logs/' + args.exp_name + '.txt'
def log_statement(s):
    print(s)
    with open(logfile, 'a') as f:
        f.write(str(s)+'\n')
        