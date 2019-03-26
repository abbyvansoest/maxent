import sys
import os
sys.path.append(os.getenv("HOME") + '/maxent')
sys.path.append(os.getenv("HOME") + '/spinningup')

import gym
import spinup.algos.ppo.core as core
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import mpi_fork

import argparse
from explorer import Explorer
import algos.ppo.ppo as ppo

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='ppo')
parser.add_argument('--env', type=str, default='Ant-v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--expert_loc', type=str, default='ant/data/expert_fully_corrective')
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

# Set up saving to appropriate directory.
direct = os.getcwd()+'/data/'
logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=direct)

args.expert_loc = '/Users/abbyvansoest/maxent/'+args.expert_loc+'/'

# Load explorer agent files
explorer = Explorer(args.expert_loc, lambda : gym.make(args.env))

ppo.ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs, explorer=explorer, eps=0.05)