# python run.py --env='Ant-v2' --algo=vpg --epochs=500 --steps=10000 --seed=0 --expert_loc=ant/data/expert_fully_corrective --eps=0 --exp_name=trial0_base

import sys
import os
sys.path.append(os.getenv("HOME") + '/maxent')
sys.path.append(os.getenv("HOME") + '/spinningup')

import random
import gym
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_tools import mpi_fork

import argparse
from explorer import Explorer
import algos.ppo.ppo as ppo
import algos.vpg.vpg as vpg
import algos.ddpg.ddpg as ddpg

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='vpg') # valid options: vpg, ppo, ddpg
parser.add_argument('--env', type=str, default='Ant-v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--steps', type=int, default=4000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--expert_loc', type=str, default=None)
parser.add_argument('--eps', type=float, default=0.05)
args = parser.parse_args()

mpi_fork(args.cpu)  # run parallel code with mpi

# Set up saving to appropriate directory.
direct = os.getcwd()+'/data/'+args.algo
logger_kwargs = setup_logger_kwargs(args.exp_name, data_dir=direct)

# Load explorer agent files
explorer = None
if args.expert_loc is not None:
    args.expert_loc = os.getenv("HOME") + '/maxent/' + args.expert_loc + '/'

if args.seed < 0:
    seed = random.randint(0, 100000)
else:
    seed = args.seed

train_fn = None
if args.algo == 'vpg': 
    train_fn = vpg.vpg
elif args.algo == 'ppo':
    train_fn = ppo.ppo
elif args.algo == 'ddpg':
    train_fn = ddpg.ddpg
    
if train_fn is None:
    raise ValueError('Failed to select valid train_fn')

train_fn(lambda : gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
         gamma=args.gamma, seed=seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs, explorer=explorer, eps=args.eps)