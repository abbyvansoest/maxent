
"""

Run experiments for training VPG with normal and MaxEnt exploration. 

Example commands:

[train VPG on Ant mujoco env. use esp greedy exploration during training. run for 4 random seeds]
python run.py --env Ant-v2 --algo vpg --expert_loc ant/data/expert_fully_corrective --explore_during_training  --exp_name epsgreedy_eps100 --eps 0.100000 --pretrain_epochs 0 --exp_id epsgreedy_test --seed -1 -1 -1 -1

[train VPG on Ant mujoco env. pretrain env using MaxEnt (this will also produce a baseline randomly pre-trained run). run for 4 random seeds]
python run.py --env Ant-v2 --algo vpg --expert_loc ant/data/expert_fully_corrective --explore_pre_training  --exp_name pretrain_epoch20 --eps 0 --pretrain_epochs 20 --exp_id pretrain_test --seed -1 -1 -1 -1

"""


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
import algos.sac.sac as sac

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='vpg') # valid options: vpg, ppo, ddpg
parser.add_argument('--env', type=str, default='Ant-v2')
parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, nargs='+', default=-1)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--exp_id', type=str, default='')
parser.add_argument('--expert_loc', type=str, default=None)

# Args to customize exploration.
parser.add_argument('--explore_pre_training', action='store_true')
parser.add_argument('--explore_during_training', action='store_true')
parser.add_argument('--pretrain_epochs', type=int, default=0)
parser.add_argument('--eps', type=float, default=0)

args = parser.parse_args()

for s in args.seed:
    if s < 0:
        seed = random.randint(0, 100000)
    else:
        seed = s

    # Set up saving to appropriate directory.
    direct = os.getcwd() + '/data/' + args.algo + '/' +args.env + '/' + args.exp_id
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed, data_dir=direct)

    # Load explorer agent files
    explorer = None
    if args.explore_pre_training or args.explore_during_training:
        if args.expert_loc is None:
            raise ValueError("Expert location not set")
        expert_loc = os.getenv("HOME") + '/maxent/' + args.expert_loc + '/'
        explorer = Explorer(expert_loc, lambda : gym.make(args.env))

    train_fn = None
    if args.algo == 'vpg': 
        train_fn = vpg.vpg
    elif args.algo == 'ppo':
        train_fn = ppo.ppo
    elif args.algo == 'ddpg':
        train_fn = ddpg.ddpg
    elif args.algo == 'sac':
        train_fn = sac.sac

    if train_fn is None:
        raise ValueError('Failed to select valid train_fn')

    pretrain_epochs = args.pretrain_epochs if args.explore_pre_training else 0
    eps = args.eps if args.explore_during_training else 0

    train_fn(lambda : gym.make(args.env), 
             ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
             gamma=args.gamma, seed=seed, steps_per_epoch=args.steps, 
             epochs=args.epochs, logger_kwargs=logger_kwargs,
             explorer=explorer, eps=eps, 
             pretrain_epochs=pretrain_epochs)

    # run again with a baseline
    if args.explore_pre_training and (args.algo == 'vpg' or args.algo == 'ddpg'):
        print('Running baseline for pre-training')
        logger_kwargs = setup_logger_kwargs(args.exp_name + '_base', seed, data_dir=direct)
        train_fn(lambda : gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                 gamma=args.gamma, seed=seed, steps_per_epoch=args.steps, epochs=args.epochs,
                 logger_kwargs=logger_kwargs,
                 explorer=None, eps=eps, pretrain_epochs=pretrain_epochs)
