import argparse
import copy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.05, metavar='eps',
                    help='exploration rate')
parser.add_argument('--episodes', type=int, default=100, metavar='ep',
                    help='number of episodes per agent')
parser.add_argument('--epochs', type=int, default=50, metavar='epo',
                    help='number of models to train on entropy rewards')
parser.add_argument('--T', type=int, default=1000, metavar='T',
                    help='number of steps to roll out entropy policy')
parser.add_argument('--n', type=int, default=10, metavar='n',
                    help='number of rollouts to average over')
parser.add_argument('--env', type=str, default='Reacher-v2', metavar='env',
                    help='the env to learn')

parser.add_argument('--models_dir', type=str, default='logs/file.out', metavar='N',
                    help='directory from which to load model policies')

parser.add_argument('--hid', type=int, default=300)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--seed', '-s', type=int, default=-1)
parser.add_argument('--exp_name', type=str, default='ant_sac')

parser.add_argument('--save_models', action='store_true',
                    help='collect a video of the final policy')
parser.add_argument('--render', action='store_true',
                    help='render the environment')

parser.add_argument('--deterministic', action='store_true',
                    help='act deterministically in mixed policy')
parser.add_argument('--cumulative', action='store_true',
                    help='use cumulative reward_fn')
parser.add_argument('--gaussian', action='store_true',
                    help='reduce dimension with random gaussian')
parser.add_argument('--reduce_dim', type=int, default=5, metavar='rd',
                    help='dimension reduction parameter')

parser.add_argument('--learn_reduced', action='store_true',
                    help='sac algo learns on reduced state')
parser.add_argument('--max_sigma', action='store_true',
                    help='use max sigma approach in policy averaging')
parser.add_argument('--grad_ent', action='store_true',
                    help='use original gradient of entropy rewards')
parser.add_argument('--avg_N', type=int, default=20, metavar='aN',
                    help='unique states visited average runs')
parser.add_argument('--start_steps', type=int, default=10000, metavar='ss',
                    help='start steps parameter')
parser.add_argument('--initial_state', action='store_true',
                    help='seed learning policies with initial state')

parser.add_argument('--wrap', action='store_true',
                    help='wrap theta value in [0, 2pi)')
parser.add_argument('--fingertip', action='store_true',
                    help='add fingertip xy location to state')

args = parser.parse_args()


def get_args():
    return copy.deepcopy(args)

