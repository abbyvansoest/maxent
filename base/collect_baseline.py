# Collect entropy-based reward policies.

# Changed from using all-1 reward to init to one-hot at: 2018_11_30-10-00

# python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --episodes=300 --epochs=50 --exp_name=test

import os 

import time
from datetime import datetime
import logging

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm

import gym

from cart_entropy_policy import CartEntropyPolicy
import base_utils
import curiosity
import plotting

import torch
from torch.distributions import Normal
import random

from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def moving_averages(values, size):
    for selection in window(values, size):
        yield sum(selection) / size

args = base_utils.get_args()
Policy = CartEntropyPolicy

def grad_ent(pt):
    
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p

    eps = 1/np.sqrt(base_utils.total_state_space)
    return 1/(pt + eps)

def online_rewards(average_p, average_ps, t):
    eps = 1/np.sqrt(base_utils.total_state_space)
    reward_fn = np.zeros(shape=average_p.shape)
    for ap in average_ps:
        reward_fn += 1/(ap + eps)
    reward_fn += np.sqrt(t)*average_p
    return reward_fn

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]

# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR):

    video_dir = 'videos/' + args.exp_name

    reward_fn = np.zeros(shape=(tuple(base_utils.num_states)))
    online_reward_fn = np.zeros(shape=(tuple(base_utils.num_states)))

    # set initial state to base, motionless state.
    seed = []
    if args.env == "Pendulum-v0":
        env.env.state = [np.pi, 0]
        seed = env.env._get_obs()
    elif args.env == "MountainCarContinuous-v0":
        env.env.state = [-0.50, 0]
        seed = env.env.state

    running_avg_p = np.zeros(shape=(tuple(base_utils.num_states)))
    running_avg_ent = 0
    running_avg_entropies = []
    running_avg_ps = []

    running_avg_p_online = np.zeros(shape=(tuple(base_utils.num_states)))
    running_avg_ent_online = 0
    running_avg_entropies_online = []
    running_avg_ps_online = []

    running_avg_p_baseline = np.zeros(shape=(tuple(base_utils.num_states)))
    running_avg_ent_baseline = 0
    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []

    online_average_ps = []
    
    policies = []
    initial_state = init_state(args.env)

    online_policies = []
    online_initial_state = init_state(args.env)

    for i in range(epochs):

        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim)
        online_policy = Policy(env, args.gamma, args.lr, base_utils.obs_dim, base_utils.action_dim) 

        if i == 0:
            policy.learn_policy(reward_fn, 
                episodes=0, 
                train_steps=0)
            online_policy.learn_policy(online_reward_fn, 
                episodes=0, 
                train_steps=0)
        else:
            policy.learn_policy(reward_fn, 
                initial_state=initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)
            online_policy.learn_policy(online_reward_fn, 
                initial_state=online_initial_state, 
                episodes=args.episodes, 
                train_steps=args.train_steps)

        policies.append(policy)
        online_policies.append(online_policy)

        epoch = 'epoch_%02d/' % (i) 
        
        a = 10 # average over this many rounds
        p_baseline = policy.execute_random(T,
            render=args.render, video_dir=video_dir+'/baseline/'+epoch)
       
        round_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())
        for av in range(a - 1):
            next_p_baseline = policy.execute_random(T)
            p_baseline += next_p_baseline
            round_entropy_baseline += scipy.stats.entropy(next_p_baseline.flatten())
        p_baseline /= float(a)
        round_entropy_baseline /= float(a) # running average of the entropy

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state = \
            curiosity.execute_average_policy(env, policies, T, 
                initial_state=initial_state, 
                avg_runs=a, 
                render=False)
        online_average_p, online_round_avg_ent, online_initial_state = \
            curiosity.execute_average_policy(env, online_policies, T, 
                initial_state=online_initial_state, 
                avg_runs=a, 
                render=False)

        # Get next distribution p by executing pi for T steps.
        # ALSO: Collect video of each policy
        p = policy.execute(T, initial_state=initial_state, 
            render=args.render, video_dir=video_dir+'/normal/'+epoch)
        p_online = online_policy.execute(T, initial_state=initial_state, 
            render=args.render, video_dir=video_dir+'/online/'+epoch)
        
        # Force first round to be equal
        if i == 0:
            average_p = p_baseline
            round_avg_ent = round_entropy_baseline
            online_average_p = p_baseline
            online_round_avg_ent = round_entropy_baseline

        # If in pendulum, set velocity to 0 with some probability
        if args.env == "Pendulum-v0" and random.random() < 0.3:
            initial_state[1] = 0

        # goal: try online reward structure
        online_reward_fn = online_rewards(online_average_p, online_average_ps, epochs)
        online_average_ps.append(online_average_p)

        reward_fn = grad_ent(average_p)

        # Update experimental running averages.
        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)  

        # Update online running averages.
        running_avg_ent_online = running_avg_ent_online * (i)/float(i+1) + online_round_avg_ent/float(i+1)
        running_avg_p_online = running_avg_p_online * (i)/float(i+1) + online_average_p/float(i+1)
        running_avg_entropies_online.append(running_avg_ent_online)
        running_avg_ps_online.append(running_avg_p_online)     

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline = running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline) 

        print("--------------------------------")
        print("p=")
        print(p)

        print("average_p =") 
        print(average_p)

        print("online_average_p")
        print(online_average_p)

        print("---------------------")

        print("round_avg_ent[%d] = %f" % (i, round_avg_ent))
        print("running_avg_ent = %s" % running_avg_ent)

        print("..........")

        print("online_round_avg_ent[%d] = %f" % (i, online_round_avg_ent))
        print("running_avg_ent_online = %s" % running_avg_ent_online)

        print("..........")

        print("round_entropy_baseline[%d] = %f" % (i, round_entropy_baseline))
        print("running_avg_ent_baseline = %s" % running_avg_ent_baseline)

        print("--------------------------------")

        plotting.heatmap(running_avg_p, average_p, i, args.env)

    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    plotting.running_average_entropy3(running_avg_entropies, running_avg_entropies_baseline, running_avg_entropies_online)

    indexes = [1,2,5,10]
    plotting.heatmap4(running_avg_ps, running_avg_ps_baseline, indexes)
    plotting.heatmap3x4(running_avg_ps, running_avg_ps_online, running_avg_ps_baseline, indexes)

    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    # TODO: limit acceleration (maybe also speed?) for Pendulum.
    if args.env == "Pendulum-v0":
        env.env.max_speed = 8
        env.env.max_torque = 1
    env.seed(int(time.time())) # seed environment

    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    MODEL_DIR = 'models-' + args.env + '/models_' + TIME + '/'

    if args.save_models:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # save metadata from the run. 
        with open(MODEL_DIR + "metadata", "w") as metadata:
            metadata.write("args: %s\n" % args)
            metadata.write("num_states: %s\n" % str(base_utils.num_states))
            metadata.write("state_bins: %s\n" % base_utils.state_bins)

    plotting.FIG_DIR = 'figs/' + args.env + '/'
    plotting.model_time = args.exp_name + '/'
    if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
        os.makedirs(plotting.FIG_DIR+plotting.model_time)

    policies = collect_entropy_policies(env, args.epochs, args.T, MODEL_DIR)
    env.close()
    print("DONE")

if __name__ == "__main__":
    main()


