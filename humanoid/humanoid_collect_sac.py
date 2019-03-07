# Collect entropy-based reward policies.

# python humanoid_collect_sac.py --env="Humanoid-v2" --exp_name=test --T=1000 --n=20 --l=2 --hid=300 --epochs=16 --episodes=16 --gaussian --reduce_dim=5

import sys
sys.path.append('/home/abby')

import os
import time
import random

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from tabulate import tabulate

import gym
import tensorflow as tf

import utils
import humanoid_utils
import plotting
from humanoid_soft_actor_critic import HumanoidSoftActorCritic
from reward_fn import RewardFn

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

def get_state(env, obs):
    state = env.env.state_vector()
    if not np.array_equal(obs[:len(state) - 2], state[2:]):
        utils.log_statement(obs)
        utils.log_statement(state)
        raise ValueError("state and observation are not equal")
    return state

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], n=10, render=False, epoch=0):
    
    random_initial_state = []
    p_xy = np.zeros(shape=(tuple(humanoid_utils.num_states_2d)))
    
    denom = 0
    max_idx = len(policies) - 1

    data = []

    # average results over n rollouts
    for iteration in range(n):

        print('---- ' + str(iteration)+ ' ----')
        
        env.reset()
        obs = get_state(env, env.env._get_obs())
        random_T = np.floor(random.random()*T)
        random_initial_state = []
       
        for t in range(T):

            if t % 1000 == 0:
                print(t)
            
            # action = np.zeros(shape=(1,humanoid_utils.action_dim))
            idx = random.randint(0, max_idx)
            if idx == 0:
                action = env.action_space.sample()
            else:
                action = policies[idx].get_action(obs, deterministic=args.deterministic)
                
            # Count the cumulative number of new states visited as a function of t.
            obs, _, done, _ = env.step(action)
            obs = get_state(env, obs)
            data.append(obs)
            
            p_xy[tuple(humanoid_utils.discretize_state_2d(obs, env))] += 1
            denom += 1
            
            if t == random_T:
                random_initial_state = obs

            if render:
                env.render()
            if done: # CRITICAL: ignore done signal
                done = False
            
    env.close()
    p_xy /= float(denom)

    return data, p_xy, random_initial_state

def entropy(pt):
    utils.log_statement("pt size %d" % pt.size)
    entropy = 0.0
    for p in pt:
        if p == 0.0:
            continue
        entropy += p*np.log(p)
    return -entropy

# Main loop of maximum entropy program. WORKING HERE
# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR=''):
    
    direct = os.getcwd()+ '/data/'
    experiment_directory = direct + args.exp_name
    print(experiment_directory)
    
    indexes = [1,5,10,15]

    running_avg_p_xy = np.zeros(shape=(tuple(humanoid_utils.num_states_2d)))
    running_avg_ent_xy = 0

    running_avg_p_baseline_xy = np.zeros(shape=(tuple(humanoid_utils.num_states_2d)))
    running_avg_ent_baseline_xy = 0

    running_avg_entropies_xy = []
    running_avg_ps_xy = []
    avg_ps_xy = []

    running_avg_entropies_baseline_xy = []
    running_avg_ps_baseline_xy = []
    avg_ps_baseline_xy = []

    policies = []
    initial_state = []

    # initial reward function = all ones
    reward_fn = RewardFn(None)

    for i in range(epochs):
        utils.log_statement("*** ------- EPOCH %d ------- ***" % i)
        
        # clear initial state if applicable.
        if not args.initial_state:
            initial_state = []
        else:
            utils.log_statement(initial_state)

        logger_kwargs = setup_logger_kwargs("model" + str(i), data_dir=experiment_directory)

        # Learn policy that maximizes current reward function.
        print("Learning new oracle...")
        if args.seed != -1:
            seed = args.seed
        else:
            seed = random.randint(1, 100000)
        
        sac = HumanoidSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=seed, gamma=args.gamma, max_ep_len=1000,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            logger_kwargs=logger_kwargs)
        
        # The first policy is random
        if i == 0:
            sac.soft_actor_critic(epochs=0) 
        else:
            sac.soft_actor_critic(epochs=args.episodes, 
                                  initial_state=initial_state, 
                                  start_steps=args.start_steps) 
        policies.append(sac)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        print("Executing mixed policy...")
        data, average_p_xy, initial_state = execute_average_policy(env, policies, T,
                                       initial_state=initial_state, n=args.n, 
                                       render=False, epoch=i)
        test_data,_,_= execute_average_policy(env, policies, T=1000, n=3)

        np.save("data/test_data_"+args.exp_name, test_data)
        np.save("data/data_"+args.exp_name, data)
            
        print("Calculating maxEnt entropy...")
        round_entropy_xy = entropy(average_p_xy.ravel())
        
        # Update running averages for maxEnt.
        print("Updating maxEnt running averages...")
        running_avg_ent_xy = running_avg_ent_xy * (i)/float(i+1) + round_entropy_xy/float(i+1)
        running_avg_p_xy *= (i)/float(i+1)
        running_avg_p_xy += average_p_xy/float(i+1)
        
        # TODO: collect a lot of data from the current mixed policy
        # use this data to learn a new distribution in reward_fn object
        # then, the reward_fn object will use the new distribution to 
        # compute rewards
        # update reward function
        print("Update reward function")
        reward_fn = RewardFn(data, eps=.001)
        reward_fn.test(test_data, env)

        # (save for plotting)
        running_avg_entropies_xy.append(running_avg_ent_xy)
        if i in indexes:
            running_avg_ps_xy.append(np.copy(running_avg_p_xy))
            avg_ps_xy.append(np.copy(average_p_xy))

        print("Collecting baseline experience....")
        p_baseline_xy,_ = sac.test_agent_random(T, n=args.n)
        
        print("Compute baseline entropy....")
        round_entropy_baseline_xy = entropy(p_baseline_xy.ravel())

        # Update baseline running averages.
        print("Updating baseline running averages...")
        running_avg_ent_baseline_xy = running_avg_ent_baseline_xy * (i)/float(i+1) + round_entropy_baseline_xy/float(i+1)
        running_avg_p_baseline_xy *= (i)/float(i+1) 
        running_avg_p_baseline_xy += p_baseline_xy/float(i+1)
        
        # (save for plotting)
        running_avg_entropies_baseline_xy.append(running_avg_ent_baseline_xy)
        if i in indexes:
            running_avg_ps_baseline_xy.append(np.copy(running_avg_p_baseline_xy))
            avg_ps_baseline_xy.append(np.copy(p_baseline_xy))
    
        utils.log_statement(average_p_xy)
        utils.log_statement(p_baseline_xy)
        
        # Print round summary.
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy_xy", 
                "running_avg_ent_xy"]
        col2 = [round_entropy_baseline_xy, running_avg_ent_baseline_xy]
        col3 = [round_entropy_xy, running_avg_ent_xy]
        table = tabulate(np.transpose([col1, col2, col3]), 
            col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        utils.log_statement(table)
        
        # Plot from round.
        plotting.heatmap(running_avg_p_xy, average_p_xy, i)
        plotting.heatmap1(running_avg_p_baseline_xy, i)
        

    # cumulative plots.
    plotting.heatmap4(running_avg_ps_xy, running_avg_ps_baseline_xy, indexes, ext="cumulative")
    plotting.heatmap4(avg_ps_xy, avg_ps_baseline_xy, indexes, ext="epoch")
    plotting.running_average_entropy(running_avg_entropies_xy, running_avg_entropies_baseline_xy, ext='_xy')

    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100, linewidth=150, precision=8)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment

    plotting.FIG_DIR = 'figs/' + args.env + '/'
    plotting.model_time = args.exp_name + '/'
    if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
        os.makedirs(plotting.FIG_DIR+plotting.model_time)

    policies = collect_entropy_policies(env, args.epochs, args.T)
    env.close()

    print("*** ---------- ***")
    print("DONE")

if __name__ == "__main__":
    main()


