# Collect entropy-based reward policies.

# python swimmer_collect_sac.py --env="Swimmer-v2" --exp_name=test --T=1000 --n=20 --l=2 --hid=300 --epochs=16 --episodes=16

import sys
import os
sys.path.append(os.getenv("HOME")+'/maxent')
sys.path.append(os.getenv("HOME") + '/spinningup')

import os
import time
import random

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from tabulate import tabulate

import gym
from gym import wrappers
import tensorflow as tf

import utils
import swimmer_utils
import plotting
from swimmer_soft_actor_critic import SwimmerSoftActorCritic
from reward_fn import RewardFn

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

def select_action(policies, weights, env, obs):
    
    if len(weights) != len(policies):
        print("Weights array is wrong dimension -- using uniform weighting")
        weights = np.ones(len(policies))/float(len(policies))
    
    indexes = np.arange(len(policies))
    idx = np.random.choice(indexes, p=weights)
    
    if idx == 0:
        action = env.action_space.sample()
    else:
        action = policies[idx].get_action(obs, deterministic=args.deterministic)
    return action

def execute_one_rollout(policies, weights, env, obs, T, data, video_dir='', wrapped=False):

    state_data, p_xy, random_initial_state = data
    random_T = np.floor(random.random()*T)

    done = False
    uid = 1
    t = 0
    while (t < T) and not done:
        t = t + 1

        action = select_action(policies, weights, env, obs)
        
        # Count the cumulative number of new states visited as a function of t.
        obs, _, done, _ = env.step(action)
        obs = swimmer_utils.get_state(env, obs, wrapped)
        state_data.append(obs)

        p_xy[tuple(swimmer_utils.discretize_state_2d(obs, env))] += 1
        if t == random_T:
            random_initial_state = obs

        if done: # CRITICAL: ignore done signal
            done = False
            if wrapped:
                print(t)
                env.close()
                base_env = gym.make('Swimmer-v2')
                env = wrappers.Monitor(base_env, video_dir+'/%d' % uid)
                env.reset()
                uid = uid + 1
                qpos = obs[:len(swimmer_utils.qpos)]
                qvel = obs[len(swimmer_utils.qpos):]
                env.unwrapped.set_state(qpos, qvel)
                d = False

    p_xy /= float(t)
    data = (state_data, p_xy, random_initial_state)
    return data
                
# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, weights=[], initial_state=[], n=10, render=False, video_dir='', epoch=0):
       
    state_data = []
    random_initial_state = []
    p_xy = np.zeros(shape=(tuple(swimmer_utils.num_states_2d)))
    
    data = (state_data, p_xy, random_initial_state)

    # average results over n rollouts
    for iteration in range(n):
        print(iteration)
        env.reset()
        
        if len(initial_state) == 0:
            env.reset()
            initial_state = env.env.state_vector()
                 
        # only get a recording of first iteration
        if render and iteration == 0:
            print('recording mixed iteration....')
            wrapped_env = wrappers.Monitor(env, video_dir+'/%d' % 0)
            wrapped_env.reset()
            qpos = initial_state[:len(swimmer_utils.qpos)]
            qvel = initial_state[len(swimmer_utils.qpos):]
            wrapped_env.unwrapped.set_state(qpos, qvel)
            obs = swimmer_utils.get_state(wrapped_env, \
                                           wrapped_env.unwrapped._get_obs(), wrapped=True)
            data = execute_one_rollout(policies, weights, wrapped_env, obs, \
                                       T=args.record_steps, data=data, \
                                       video_dir=video_dir, wrapped=True)
        else:
            obs = swimmer_utils.get_state(env, env.env._get_obs())
            data = execute_one_rollout(policies, weights, env, obs, T, data)
            
    env.close()
    
    state_data, p_xy, random_initial_state = data
    return state_data, p_xy, random_initial_state

def entropy(pt):
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
    
    video_dir = 'videos/' + args.exp_name
    
    direct = os.getcwd()+ '/data/'
    experiment_directory = direct + args.exp_name
    print(experiment_directory)
    
    indexes = [1,5,10,15]

    running_avg_p_xy = np.zeros(shape=(tuple(swimmer_utils.num_states_2d)))
    running_avg_ent_xy = 0

    running_avg_p_baseline_xy = np.zeros(shape=(tuple(swimmer_utils.num_states_2d)))
    running_avg_ent_baseline_xy = 0

    running_avg_entropies_xy = []
    running_avg_ps_xy = []
    avg_ps_xy = []

    running_avg_entropies_baseline_xy = []
    running_avg_ps_baseline_xy = []
    avg_ps_baseline_xy = []

    policies = []
    distributions = []
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

        logger_kwargs = setup_logger_kwargs("model%02d" % i, data_dir=experiment_directory)

        # Learn policy that maximizes current reward function.
        print("Learning new oracle...")
        if args.seed != -1:
            seed = args.seed
        else:
            seed = random.randint(1, 100000)
        
        sac = SwimmerSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
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
        
        p_xy = sac.test_agent(T)
        distributions.append(p_xy)
        weights = utils.get_weights(distributions)
        
        epoch = 'epoch_%02d' % (i)
        if args.render:
            print('Collecting videos....') 
            sac.record(T=args.record_steps, n=1, video_dir=video_dir+'/baseline/'+epoch, on_policy=False) 
            sac.record(T=args.record_steps, n=1, video_dir=video_dir+'/entropy/'+epoch, on_policy=True) 

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        print("Executing mixed policy...")
        data, average_p_xy, initial_state = execute_average_policy(env, policies, T, weights, 
                                                                   initial_state=initial_state, n=args.n, render=args.render, 
                                                                   video_dir=video_dir+'/mixed/'+epoch, epoch=i)
        test_data,_,_ = execute_average_policy(env, policies, T=2000, weights=weights,
                                               initial_state=initial_state, n=1, 
                                               render=False, epoch=i)
        
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
        reward_fn = RewardFn(data, n_components=8, eps=.001)
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


