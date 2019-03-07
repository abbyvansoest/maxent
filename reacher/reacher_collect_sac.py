# Collect entropy-based reward policies.

# python reacher_collect_sac.py --env="Reacher-v2" --T=1000 --episodes=100 --epochs=10 

# TODO: normalize obs for autoencoder
# TODO: make sure graph is being reset? Or not?

import sys
sys.path.append('/home/abby')

import os
import time

import random
import numpy as np
from tabulate import tabulate

import gym
import tensorflow as tf
from math import log, e

import utils
import reacher_utils
import plotting
from reacher_soft_actor_critic import ReacherSoftActorCritic
from experience_buffer import ExperienceBuffer
# from autoencoder.sparse import SparseAutoencoder
from autoencoder.contractive import ContractiveAutoencoder

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

def compute_states_visited_xy(env, policies, T, n, N=20, initial_state=[], baseline=False):
    
    states_visited_xy = np.zeros(T*n)
    max_idx = len(policies) - 1
    
    for it in range(N): 
        print(it)
        
        p_xy = np.zeros(shape=(tuple(reacher_utils.num_states_2d))) 
        cumulative_states_visited_xy = 0
        
        # average results over n rollouts
        for iteration in range(n):

            env.reset()
            if len(initial_state) > 0:
                qpos = initial_state[:len(reacher_utils.qpos)]
                qvel = initial_state[len(reacher_utils.qpos):]
                env.env.set_state(qpos, qvel)
            obs = reacher_utils.get_state(env, env.env._get_obs())

            for t in range(T):
                action = np.zeros(shape=(1,reacher_utils.action_dim))
                idx = random.randint(0, max_idx)
                
                if idx == 0 or baseline:
                    action = env.action_space.sample()
                else:
                    action = policies[idx].get_action(obs, deterministic=args.deterministic)

                # Count the cumulative number of new states visited as a function of t.
                obs, _, done, _ = env.step(action)
                obs = reacher_utils.get_state(env, obs)

                # if this is the first time you are seeing this state, increment.
                if p_xy[tuple(reacher_utils.discretize_state_2d(obs, norm))]  == 0:
                    cumulative_states_visited_xy += 1
                
                step = iteration*T + t
                states_visited_xy[step] += cumulative_states_visited_xy
                p_xy[tuple(reacher_utils.discretize_state_2d(obs, norm))] += 1

                if done: # CRITICAL: ignore done signal
                    done = False
                
    env.close()
    states_visited_xy /= float(N)
    return states_visited_xy

# run a simulation to see how the average policy behaves.
def collect_avg_obs(env, policies, T, n=10):
    data = []
    max_idx = len(policies) - 1
    
    for iteration in range(n):        
        env.reset()
        obs = reacher_utils.get_state(env, env.env._get_obs())
       
        for t in range(T):
            
            action = np.zeros(shape=(1,reacher_utils.action_dim))
            
            # select random policy uniform distribution
            # take non-deterministic action for that policy
            idx = random.randint(0, max_idx)
            if idx ==0:
                action = env.action_space.sample()
            else:
                action = policies[idx].get_action(obs, deterministic=args.deterministic)
                
            # Count the cumulative number of new states visited as a function of t.
            obs, _, done, _ = env.step(action)
            data.append(obs)
            obs = reacher_utils.get_state(env, obs)
    
    return data

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, reward_fn=[], norm=[], initial_state=[], n=10, render=False, epoch=0):
    
    p = np.zeros(shape=(tuple(reacher_utils.num_states)))
    p_joint0 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    p_joint1 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    random_initial_state = []
    
    cumulative_states_visited = 0
    states_visited = []
    cumulative_states_visited_joint0 = 0
    states_visited_joint0 = []
    cumulative_states_visited_joint1 = 0
    states_visited_joint1 = []
    
    rewards = np.zeros(T)
    
    denom = 0
    max_idx = len(policies) - 1

    # average results over n rollouts
    for iteration in range(n):
        
        env.reset()

        obs = reacher_utils.get_state(env, env.env._get_obs())

        random_T = np.floor(random.random()*T)
        random_initial_state = []
       
        for t in range(T):
            
            action = np.zeros(shape=(1,reacher_utils.action_dim))
            
            if args.max_sigma:
                mu = np.zeros(shape=(1,reacher_utils.action_dim))
                sigma = np.zeros(shape=(1,reacher_utils.action_dim))
                mean_sigma = np.zeros(shape=(1,reacher_utils.action_dim))
                for sac in policies:
                    mu += sac.get_action(obs, deterministic=True)
                    sigma = np.maximum(sigma, sac.get_sigma(obs))
                    mean_sigma += sac.get_sigma(obs)
                mu /= float(len(policies))
                mean_sigma /= float(len(policies))

                action = np.random.normal(loc=mu, scale=sigma)
            else:
                # select random policy uniform distribution
                # take non-deterministic action for that policy
                idx = random.randint(0, max_idx)
                if idx ==0:
                    action = env.action_space.sample()
                else:
                    action = policies[idx].get_action(obs, deterministic=args.deterministic)
                
            # Count the cumulative number of new states visited as a function of t.
            obs, _, done, _ = env.step(action)
            obs = reacher_utils.get_state(env, obs)
            
            reward = reward_fn[tuple(reacher_utils.discretize_state(obs, norm))]
            rewards[t] += reward

            # if this is the first time you are seeing this state, increment.
            if p[tuple(reacher_utils.discretize_state(obs, norm))] == 0:
                cumulative_states_visited += 1
            states_visited.append(cumulative_states_visited)
            if p_joint0[tuple(reacher_utils.discretize_state_2d(obs, reacher_utils.joint0th, reacher_utils.joint0v, norm))]  == 0:
                cumulative_states_visited_joint0 += 1
            states_visited_joint0.append(cumulative_states_visited_joint0)
            
            if p_joint1[tuple(reacher_utils.discretize_state_2d(obs, reacher_utils.joint1th, reacher_utils.joint1v, norm))]  == 0:
                cumulative_states_visited_joint1 += 1
            states_visited_joint1.append(cumulative_states_visited_joint1)

            p[tuple(reacher_utils.discretize_state(obs, norm))] += 1
            p_joint0[tuple(reacher_utils.discretize_state_2d(obs, reacher_utils.joint0th, reacher_utils.joint0v, norm))] += 1
            p_joint1[tuple(reacher_utils.discretize_state_2d(obs, reacher_utils.joint1th, reacher_utils.joint1v, norm))] += 1
            denom += 1
            
            if t == random_T:
                random_initial_state = obs

            if render:
                env.render()
            if done: # CRITICAL: ignore done signal
                done = False
                
    env.close()
    rewards /= float(n)
    plotting.reward_vs_t(rewards, epoch)

    p /= float(denom)
    p_joint0 /= float(denom)
    p_joint1 /= float(denom)

    return p, p_joint0, p_joint1, random_initial_state, states_visited, states_visited_joint0, states_visited_joint1

def grad_ent(pt):
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p

    eps = 1/np.sqrt(reacher_utils.total_state_space)
    return 1/(pt + eps)

def init_state(env):    
    env.env.set_state(reacher_utils.qpos, reacher_utils.qvel)
    state = env.env.state_vector()
    return state

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
    
    direct = os.getcwd()+ '/data/'
    experiment_directory = direct + args.exp_name
    print(experiment_directory)
    
    indexes = [1,5,10,15]
    states_visited_indexes = [0,5,10,15]
    
#     indexes = [0,1,2,3]
#     states_visited_indexes = [0,1,2,3]
    
    states_visited_cumulative = []
    states_visited_cumulative_baseline = []

    running_avg_p = np.zeros(shape=(tuple(reacher_utils.num_states)))
    running_avg_p_joint0 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    running_avg_p_joint1 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    running_avg_ent = 0
    running_avg_ent_joint0 = 0
    running_avg_ent_joint1 = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(reacher_utils.num_states)))
    running_avg_p_baseline_joint0 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    running_avg_p_baseline_joint1 = np.zeros(shape=(tuple(reacher_utils.num_states_2d)))
    running_avg_ent_baseline = 0
    running_avg_ent_baseline_joint0 = 0
    running_avg_ent_baseline_joint1 = 0

    pct_visited = []
    pct_visited_baseline = []
    pct_visited_joint0 = []
    pct_visited_joint0_baseline = []
    pct_visited_joint1 = []
    pct_visited_joint1_baseline = []

    running_avg_entropies = []
    running_avg_entropies_joint0 = []
    running_avg_ps_joint0 = []
    avg_ps_joint0 = []
    
    running_avg_entropies_joint1 = []
    running_avg_ps_joint1 = []
    avg_ps_joint1 = []

    running_avg_entropies_baseline = []
    running_avg_entropies_baseline_joint0 = []
    running_avg_ps_baseline_joint0 = []
    avg_ps_baseline_joint0 = []
    
    running_avg_entropies_baseline_joint1 = []
    running_avg_ps_baseline_joint1 = []
    avg_ps_baseline_joint1 = []

    policies = []
    initial_state = init_state(env)
    
    normalization_factors = []
    if args.gaussian:
        prebuf = ExperienceBuffer()
        env.reset()
        for t in range(10000):     
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)

            prebuf.store(reacher_utils.get_state(env, obs))
            
            if done:
                env.reset()
                done = False
                
        prebuf.normalize()
        normalization_factors = prebuf.normalization_factors
        print(normalization_factors)
        prebuf = None

    reward_fn = np.zeros(shape=(tuple(reacher_utils.num_states)))

    for i in range(epochs):
        print("*** ------- EPOCH %d ------- ***" % i)
        
        # clear initial state if applicable.
        if not args.initial_state:
            initial_state = []
        else:
            print(initial_state)
            print(tuple(reacher_utils.discretize_state_2d(initial_state, normalization_factors)))
            print(tuple(reacher_utils.discretize_state(initial_state, normalization_factors)))
        print("max reward: " + str(np.max(reward_fn)))

        logger_kwargs = setup_logger_kwargs("model" + str(i), data_dir=experiment_directory)

        # Learn policy that maximizes current reward function.
        print("Learning new oracle...")
        if args.seed != -1:
            seed = args.seed
        else:
            seed = int(time.time())
        
        sac = ReacherSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=seed, gamma=args.gamma, max_ep_len=T,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            logger_kwargs=logger_kwargs, 
            normalization_factors=normalization_factors,
            learn_reduced=args.learn_reduced)
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
        average_p, average_p_joint0, average_p_joint1, initial_state, \
        states_visited, states_visited_joint0, states_visited_joint1 = \
            execute_average_policy(env, policies, T,
                                   reward_fn=reward_fn, norm=normalization_factors, 
                                   initial_state=initial_state, n=args.n, 
                                   render=False, epoch=i)

        print("Learning autoencoding....")
        autoencoder = ContractiveAutoencoder(reacher_utils.env_state_dim, reduce_dim=4)
        autoencoder.set_data(collect_avg_obs(env, policies, T, n=500))
        autoencoder.set_test_data(collect_avg_obs(env, policies, T, n=50))
        autoencoder.train()

        print("Calculating maxEnt entropy...")
        round_entropy = entropy(average_p.ravel())
        round_entropy_joint0 = entropy(average_p_joint0.ravel())
        round_entropy_joint1 = entropy(average_p_joint1.ravel())
        
        # Update running averages for maxEnt.
        print("Updating maxEnt running averages...")
        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_entropy/float(i+1)
        running_avg_ent_joint0 = running_avg_ent_joint0 * (i)/float(i+1) + round_entropy_joint0/float(i+1)
        running_avg_ent_joint1 = running_avg_ent_joint1 * (i)/float(i+1) + round_entropy_joint1/float(i+1)
        running_avg_p *= (i)/float(i+1)
        running_avg_p += average_p/float(i+1)
        running_avg_p_joint0 *= (i)/float(i+1)
        running_avg_p_joint0 += average_p_joint0/float(i+1)
        running_avg_p_joint1 *= (i)/float(i+1)
        running_avg_p_joint1 += average_p_joint1/float(i+1)
        
        # update reward function
        print("Update reward function")
        eps = 1/np.sqrt(reacher_utils.total_state_space)
        if args.cumulative:
            reward_fn = grad_ent(running_avg_p)
        else:
            reward_fn = 1.
            average_p += eps
            reward_fn /= average_p
        average_p = None # delete big array
        
        # (save for plotting)
        running_avg_entropies.append(running_avg_ent)
        running_avg_entropies_joint0.append(running_avg_ent_joint0)
        running_avg_entropies_joint1.append(running_avg_ent_joint1)
        if i in indexes:
            running_avg_ps_joint0.append(np.copy(running_avg_p_joint0))
            avg_ps_joint0.append(np.copy(average_p_joint0))
            
            running_avg_ps_joint1.append(np.copy(running_avg_p_joint1))
            avg_ps_joint1.append(np.copy(average_p_joint1))

        print("Collecting baseline experience....")
        p_baseline, p_baseline_joint0, p_baseline_joint1, \
        states_visited_baseline, states_visited_joint0_baseline, \
        states_visited_joint1_baseline = sac.test_agent_random(T, normalization_factors=normalization_factors, n=args.n)
        
        plotting.states_visited_over_time(states_visited, states_visited_baseline, i)
        plotting.states_visited_over_time(states_visited_joint0, states_visited_joint0_baseline, i, ext='_joint0')
        plotting.states_visited_over_time(states_visited_joint1, states_visited_joint1_baseline, i, ext='_joint1')
        
        # save for cumulative plot.
        # if i in states_visited_indexes:
        #     # average over a whole bunch of rollouts
        #     # slow: so only do this when needed.
        #     print("Averaging unique xy states visited....")
        #     states_visited_joint0 = compute_states_visited_xy(env, policies, T=T, n=args.n, N=args.avg_N)
        #     states_visited_joint0_baseline = compute_states_visited_xy(env, policies, 
        #                                                            T=T, n=args.n, N=args.avg_N, 
        #                                                            initial_state=initial_state, 
        #                                                            baseline=True)
        #     states_visited_cumulative.append(states_visited_joint0)
        #     states_visited_cumulative_baseline.append(states_visited_joint0_baseline)

        print("Compute baseline entropy....")
        round_entropy_baseline = entropy(p_baseline.ravel())
        round_entropy_baseline_joint0 = entropy(p_baseline_joint0.ravel())
        round_entropy_baseline_joint1 = entropy(p_baseline_joint1.ravel())

        # Update baseline running averages.
        print("Updating baseline running averages...")
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_ent_baseline_joint0 = running_avg_ent_baseline_joint0 * (i)/float(i+1) + round_entropy_baseline_joint0/float(i+1)
        running_avg_ent_baseline_joint1 = running_avg_ent_baseline_joint1 * (i)/float(i+1) + round_entropy_baseline_joint1/float(i+1)

        running_avg_p_baseline *= (i)/float(i+1) 
        running_avg_p_baseline += p_baseline/float(i+1)
        running_avg_p_baseline_joint0 *= (i)/float(i+1) 
        running_avg_p_baseline_joint0 += p_baseline_joint0/float(i+1)
        running_avg_p_baseline_joint1 *= (i)/float(i+1) 
        running_avg_p_baseline_joint1 += p_baseline_joint1/float(i+1)
        
        p_baseline = None
        
        # (save for plotting)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_entropies_baseline_joint0.append(running_avg_ent_baseline_joint0)
        running_avg_entropies_baseline_joint1.append(running_avg_ent_baseline_joint1)
        if i in indexes:
            running_avg_ps_baseline_joint0.append(np.copy(running_avg_p_baseline_joint0))
            avg_ps_baseline_joint0.append(np.copy(p_baseline_joint0))
            
            running_avg_ps_baseline_joint1.append(np.copy(running_avg_p_baseline_joint1))
            avg_ps_baseline_joint1.append(np.copy(p_baseline_joint1))
    
        print(average_p_joint0)
        print(p_baseline_joint0)
        print(average_p_joint1)
        print(p_baseline_joint1)
        
        # Calculate percent of state space visited.
        pct = np.count_nonzero(running_avg_p)/float(running_avg_p.size)
        pct_visited.append(pct)
        pct_joint0 = np.count_nonzero(running_avg_p_joint0)/float(running_avg_p_joint0.size)
        pct_visited_joint0.append(pct_joint0)
        pct_joint1 = np.count_nonzero(running_avg_p_joint1)/float(running_avg_p_joint1.size)
        pct_visited_joint1.append(pct_joint1)
        
        pct_baseline = np.count_nonzero(running_avg_p_baseline)/float(running_avg_p_baseline.size)
        pct_visited_baseline.append(pct_baseline)
        pct_joint0_baseline = np.count_nonzero(running_avg_p_baseline_joint0)/float(running_avg_p_baseline_joint0.size)
        pct_visited_joint0_baseline.append(pct_joint0_baseline)
        pct_joint1_baseline = np.count_nonzero(running_avg_p_baseline_joint1)/float(running_avg_p_baseline_joint1.size)
        pct_visited_joint1_baseline.append(pct_joint1_baseline)
        
        # Print round summary.
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy_joint0", 
                "running_avg_ent_joint0", 
                "round_entropy_joint1", 
                "running_avg_ent_joint1", 
                "round_entropy", 
                "running_avg_ent", 
                "% xy joint0",
                "% xy joint1",
                "% total state space"]
        col2 = [round_entropy_baseline_joint0, running_avg_ent_baseline_joint0, 
                round_entropy_baseline_joint1, running_avg_ent_baseline_joint1, 
                round_entropy_baseline, running_avg_ent_baseline, 
                pct_joint0_baseline, pct_joint1_baseline, pct_baseline]
        col3 = [round_entropy_joint0, running_avg_ent_joint0,
                round_entropy_joint1, running_avg_ent_joint1,
                round_entropy, running_avg_ent, 
                pct_joint0, pct_joint1, pct]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        print(table)
        
        # Plot from round.
        plotting.heatmap(running_avg_p_joint0, average_p_joint0, i, directory='joint0_')
        plotting.heatmap(running_avg_p_joint1, average_p_joint1, i, directory='joint1_')
        plotting.heatmap1(running_avg_p_baseline_joint0, i, directory='baseline_joint0')
        plotting.heatmap1(running_avg_p_baseline_joint1, i, directory='baseline_joint0')
        
#         if i == states_visited_indexes[3]:
#              plotting.states_visited_over_time_multi(states_visited_cumulative, 
#                                                      states_visited_cumulative_baseline, 
#                                                      states_visited_indexes)
        
    # cumulative plots.
    plotting.heatmap4(running_avg_ps_joint0, running_avg_ps_baseline_joint0, indexes, ext="cumulative_joint0")
    plotting.heatmap4(avg_ps_joint0, avg_ps_baseline_joint0, indexes, ext="epoch_joint0")
    plotting.heatmap4(running_avg_ps_joint1, running_avg_ps_baseline_joint1, indexes, ext="cumulative_joint1")
    plotting.heatmap4(avg_ps_joint1, avg_ps_baseline_joint1, indexes, ext="epoch_joint1")
    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    plotting.running_average_entropy(running_avg_entropies_joint0, running_avg_entropies_baseline_joint0, ext='_joint0')
    plotting.percent_state_space_reached(pct_visited, pct_visited_baseline, ext='_total')
    plotting.percent_state_space_reached(pct_visited_joint0, pct_visited_joint0_baseline, ext="_joint0")
    plotting.running_average_entropy(running_avg_entropies_joint1, running_avg_entropies_baseline_joint1, ext='_joint1')
    plotting.percent_state_space_reached(pct_visited_joint1, pct_visited_joint1_baseline, ext="_joint1")
    
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


