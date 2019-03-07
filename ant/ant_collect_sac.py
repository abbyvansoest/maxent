# Collect entropy-based reward policies.

# python ant_collect_sac.py --env="Ant-v2" --exp_name=test --T=1000 --n=20 --l=2 --hid=300 --epochs=16 --episodes=16 --gaussian --reduce_dim=5

# for discretizing with autoencoding
# python ant_collect_sac.py --env="Ant-v2" --exp_name=_discretize_autoencoder_6 --T=1000 --n=20 --l=2 --hid=300 --epochs=16 --episodes=30 --autoencode --autoencoder_reduce_dim=6

import sys
sys.path.append('/home/abby')

import os
import time
from datetime import datetime
import random

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm
from tabulate import tabulate

import gym
import tensorflow as tf

import utils
import ant_utils
import plotting
from ant_soft_actor_critic import AntSoftActorCritic
from experience_buffer import ExperienceBuffer

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

# collect data to be learned by autoencoder
def collect_avg_obs(env, policies, T, n=100):
    data = []
    max_idx = len(policies) - 1
    
    utils.log_statement('Collecting ' + str(n*T) + ' steps')
    
    for iteration in range(n):        
        env.reset()
        obs = get_state(env, env.env._get_obs())
       
        for t in range(T):
            action = np.zeros(shape=(1,ant_utils.action_dim))
            
            # select random policy uniform distribution
            # take non-deterministic action for that policy
            idx = random.randint(0, max_idx)
            if idx == 0:
                action = env.action_space.sample()
            else:
                action = policies[idx].get_action(obs, deterministic=args.deterministic)
                
            # Count the cumulative number of new states visited as a function of t.
            obs, _, done, _ = env.step(action)
            data.append(obs[:29])
            obs = get_state(env, obs)
        
        print('Iteration %i/%i' % (iteration, n))
    
    return data

def get_state(env, obs):
    state = env.env.state_vector()
    if not np.array_equal(obs[:len(state) - 2], state[2:]):
        utils.log_statement(obs)
        utils.log_statement(state)
        raise ValueError("state and observation are not equal")
    return state

def compute_states_visited_xy(env, policies, T, n, N=20, initial_state=[], baseline=False):
    
    states_visited_xy = np.zeros(T*n)
    max_idx = len(policies) - 1
    
    for it in range(N): 
        p_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d))) 
        cumulative_states_visited_xy = 0
        
        # average results over n rollouts
        for iteration in range(n):

            env.reset()
            if len(initial_state) > 0:
                qpos = initial_state[:len(ant_utils.qpos)]
                qvel = initial_state[len(ant_utils.qpos):]
                env.env.set_state(qpos, qvel)
            obs = get_state(env, env.env._get_obs())

            for t in range(T):
                action = np.zeros(shape=(1,ant_utils.action_dim))
                idx = random.randint(0, max_idx)
                
                if idx == 0 or baseline:
                    action = env.action_space.sample()
                else:
                    action = policies[idx].get_action(obs, deterministic=args.deterministic)

                # Count the cumulative number of new states visited as a function of t.
                obs, _, done, _ = env.step(action)
                obs = get_state(env, obs)

                # if this is the first time you are seeing this state, increment.
                if p_xy[tuple(ant_utils.discretize_state_2d(obs, norm, env))]  == 0:
                    cumulative_states_visited_xy += 1
                
                step = iteration*T + t
                states_visited_xy[step] += cumulative_states_visited_xy
                p_xy[tuple(ant_utils.discretize_state_2d(obs, norm, env))] += 1

                if done: # CRITICAL: ignore done signal
                    done = False
                
    env.close()
    states_visited_xy /= float(N)
    return states_visited_xy

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T,
                           reward_fn=[], norm=[], initial_state=[], 
                           n=10, render=False, epoch=0):
    
    p = np.zeros(shape=(tuple(ant_utils.num_states)))
    p_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))
    random_initial_state = []
    
    discretized = []
    
    cumulative_states_visited = 0
    states_visited = []
    cumulative_states_visited_xy = 0
    states_visited_xy = []
    
    rewards = np.zeros(T)
    
    denom = 0
    max_idx = len(policies) - 1

    # average results over n rollouts
    for iteration in range(n):
        
        env.reset()
        
        # TODO: when testing, do not want initial state.
        if len(initial_state) > 0:
            qpos = initial_state[:len(ant_utils.qpos)]
            qvel = initial_state[len(ant_utils.qpos):]
            env.env.set_state(qpos, qvel)

        obs = get_state(env, env.env._get_obs())

        random_T = np.floor(random.random()*T)
        random_initial_state = []
       
        for t in range(T):
            
            action = np.zeros(shape=(1,ant_utils.action_dim))
            
            if args.max_sigma:
                mu = np.zeros(shape=(1,ant_utils.action_dim))
                sigma = np.zeros(shape=(1,ant_utils.action_dim))
                mean_sigma = np.zeros(shape=(1,ant_utils.action_dim))
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
            obs = get_state(env, obs)
            
            # TODO: collect all discretized obs. compute histogram for
            # each axis of the obs.
            discretized_obs = ant_utils.discretize_state(obs, norm, env)
            discretized.append(discretized_obs)
            
            reward = reward_fn[tuple(discretized_obs)]
            rewards[t] += reward

            # if this is the first time you are seeing this state, increment.
            if p[tuple(ant_utils.discretize_state(obs, norm, env))] == 0:
                cumulative_states_visited += 1
            states_visited.append(cumulative_states_visited)
            if p_xy[tuple(ant_utils.discretize_state_2d(obs, norm, env))]  == 0:
                cumulative_states_visited_xy += 1
            states_visited_xy.append(cumulative_states_visited_xy)

            p[tuple(ant_utils.discretize_state(obs, norm, env))] += 1
            p_xy[tuple(ant_utils.discretize_state_2d(obs, norm, env))] += 1
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
    plotting.discretized_histograms(discretized, epoch)

    p /= float(denom)
    p_xy /= float(denom)

    return p, p_xy, random_initial_state, states_visited, states_visited_xy

def grad_ent(pt):
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p

    eps = 1/np.sqrt(ant_utils.total_state_space)
    return 1/(pt + eps)

def init_state(env):    
    env.env.set_state(ant_utils.qpos, ant_utils.qvel)
    state = env.env.state_vector()
    return state

def entropy(pt):
    utils.log_statement("pt size %d" % pt.size)
    # entropy = -sum(pt*log(pt))
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
    states_visited_indexes = [0,5,10,15]
    
    states_visited_cumulative = []
    states_visited_cumulative_baseline = []

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))
    running_avg_ent = 0
    running_avg_ent_xy = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_baseline_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))
    running_avg_ent_baseline = 0
    running_avg_ent_baseline_xy = 0

    pct_visited = []
    pct_visited_baseline = []
    pct_visited_xy = []
    pct_visited_xy_baseline = []

    running_avg_entropies = []
    running_avg_entropies_xy = []
    running_avg_ps_xy = []
    avg_ps_xy = []

    running_avg_entropies_baseline = []
    running_avg_entropies_baseline_xy = []
    running_avg_ps_baseline_xy = []
    avg_ps_baseline_xy = []

    policies = []
    initial_state = init_state(env)
    
    prebuf = ExperienceBuffer()
    env.reset()
    for t in range(10000):     
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        prebuf.store(get_state(env, obs))
        if done:
            env.reset()
            done = False
            
    prebuf.normalize()
    normalization_factors = prebuf.normalization_factors
    utils.log_statement(normalization_factors)
    prebuf = None
    if not args.gaussian:
        normalization_factors = []

    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))

    for i in range(epochs):
        utils.log_statement("*** ------- EPOCH %d ------- ***" % i)
        
        # clear initial state if applicable.
        if not args.initial_state:
            initial_state = []
        else:
            utils.log_statement(initial_state)
        utils.log_statement("max reward: " + str(np.max(reward_fn)))

        logger_kwargs = setup_logger_kwargs("model" + str(i), data_dir=experiment_directory)

        # Learn policy that maximizes current reward function.
        print("Learning new oracle...")
        if args.seed != -1:
            seed = args.seed
        else:
            seed = random.randint(1, 100000)
        
        sac = AntSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=seed, gamma=args.gamma, 
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

        if args.render:
            epoch = 'epoch_%02d' % (i) 
            sac.record(T=1000, video_dir=video_dir+'/baseline/'+epoch, on_policy=False) 
            sac.record(T=1000, video_dir=video_dir+'/entropy/'+epoch, on_policy=True) 
        
        if args.autoencode:
            print("Learning autoencoding....")
            train = collect_avg_obs(env, policies, T=1000, n=1000)
            test = collect_avg_obs(env, policies, T=1000, n=200)
            ant_utils.learn_encoding(train, test)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        print("Executing mixed policy...")
        average_p, average_p_xy, initial_state, states_visited, states_visited_xy = \
            execute_average_policy(env, policies, T,
                                   reward_fn=reward_fn, norm=normalization_factors, 
                                   initial_state=initial_state, n=args.n, 
                                   render=False, epoch=i)
        
        print("Calculating maxEnt entropy...")
        round_entropy = entropy(average_p.ravel())
        round_entropy_xy = entropy(average_p_xy.ravel())
        
        # Update running averages for maxEnt.
        print("Updating maxEnt running averages...")
        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_entropy/float(i+1)
        running_avg_ent_xy = running_avg_ent_xy * (i)/float(i+1) + round_entropy_xy/float(i+1)
        running_avg_p *= (i)/float(i+1)
        running_avg_p += average_p/float(i+1)
        running_avg_p_xy *= (i)/float(i+1)
        running_avg_p_xy += average_p_xy/float(i+1)
        
        # update reward function
        print("Update reward function")
        eps = 1/np.sqrt(ant_utils.total_state_space)
        if args.cumulative:
            reward_fn = grad_ent(running_avg_p)
        else:
            reward_fn = 1.
            average_p += eps
            reward_fn /= average_p
        average_p = None # delete big array
        
        # (save for plotting)
        running_avg_entropies.append(running_avg_ent)
        running_avg_entropies_xy.append(running_avg_ent_xy)
        if i in indexes:
            running_avg_ps_xy.append(np.copy(running_avg_p_xy))
            avg_ps_xy.append(np.copy(average_p_xy))

        print("Collecting baseline experience....")
        p_baseline, p_baseline_xy, states_visited_baseline, states_visited_xy_baseline = sac.test_agent_random(T, normalization_factors=normalization_factors, n=args.n)
        
        plotting.states_visited_over_time(states_visited, states_visited_baseline, i)
        plotting.states_visited_over_time(states_visited_xy, states_visited_xy_baseline, i, ext='_xy')
        
        # save for cumulative plot.
        if i in states_visited_indexes:
            # average over a whole bunch of rollouts
            # slow: so only do this when needed.
            print("Averaging unique xy states visited....")
            states_visited_xy = compute_states_visited_xy(env, policies, T=T, n=args.n, N=args.avg_N)
            states_visited_xy_baseline = compute_states_visited_xy(env, policies, 
                                                                   T=T, n=args.n, N=args.avg_N, 
                                                                   initial_state=initial_state, 
                                                                   baseline=True)
            states_visited_cumulative.append(states_visited_xy)
            states_visited_cumulative_baseline.append(states_visited_xy_baseline)

        print("Compute baseline entropy....")
        round_entropy_baseline = entropy(p_baseline.ravel())
        round_entropy_baseline_xy = entropy(p_baseline_xy.ravel())

        # Update baseline running averages.
        print("Updating baseline running averages...")
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_ent_baseline_xy = running_avg_ent_baseline_xy * (i)/float(i+1) + round_entropy_baseline_xy/float(i+1)

        running_avg_p_baseline *= (i)/float(i+1) 
        running_avg_p_baseline += p_baseline/float(i+1)
        running_avg_p_baseline_xy *= (i)/float(i+1) 
        running_avg_p_baseline_xy += p_baseline_xy/float(i+1)
        
        p_baseline = None
        
        # (save for plotting)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_entropies_baseline_xy.append(running_avg_ent_baseline_xy)
        if i in indexes:
            running_avg_ps_baseline_xy.append(np.copy(running_avg_p_baseline_xy))
            avg_ps_baseline_xy.append(np.copy(p_baseline_xy))
    
        utils.log_statement(average_p_xy)
        utils.log_statement(p_baseline_xy)
        
        # Calculate percent of state space visited.
        pct = np.count_nonzero(running_avg_p)/float(running_avg_p.size)
        pct_visited.append(pct)
        pct_xy = np.count_nonzero(running_avg_p_xy)/float(running_avg_p_xy.size)
        pct_visited_xy.append(pct_xy)
        
        pct_baseline = np.count_nonzero(running_avg_p_baseline)/float(running_avg_p_baseline.size)
        pct_visited_baseline.append(pct_baseline)
        pct_xy_baseline = np.count_nonzero(running_avg_p_baseline_xy)/float(running_avg_p_baseline_xy.size)
        pct_visited_xy_baseline.append(pct_xy_baseline)
        
        # Print round summary.
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy_xy", 
                "running_avg_ent_xy", 
                "round_entropy", 
                "running_avg_ent", 
                "% state space xy", 
                "% total state space"]
        col2 = [round_entropy_baseline_xy, running_avg_ent_baseline_xy, 
                round_entropy_baseline, running_avg_ent_baseline, 
                pct_xy_baseline, pct_baseline]
        col3 = [round_entropy_xy, running_avg_ent_xy, 
                round_entropy, running_avg_ent, 
                pct_xy, pct]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        utils.log_statement(table)
        
        # Plot from round.
        plotting.heatmap(running_avg_p_xy, average_p_xy, i)
        plotting.heatmap1(running_avg_p_baseline_xy, i)
        
        if i == states_visited_indexes[3]:
             plotting.states_visited_over_time_multi(states_visited_cumulative, 
                                                     states_visited_cumulative_baseline, 
                                                     states_visited_indexes)
        
    # cumulative plots.
    plotting.heatmap4(running_avg_ps_xy, running_avg_ps_baseline_xy, indexes, ext="cumulative")
    plotting.heatmap4(avg_ps_xy, avg_ps_baseline_xy, indexes, ext="epoch")
    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    plotting.running_average_entropy(running_avg_entropies_xy, running_avg_entropies_baseline_xy, ext='_xy')
    plotting.percent_state_space_reached(pct_visited, pct_visited_baseline, ext='_total')
    plotting.percent_state_space_reached(pct_visited_xy, pct_visited_xy_baseline, ext="_xy")
    
    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100, linewidth=150, precision=8)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment

    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
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


