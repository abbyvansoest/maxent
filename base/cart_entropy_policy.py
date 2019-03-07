import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import gym
from gym import wrappers
import utils

# Get the initial zero-state for the env.
def init_state(env):
    if env == "Pendulum-v0":
        return [np.pi, 0] 
    elif env == "MountainCarContinuous-v0":
        return [-0.50, 0]


class CartEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(CartEntropyPolicy, self).__init__()

        self.affine1 = nn.Linear(obs_dim, 128)
        self.middle = nn.Linear(128, 128)
        self.affine2 = nn.Linear(128, action_dim)

        torch.nn.init.xavier_uniform_(self.affine1.weight)
        torch.nn.init.xavier_uniform_(self.middle.weight)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.init_state = np.array(init_state(utils.args.env))
        self.env.seed(int(time.time())) # seed environment

    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.middle(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))

        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]

    def update_policy(self):
        R = 0
        policy_loss = []
        rewards = []

        # Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() # cost function?
        policy_loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.saved_log_probs.clear()

        return policy_loss

    def get_initial_state(self):
        if utils.args.env == "Pendulum-v0":
            self.env.env.state = [np.pi, 0] 
            theta, thetadot = self.env.env.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif utils.args.env == "MountainCarContinuous-v0":
            self.env.env.state = [-0.50, 0]
            return np.array(self.env.env.state)

    def get_obs(self):
        if utils.args.env == "Pendulum-v0":
            theta, thetadot = self.env.env.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
        elif utils.args.env == "MountainCarContinuous-v0":
            return np.array(self.env.env.state)

    def learn_policy(self, reward_fn, 
        episodes=1000, train_steps=1000, 
        initial_state=[], start_steps=10000):

        if len(initial_state) == 0:
            initial_state = self.init_state
        print("init: " + str(initial_state))

        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            if i_episode % 5 == 0:
                self.env.env.reset_state = initial_state
            self.env.reset()
            state = self.get_obs()
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, _, done, _ = self.env.step(action)
                reward = reward_fn[tuple(utils.discretize_state(state))]
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    # TODO: self.env.env.reset_state = initial_state ? 
                    self.env.reset()

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            loss = self.update_policy()
            running_loss = running_loss * 0.99 + loss*.01

            # Log to console.
            if i_episode % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

    def execute_internal(self, env, T, state, render):
        print("Simulation starting at = " + str(state))
        p = np.zeros(shape=(tuple(utils.num_states)))
        for t in range(T):  
            action = self.select_action(state)[0]
            state, reward, done, _ = env.step([action])
            p[tuple(utils.discretize_state(state))] += 1
            
            if render:
                env.render()
            if done:
                break
        env.close()
        return p

    def execute(self, T, initial_state=[], render=False, video_dir=''):

        p = np.zeros(shape=(tuple(utils.num_states)))

        if len(initial_state) == 0:
            initial_state = self.env.reset() # get random starting location

        print("initial_state= " + str(initial_state))

        if render:
            print("rendering env in execute()")
            wrapped_env = wrappers.Monitor(self.env, video_dir)
            wrapped_env.unwrapped.reset_state = initial_state
            state = wrapped_env.reset()
            state = self.get_obs()
            # print(initial_state)
            # print(state)
            p = self.execute_internal(wrapped_env, T, state, render)
        else:
            self.env.env.reset_state = initial_state
            state = self.env.reset()
            state = self.get_obs()

            print(state)
            print(initial_state)
            p = self.execute_internal(self.env, T, state, render)

        return p/float(T)

    def execute_random_internal(self, env, T, state, render):
        p = np.zeros(shape=(tuple(utils.num_states)))
        for t in range(T):  
            r = random.random()
            action = -1
            if (r < 1/3.):
                action = 0
            elif r < 2/3.:
                action = 1

            state, reward, done, _ = env.step([action])
            p[tuple(utils.discretize_state(state))] += 1
            
            if render:
                env.render()
            if done:
                break
        env.close()
        return p

    # TODO: render == True => record videos
    def execute_random(self, T, initial_state=[], render=False, video_dir=''):
        p = np.zeros(shape=(tuple(utils.num_states)))

        if len(initial_state) == 0:
            initial_state = self.env.reset() # get random starting location
            initial_state = self.init_state

        print("initial_state= " + str(initial_state))

        if render:
            print("rendering env in execute_random()")
            wrapped_env = wrappers.Monitor(self.env, video_dir)
            wrapped_env.unwrapped.reset_state = initial_state
            state = wrapped_env.reset()
            state = self.get_obs()
            p = self.execute_random_internal(wrapped_env, T, state, render)
        else:
            self.env.env.reset_state = initial_state
            state = self.env.reset()
            state = self.get_obs()

            print(state)
            print(initial_state)

            p = self.execute_random_internal(self.env, T, state, render)

        return p/float(T)

    def save(self, filename):
        self.env.close()
        torch.save(self, filename)