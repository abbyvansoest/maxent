# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration: the MAXENT algorithm

MAXENT is a new algorithm to encourage efficient discovery of an unknown state space in RL problems: https://arxiv.org/abs/1812.02690

This repo contains the experimental code for various OpenAI/Mujoco environments: Swimmer, Ant, HalfCheetah, Walker2d, and Humanoid. The stable code for two classic control tasks is in a different repo: https://github.com/abbyvansoest/maxent_base

All implemetations use a forked copy of OpenAI Gym available at: https://github.com/abbyvansoest/gym-fork. Changes were made to the graphics used for rendering and the behavior of state reseting.

Note that this code is memory-intensive. It is set up to run on a specialized deep-learning machine. To reduce the dimensionality, change the discretization setup in swimmer_utils.py.

Dependencies: Tensoflow, OpenAI Gym/Mujoco license, matplotlib, numpy, OpenAI SpinningUp, scipy

See the respective directories for each environment for commands to run and recreate experiments. 
