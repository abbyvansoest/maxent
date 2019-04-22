# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Humanoid-v2 OpenAI/Mujoco environment. 

Entropy-based exploration is a new algorithm to encourage efficient discovery of an unknown state space in RL problems: https://arxiv.org/abs/1812.02690

Note that this code is memory-intensive. It is set up to run on a specialized deep-learning machine. To reduce the dimensionality, change the discretization setup in swimmer_utils.py.

Dependencies: Tensoflow, OpenAI Gym/Mujoco license, matplotlib, numpy, OpenAI SpinningUp, scipy

Commands to recreate:
python humanoid_collect_sac.py --env="Humanoid-v2" --exp_name=test --T=50000 --n=20 --l=2 --hid=300 --epochs=30 --episodes=30 --geometric
