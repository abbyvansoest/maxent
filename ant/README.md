# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Ant-v2 OpenAI/Mujoco environment. 

Entropy-based exploration is a new algorithm to encourage efficient discovery of an unknown state space in RL problems: https://arxiv.org/abs/1812.02690

Note that this code is memory-intensive. It is set up to run on a specialized deep-learning machine. To reduce the dimensionality, change the discretization setup in ant_utils.py.

Dependencies: Tensoflow, OpenAI Gym/Mujoco license, matplotlib, numpy, OpenAI SpinningUp, scipy

Commands to recreate:
python ant_collect_sac.py --env="Ant-v2" --T=10000 --epochs=26 --episodes=30 --gaussian --exp_name=ant_experiment --reduce_dim=5 --n=10 --l=2 --avg_N=10

Example plotting command:

python -m spinup.run plot data/testing_save/model1/ data/testing_save/model2/ --xaxis Epoch --value AverageEpRet