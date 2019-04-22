# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Ant-v2 OpenAI/Mujoco environment. 

Entropy-based exploration is a new algorithm to encourage efficient discovery of an unknown state space in RL problems: https://arxiv.org/abs/1812.02690

Note that this code is memory-intensive. It is set up to run on a specialized deep-learning machine. To reduce the dimensionality, change the discretization setup in ant_utils.py.

Dependencies: Tensoflow, OpenAI Gym/Mujoco license, matplotlib, numpy, OpenAI SpinningUp, scipy

Commands to recreate:
python ant_collect_sac.py --env="Ant-v2" --exp_name=test --T=10000 --n=20 --l=2 --hid=300 --epochs=20 --episodes=30 --gaussian --reduce_dim=5 --geometric --avg_N=10

Example plotting command:

python -m spinup.run plot data/test/model1/ data/test/model2/ --xaxis Epoch --value AverageEpRet