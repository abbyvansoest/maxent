# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Ant-v2 OpenAI/Mujoco environment. 

Commands to recreate with geometric weighting:
python ant_collect_sac.py --env="Ant-v2" --exp_name=ant_experiment --T=10000 --n=20 --l=2 --hid=300 --epochs=20 --episodes=30 --gaussian --reduce_dim=5 --geometric
