# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Walker2d-v2 OpenAI/Mujoco environment. 

Commands to recreate:
python swimmer_collect_sac.py --env="Swimmer-v2" --T=10000 --epochs=26 --episodes=30 --gaussian --exp_name=swimmer_experiment --reduce_dim=5 --n=10 --l=2 --avg_N=10
