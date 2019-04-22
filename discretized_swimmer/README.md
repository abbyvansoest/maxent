# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Swimmer-v2 OpenAI/Mujoco environment. 

Commands to recreate with geometric weighting:
python swimmer_collect_sac.py --env="Swimmer-v2" --exp_name=swimmer_experiment --T=10000 --n=20 --l=2 --hid=300 --epochs=20 --episodes=30 --gaussian --reduce_dim=4 --geometric

