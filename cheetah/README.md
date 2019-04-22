# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Swimmer-v2 OpenAI/Mujoco environment. 

Commands to recreate with geometric weighting:
python cheetah_collect_sac.py --env="HalfCheetah-v2" --exp_name=cheetah_experiment --T=10000 --n=20 --l=2 --hid=300 --epochs=30 --episodes=30 --gaussian --reduce_dim=4 --geometric
