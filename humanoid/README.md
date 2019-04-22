# Efficient Maximum-Entropy Exploration

This is the experimental repository for entropy-based exploration in the Humanoid-v2 OpenAI/Mujoco environment. 

Commands to recreate:
python humanoid_collect_sac.py --env="Humanoid-v2" --exp_name=test --T=50000 --n=20 --l=2 --hid=300 --epochs=30 --episodes=30 --geometric
