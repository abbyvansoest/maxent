# maxent_base

Commands to recreate MountainCar experiments:
python collect_baseline.py --env="MountainCarContinuous-v0" --T=200 --train_steps=400 --episodes=300 --epochs=30 --exp_name=mountaincar_test

Commands to recreate Pendulum experiments:
python collect_baseline.py --env="Pendulum-v0" --T=200 --train_steps=200 --episodes=200 --epochs=15 --exp_name=pendulum_test
