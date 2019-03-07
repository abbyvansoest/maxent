import gym
from gym import wrappers

env = gym.make("Ant-v2")
wrapped_env = wrappers.Monitor(env, 'test_videos/')
o = wrapped_env.reset()

for i in range(1000):
    print(i)
    a = wrapped_env.unwrapped.action_space.sample()
    o,r,d,_ = wrapped_env.step(a)
    if d:
        break

wrapped_env.close()