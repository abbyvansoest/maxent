import gym
import numpy as np 
import os

if not os.path.exists("data"):
    os.mkdir("data")

steps = 20000000
test_steps = 5000

env = gym.make("Ant-v2")
ant_data = []
ant_test_data = []

# collect training data
print("collecting Ant training data")
obs = env.reset()
for i in range(int(steps/100)):
    for j in range(int(steps/(steps/100))):
        ant_data.append(obs[:29])
        action = env.action_space.sample()
        obs,r,d,_ = env.step(action)
    
    obs = env.reset()
    
    if i % 1000 == 0:
        print(i)
    
# collect test data
print("collecting Ant testing data")
obs = env.reset()
for i in range(int(test_steps/100)):
    for j in range(int(test_steps/(test_steps/100))):
        ant_test_data.append(obs[:29])
        action = env.action_space.sample()
        obs,r,d,_ = env.step(action)
        
    obs = env.reset()
    
    if i % 1000 == 0:
        print(i)

np.save("data/ant_data_reset", ant_data)
np.save("data/ant_test_data_reset", ant_test_data)

#######################################

env = gym.make("Humanoid-v2")
humanoid_data = []
humanoid_test_data = []

# collect training data
print("collecting Humanoid training data")
obs = env.reset()
for i in range(int(steps/100)):
    for j in range(int(steps/(steps/100))):
        humanoid_data.append(obs[:269])
        action = env.action_space.sample()
        obs,r,d,_ = env.step(action)
    
    obs = env.reset()
    
    if i % 1000 == 0:
        print(i)
    
# collect test data
print("collecting Humanoid testing data")
obs = env.reset()
for i in range(int(test_steps/100)):
    for j in range(int(test_steps/(test_steps/100))):
        humanoid_test_data.append(obs[:269])
        action = env.action_space.sample()
        obs,r,d,_ = env.step(action)
        
    obs = env.reset()
    
    if i % 1000 == 0:
        print(i)

np.save("data/humanoid_data_reset", humanoid_data)
np.save("data/humanoid_test_data_reset", humanoid_test_data)