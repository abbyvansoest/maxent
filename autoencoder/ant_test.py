import gym
import numpy as np 
from custom import CustomAutoencoder

env = gym.make("Ant-v2")

print("loading data")
data = list(np.load("data/ant_data.npy"))
test_data = list(np.load("data/ant_test_data.npy"))

autoencoder = CustomAutoencoder(len(data[0]), 
                                num_hid1=20,
                                num_hid2=12,
                                reduce_dim=6, 
                                normalize=False)
autoencoder.set_data(data)
autoencoder.set_test_data(test_data)
autoencoder.train(testiter=2)



