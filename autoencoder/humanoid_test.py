import gym
import numpy as np 
from custom import CustomAutoencoder

env = gym.make("Humanoid-v2")

print("loading data")
data = list(np.load('data/humanoid_data_reset.npy'))
test_data = list(np.load('data/humanoid_test_data_reset.npy'))

autoencoder = CustomAutoencoder(len(data[0]), 
                                num_hid1=128,
                                num_hid2=64,
                                reduce_dim=32, 
                                normalize=False)
autoencoder.set_data(data)
autoencoder.set_test_data(test_data)
autoencoder.train(testiter=20)
