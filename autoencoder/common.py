import scipy
import numpy as np
import random

log = True # log test data
noisy = False # learn with noisy input
sparse = False # use sparse loss constraint

BATCH_SIZE = 256
TEST_SIZE = 4

learning_rate = 0.01
num_steps = 100000
display_step = 1000

def random_sample(data, size):
    return random.sample(data, size)

# normalize data to the [0,1] range
def normalize_obs(data): 
    # normalize full state.
    for i in range(len(data[0])):
        i_vals = [x[i] for x in data]
        max_i_val = max(i_vals)
        for obs in data:
            obs[i] /= max_i_val
    return data

def log_test(test, pred, printfn=print):
    print("-----")
    if not log:
        return    
    for j in range(len(pred)):
        printfn("eucl: " + str(np.sqrt(((pred[j] - test[j]) ** 2).sum())))
        printfn("mse: " + str(((pred[j] - test[j]) ** 2).mean()))
    printfn("GLOBAL MSE: " + str(((pred - test) ** 2).mean()))