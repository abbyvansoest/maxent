import numpy as np
import reacher_utils
import time

class ExperienceBuffer:
    
    def __init__(self):
        self.buffer = []
        self.normalization_factors = []
        self.normalized = False
        self.p = None
        self.p_2d = None
        
    def store(self, state):
        if self.normalized:
            raise ValueError("ERROR! Do not store in buffer after normalizing.")
        # TODO: check dimension of obs: should be dimension of ant env.env.state_vector()
        self.buffer.append(reacher_utils.convert_obs(state))
    
    def reset(self):
        self.buffer = []
        self.normalization_factors = []
        self.normalized = False
        self.p = None
        self.p_2d = None
    
    def normalize(self):
        # for each index in reduced dimension,
        # find the largest value in self.buffer
        # divide all elements at that index 
        for i in range(reacher_utils.expected_state_dim):
            
            # Do not normalize special values
            if i in reacher_utils.special:
                self.normalization_factors.append(1.0)
                continue
             
            i_vals = [x[i] for x in self.buffer]
            max_i_val = max(i_vals)
            self.normalization_factors.append(max_i_val)
            for obs in self.buffer:
                obs[i] = obs[i]/max_i_val
        
        # you don't need to do this again.
        self.normalized = True
        return self.buffer
    
    def get_discrete_distribution(self):
        
        if self.p is not None:
            return self.p
        
        # normalize buffer experience
        if not self.normalized:
            self.normalize()
            
        p = np.zeros(shape=(tuple(reacher_utils.num_states)))
        for obs in self.buffer:
            # discritize obs, add to distribution tabulation.
            p[tuple(reacher_utils.discretize_state(obs))] += 1
        
        p /= len(self.buffer)
        self.p = p
            
        return p
        
    def get_discrete_distribution_2d(self):
        
        if self.p_2d is not None:
            return self.p_2d
        
        # normalized buffer experience
        if not self.normalized:
            self.normalize()
            
        p_2d = np.zeros(shape=(reacher_utils.num_states_2d))
        for obs in self.buffer:
            # discritize obs, add to distribution tabulation.
            p_2d[tuple(reacher_utils.discretize_state_2d(obs))] += 1
        
        p_2d /= len(self.buffer)
        self.p_2d = p_2d
                
        return p_2d
