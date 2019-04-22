# The RewardFn object determines the reward the ApproxPlan implementation awards
# to a given state if the MaxEnt implmentation uses a density estimation kernel. 
# (i.e currently only Humanoid)

import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA 
from sklearn.model_selection import GridSearchCV

import time

class RewardFn:

    def __init__(self, data, n_components=None, eps=.001):
        self.eps = eps
        self.kde = None
        self.pca = None

        if data is not None:
            data = np.array(data)

            if n_components is not None:
                self.pca = PCA(n_components=n_components, whiten=False)
                data = self.pca.fit_transform(data)

            self.kde = self.fit_distribution(data)

    def fit_distribution(self, data):
        kde = KernelDensity(bandwidth=.1, kernel='epanechnikov').fit(data)
        return kde

    def get_prob(self, x):
        if self.pca is not None:
            x = self.pca.transform(x)
        return np.exp(self.kde.score(x))

    def reward(self, x):
        if self.kde is None:
            return 0.
        
        prob_x = self.get_prob(x)
        return 1/(prob_x + self.eps)

    def test(self, x, env):
        for data in x:
            continue
            prob = self.get_prob([data])

