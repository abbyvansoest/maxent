import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA 
from sklearn.model_selection import GridSearchCV

import time

class RewardFn:

    def __init__(self, data, eps=.001):
        self.eps = eps
        self.kde = None

        if data is not None:
            data = np.array(data)
            self.pca = PCA(n_components=32, whiten=False)
            data = self.pca.fit_transform(data)
            self.kde = self.fit_distribution(data)

    def fit_distribution(self, data):
        print(data.shape)

        # # use grid search cross-validation to optimize the bandwidth
        # params = {'bandwidth': np.logspace(-2, 0, 2)}
        # grid = GridSearchCV(KernelDensity(kernel='epanechnikov'), params, cv=5)
        # grid.fit(data)

        # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

        # # use the best estimator to compute the kernel density estimate
        # return grid.best_estimator_
        
        kde = KernelDensity(bandwidth=.4, kernel='epanechnikov').fit(data)
        return kde

    def get_prob(self, x):
        x = self.pca.transform(x)
        # print(self.kde.score(x))
        # print(np.exp(self.kde.score(x)))
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
            # print(prob)
            # print('----------')

