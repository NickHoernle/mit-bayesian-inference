#!/usr/bin/env python

'''
Need a custom multivariate normal class for fast evaluation of differnet normals for one datapoint in the
forward and backward computations.
'''
import numpy as np

class MultivariateNormal:
    def __init__(self, means, covariances):
        self.means = np.array(means)
        self.cov = np.array(covariances)
        self.cov_inv = np.linalg.pinv(self.cov)
        self.const = -0.5*np.log(np.linalg.det(2*np.pi*self.cov))

        mean_shape = self.means.shape
        if len(mean_shape) == 1:
            def logpdf(self, x):
                return self.const - 0.5*np.array((x-self.means).T.dot(self.cov_inv).dot(x-self.means))
        elif (mean_shape[0] == mean_shape[1]) and (len(mean_shape) > 2):
            def logpdf(self, x):
                return self.const - 0.5*np.array([[(x-self.means[i][j]).T.dot(self.cov_inv[i][j]).dot(x-self.means[i][j]) for j,_ in enumerate(row)] for i,row in enumerate(self.means)])
        else:
            def logpdf(self, x):
                return self.const - 0.5*np.array([(x-self.means[i]).T.dot(self.cov_inv[i]).dot(x-self.means[i]) for i,_ in enumerate(self.means)])

        setattr(self.__class__, 'logpdf', logpdf)
