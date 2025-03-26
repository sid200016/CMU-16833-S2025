'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        weights = X_bar[:, 3]
        weights = weights / np.sum(weights)
        
        num_particles = X_bar.shape[0] #M
        X_bar_resampled =  np.zeros_like(X_bar)
        r = np.random.uniform(0, 1.0/num_particles) 
        c = weights[0]
        i  = 0
        #u = 0.0
        for m in range(0,  num_particles):
            u = r+(m)/num_particles
            while u > c:
                i = i+1
                c = c + weights[i]
                
            
            X_bar_resampled[m] = X_bar[i]
        #X_bar_resampled = np.array(X_bar_resampled)
        X_bar_resampled[:, 3]= X_bar_resampled[:, 3] / np.sum(X_bar_resampled[:, 3])
        #print(X_bar_resampled.shape)
        return X_bar_resampled
       