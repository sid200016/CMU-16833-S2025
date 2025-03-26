'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as npy
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0006
        self._alpha2 = 0.0006
        self._alpha3 = 0.001
        self._alpha4 = 0.001
        
    def wrap_to_pi(self, angle):
        return ((angle + npy.pi) % (2 * npy.pi)) - npy.pi

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        #Calculate rotation steps
        if u_t1[0] == u_t0[0] and u_t1[1] == u_t0[1] and u_t1[2] == u_t0[2]:
            
            x_t1 = x_t0
            return x_t1
        drot1 = (npy.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2])
        dtrans = npy.sqrt(npy.square(u_t1[0] - u_t0[0]) + npy.square(u_t1[1] - u_t0[1]))
        drot2 = (u_t1[2] - u_t0[2] - drot1)
        #Add odometry noise for realism
        #Sqrts
        d_rot1_noise = (drot1 - npy.random.normal(0, ((self._alpha1*drot1**2) + (self._alpha2*dtrans**2))))
        d_trans_noise = dtrans - npy.random.normal(0,((self._alpha3*dtrans**2)+(self._alpha4*(drot1**2+drot2**2))))
        d_rot2_noise = (drot2 - npy.random.normal(0, ((self._alpha1*drot2**2) + (self._alpha2*dtrans**2))))
        # print("d_rot1", d_rot1_noise)
        # print("u_t0", u_t0)
        # print("d_trans", dtrans)
        # print("u_t1", u_t1)
        # print("d_rot2", d_rot2_noise)
        #Prediction step
        x0, y0, theta0 = x_t0[:, 0], x_t0[:, 1], x_t0[:, 2]
        #t = theta0+d_rot1_noise
        #print("theta0 dtype:", t)
        x1 = (x0 + d_trans_noise*npy.cos((theta0+d_rot1_noise))).reshape(-1,1)
        y1 = (y0 + d_trans_noise*npy.sin((theta0+d_rot1_noise))).reshape(-1,1)
        theta1 = (theta0 + d_rot1_noise+d_rot2_noise).reshape(-1,1)

        x_t1 = npy.hstack((x1, y1, theta1))
        
        #print("Difference", npy.linalg.norm(x_t1 - x_t0))
        #return x_t0
        return x_t1

