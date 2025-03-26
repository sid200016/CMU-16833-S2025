'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 2.95
        self._z_short = 0.05
        self._z_max = 0.0015
        self._z_rand =5  

        self._sigma_hit = 70  
        self._lambda_short = 20

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 800
        self._map_resolution = 10
        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in skipping angles in ray casting
        self._subsampling = 1
        self.map = occupancy_map
    #Raycasting
    def raycast(self, map, x,y, theta, theta_robot):
        x = np.squeeze(x)
        y = np.squeeze(y)
        theta_robot = np.squeeze(theta_robot)
        grid_max_range = int(self._max_range//10)
        theta_transform = (theta_robot - np.pi/2)
        # angle = self.wrap_to_pi(theta_transform[:, np.newaxis] + theta[np.newaxis, :])
        angle = (theta_transform[:, np.newaxis] + theta[np.newaxis, :])
        #angle = self.wrap_to_pi(angle)
        #theta_robot = self.wrap_to_pi(theta_robot)
        x_las = x + 25*np.cos(theta_robot)
        y_las = y + 25*np.sin(theta_robot)
        
        #distance = np.empty((0, 1))
        steps=  np.arange(grid_max_range)
        #steps=  np.arange(self._max_range)
        
        
        x_new = x_las[:, np.newaxis, np.newaxis] + np.cos(angle)[:, :, np.newaxis]*steps[np.newaxis, np.newaxis, :]
        y_new = y_las[:, np.newaxis, np.newaxis] + np.sin(angle)[:, :, np.newaxis]*steps[np.newaxis, np.newaxis, :]
        
        x_ray = np.round(x_new).astype(int)
        y_ray = np.round(y_new).astype(int)
        
       
        mask = ((x_ray >= 0) & (y_ray >= 0) & (x_ray < map.shape[1]) & (y_ray < map.shape[0]))
        valid_x = np.where(mask, x_ray, 0)
        valid_y = np.where(mask, y_ray, 0)
        detection = np.zeros_like(x_ray, dtype = bool)
        detection[mask] = map[valid_y[mask], valid_x[mask]] >= self._min_probability

        has_detected = detection.any(axis = 2)

        first_hit = np.argmax(detection, axis  = 2)
        
        distances = np.where(has_detected, first_hit, grid_max_range)
        #distances = np.where(has_detected, first_hit,self._max_range)
    
        
        return distances

        # for i in range(0, grid_max_range):

        #     x_new = (np.round(x_las+np.cos(theta)*i)).astype(int)
        #     y_new = (np.round(y_las+np.sin(theta)*i)).astype(int)
        #     #Ray casted out of map again
        #     if y_new >= map.shape[0] or x_new >= map.shape[1]:
                
        #         return grid_max_range
        #     #Ray casted out of map
        #     elif y_new < 0 or x_new < 0:
        #         return grid_max_range
        #     #Obstacle detected
        #     elif map[y_new, x_new]  >= self._min_probability:
        #         return i
        # return grid_max_range
    #Gaussian
    def gaussian(self, z, mu, sigma):
        return np.exp(-np.square(z - mu)/(2*np.square(sigma)))/(np.sqrt(2*np.pi*np.square(sigma)))
    #Exponential
    def exponential(self, z, lambda_short):
        
        return lambda_short*np.exp(-lambda_short*z)
    
          
    def gaus_cdf(self, upper, mu, sigma):
        return norm.cdf(upper, mu, sigma) - norm.cdf(0, mu, sigma)
    
    def exp_cdf(self, zstar, lambda_short):
        if 1-np.exp(-lambda_short*zstar) ==  0:
            return 1e-6
        return 1/(1-np.exp(-lambda_short*zstar))
    
    def plot_rays(self, distances, x, y, theta_robot, z):
        
        x_las = x + 25*np.cos(theta_robot)
        y_las = y + 25*np.sin(theta_robot)
        theta_transform = (theta_robot - np.pi/2)
        
        #distances = np.array(distances)
        #distances = distances.reshape((-1, 2))
        plt.ion()
        fig = plt.figure()
        plt.imshow(self.map, cmap='Greys')
        angles = np.arange(0, 180, self._subsampling)*np.pi/180
        #angles = self.wrap_to_pi(angles + theta_robot)
        # angles = self.wrap_to_pi(angles + theta_transform)
        angles = (angles + theta_transform)
        #for end_point in distances:
        
        for i in range(len(distances)):
        
            #Draw line from laser to z (from log file) endpoint
            dist_endpoint = [x_las + z[i]*np.cos(angles[i]), y_las + z[i]*np.sin(angles[i])]
            # Draw line from laser position to endpoint of raycasst
            end_point = [x_las + distances[i]*np.cos(angles[i]),y_las + distances[i]*np.sin(angles[i])]
            robot_x2 = x + (np.cos(theta_robot))*50
            robot_y2 = y + (np.sin(theta_robot))*50
            #Plot robot heading
            plt.plot([x, robot_x2], [y, robot_y2], 'b-', alpha = 0.5)
            #plot endpoints
            plt.plot([x_las, dist_endpoint[0]], [y_las, dist_endpoint[1]], 'g-', alpha = 1)
            plt.plot([x_las, end_point[0]], [y_las, end_point[1]], 'r-', alpha=0.5)
            
        plt.plot(x_las, y_las, 'bo')  # Blue dot for laser position
        plt.plot(x, y, 'yo')  # yellow dot for robot position
        plt.grid(True)
        plt.draw()
        plt.pause(3)
        plt.savefig('ray_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    def wrap_to_pi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def beam_range_finder_model(self, z_t1_arr, x_t1, plot_true):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        
        TODO : Add your code here
        
        """
        #print(z_t1_arr)
        
        #src_path_map = '../data/map/wean.dat'
        grid_max_range = int(self._max_range//10)
        z_t1_arr = z_t1_arr // 10
        map_occupancy = self.map
        #print(map_occupancy.shape)
        x = x_t1[:, 0]/10
        y = x_t1[:, 1]/10
        x = x.reshape((-1,1))
        y= y.reshape((-1,1))
        #theta_world = self.wrap_to_pi(x_t1[:, 2])
        theta_world = (x_t1[:, 2])
        theta_world = theta_world.reshape(-1,1)
        max_iter = len(z_t1_arr) 
       
        # For plot
        
        

        ## Vectorized
        angles = np.arange(0, 180, self._subsampling)*np.pi/180
        #angles = np.arange(-90, 90, self._subsampling)*np.pi/180
        distances = self.raycast(map_occupancy, x, y, angles, theta_world)
        #self.plot_rays(distances, x, y, theta_world)
        #thetas = self.wrap_to_pi(thetas)
        #distances = self.raycast(self.map, x[angles], y[angles], angles, theta_world[angles])
        prob_z_t1 = []
        for row in range(distances.shape[0]):
            prob = 1.0
             
            for col in range(distances.shape[1]):
                z = z_t1_arr[col]
                z_star = distances[row, col]
                if 0 <= z <= grid_max_range:
                #if 0 <= z <= self._max_range:
                    p_hit = self.gaussian(z, z_star, self._sigma_hit)
                    #normalizer_gaus = 1/self.gaus_cdf(grid_max_range, z_star, self._sigma_hit)
                    #p_hit = p_hit*normalizer_gaus
                else:
                    p_hit = 0
                if 0 <= z <= z_star:
                    p_short = self.exponential(z, self._lambda_short )
                    #normalizer_exp = self.exp_cdf(z_star, self._lambda_short)
                    #p_short = p_short * normalizer_exp
                else:
                    p_short = 0
                if z >= grid_max_range:
                #if z >= self._max_range:
                    p_max = 1
                else:
                    p_max = 0
                if 0 <= z < grid_max_range:
                #if 0 <= z < self._max_range:
                    p_rand = 1/grid_max_range
                    #p_rand = 1/self._max_range
                    
                else:
                    p_rand = 0
                p_current_step = (p_hit*self._z_hit+p_short*self._z_short+p_max*self._z_max+p_rand*self._z_rand)
                #p_current_step /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
                prob *= p_current_step
            if plot_true:
                self.plot_rays(distances[row], x[row], y[row], theta_world[row], z_t1_arr)
            prob_z_t1.append(prob)
            
        prob_z_t1 = np.array(prob_z_t1).reshape(-1,1)
        #print("prob", prob_z_t1.shape)
        np.save("probs", prob_z_t1)
        return prob_z_t1

        ##
        for j in range(0, len(x)):
            for i in range(0, max_iter, self._subsampling):
            #Angles in radians
                x_las = x_laser[j]
                y_las = y_laser[j] 
                theta = i * np.pi/180 + theta_world[i]
                theta = self.wrap_to_pi(theta)
                z = z_t1_arr[i]
            
                distance = self.raycast(map_occupancy, x[j], y[j], theta, theta_world[j]) #z_start
                distances.append(x_las + distance*np.cos(theta))
                distances.append(y_las + distance*np.sin(theta))
            #print(theta)
            # x_plot = [x + 0.25*np.cos(theta_world), distance*np.cos(theta)]
            # y_plot = [y + 0.25*np.sin(theta_world), distance*np.sin(theta)]
            # # print("xplot", x_plot)
            # # print("yplot", y_plot)``
            # plt.plot(x_plot, y_plot,'bo')

            
            ##Condition switch around
                if 0 <= z <= self._max_range:
                #p_hit
                    p_hit = self.gaussian(z, distance, self._sigma_hit)
                    normalizer_gaus = 1/self.gaus_cdf(self._max_range, distance, self._sigma_hit)
                    p_hit = p_hit*normalizer_gaus
                    
                

                else:
                    p_hit = 0
                    
                    
                if 0<=z<=distance:
                    #p_short
                    p_short = self.exponential(z, self._lambda_short, distance)
                    normalizer_short = self.exp_cdf(distance, self._lambda_short)
                    p_short = p_short*normalizer_short
                
                else:
                    p_short = 0
            
                if distance >= self._max_range/self._map_resolution:
                    p_max = 1.0
                else:
                    p_max = 0.0
                
                if 0 < z < self._max_range/self._map_resolution:
                    p_rand = 1/self._max_range/self._map_resolution
                else:
                    p_rand = 0
                
            
                
                #p_current_step = np.log(p_hit*self._z_hit+p_short*self._z_short+p_max*self._z_max+p_rand*self._z_rand)
                p_current_step = (p_hit*self._z_hit+p_short*self._z_short+p_max*self._z_max+p_rand*self._z_rand)
                p_current_step /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
                prob_zt1 = prob_zt1*p_current_step
            probs.append(prob_zt1)
            


            self.plot_rays(distances, x[j], y[j], theta_world[j])
            distances = []
        probs = np.array(probs)
        np.save('probability', probs)
        prob_zt1 = np.array(probs).reshape(-1,1)
        return prob_zt1
