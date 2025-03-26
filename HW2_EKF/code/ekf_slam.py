'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)

def block_from_diag(K):
    return np.block([[K, np.zeros((2, 10))], 
                    [np.zeros((2, 2)),K, np.zeros((2, 8))], 
                    [np.zeros((2, 4)), K, np.zeros((2, 6))], 
                    [np.zeros((2, 6)), K, np.zeros((2, 4))], 
                    [np.zeros((2, 8)), K, np.zeros((2, 2))], 
                    [np.zeros((2, 10)), K ]])
def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    
    angle_rad =  ((angle_rad + np.pi) % (2 * np.pi)) - np.pi
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2
    #From looking at data.txt, we have k = 6 landmarks. 
    landmark = np.zeros((2 * k, 1))
    measures = init_measure.reshape((6, 2))
    B = measures[:, 0]
    r = measures[:, 1]
    B_err = init_measure_cov[0, 0]
    r_err = init_measure_cov[1, 1]
    #landmark_cov = np.zeros((2 * k, 2 * k))
    
    x,y, theta = init_pose[0], init_pose[1], init_pose[2]
    #x_sig, y_sig, theta_sig = np.sqrt(init_pose_cov[0,0]), np.sqrt(init_pose_cov[1,1]), np.sqrt(init_pose_cov[2,2])
    #x_new = x + np.random.normal(0, x_sig)
    #y_new = y + np.random.normal(0, y_sig)
    #theta_new = theta + np.random.normal(0, theta_sig)

    for i in range(k):
        #print(np.array([[x_new+(r[i]+r_err)*np.sin(warp2pi(B[i]+B_err))], [y_new+(r[i]+r_err)*np.cos(warp2pi(B[i]+B_err))]]).reshape((-1,1)).shape)
        #landmark[2*i:2*(i+1)] = np.array([[x_new+(r[i]+r_err)*np.sin(warp2pi(B[i]+B_err))], [y_new+(r[i]+r_err)*np.cos(warp2pi(B[i]+B_err))]]).reshape((-1,1))
        landmark[2*i:2*(i+1)] = np.array([[x+(r[i])*np.cos((B[i]+theta))], [y+(r[i]+r_err)*np.sin((B[i]+theta))]]).reshape((-1,1))
    landmark_cov = block_from_diag(init_measure_cov)
    return k, landmark, landmark_cov
    


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).4
    '''
    x, y, theta= (X[0][0]), (X[1][0]),(X[2][0])
    
    R = np.block([[control_cov, np.zeros((3, 2*k)).astype(float)], 
                [np.zeros((2*k, 3+2*k)).astype(float)]])
    d, alpha = (control[0][0]), (control[1][0])
  
    

    
    G = np.zeros((3+2*k, 3+2*k))
    G[0:3, 0:3] = np.array([[1.0, 0.0, -(d*np.sin(theta))], 
                            [0.0, 1.0, (d*np.cos(theta))], 
                            [0.0, 0.0, 1.0]])
    G[3:3+2*k, 3:3+2*k] = np.eye(2*k)
    F = np.zeros((3+2*k, 3+2*k))
    F[0:3, 0:3] = np.array([[np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta), np.cos(theta), 0], 
                            [0.0, 0.0, 1.0]])
    x_new = x+d*np.cos(theta)
    y_new= y+d*np.sin(theta)
    theta_new = theta + alpha
    X_new = np.array([x_new, y_new, theta_new]).reshape((-1,1))

    X_pre = np.vstack((X_new, X[3:3+2*k, :]))
    P_pre = G@P@G.T + F@R@F.T

    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    Ht = np.zeros((12, 15))
    h = np.zeros((12,1))
    x, y, theta= (X_pre[0][0]), float(X_pre[1][0]),float(X_pre[2][0])
    measures = measure.reshape((6, 2))
    B  = measures[:, 0]
    r = measures[:, 1]
    B_err = measure_cov[0, 0]
    r_err = measure_cov[1, 1]
    Q = block_from_diag(measure_cov)
    #Jacobian Hp
    Landmarks = X_pre[3:3+2*k]
    for i in range(k):
        l_x = (Landmarks[2*i][0])
        l_y = (Landmarks[2*i+1][0]) 
        
        Hp  = np.array([[(l_y - y)/(np.square(l_x-x)+np.square(l_y-y)), -(l_x - x)/(np.square(l_x-x)+np.square(l_y-y)), -1], 
                        [-(l_x-x)/np.sqrt((np.square(l_x-x)+np.square(l_y-y))), -(l_y-y)/np.sqrt((np.square(l_x-x)+np.square(l_y-y))), 0]])
        Hl = np.array([[-(l_y - y)/(np.square(l_x-x)+np.square(l_y-y)), (l_x - x)/(np.square(l_x-x)+np.square(l_y-y))],
                       [(l_x-x)/np.sqrt((np.square(l_x-x)+np.square(l_y-y))), (l_y-y)/np.sqrt((np.square(l_x-x)+np.square(l_y-y)))]])
        Ht[2*i:2*(i+1), 0:3] = Hp
        Ht[2*i:2*(i+1), 3+2*i:3+2*(i+1)]=  Hl
        h[2*i:2*(i+1), :] = np.array([[warp2pi(np.arctan2(l_y-y, l_x-x) - theta + B_err)], 
                                   [np.sqrt((np.square(l_x-x)+np.square(l_y-y)))+r_err]])
    #Kalman Gain
    K = P_pre@Ht.T@np.linalg.inv(Ht@P_pre@Ht.T + Q)

    #New mean of X
    X_pre_new = X_pre + K@(measure - h)
    #New covariance of X
    P_pre_new = (np.eye(15) - K@Ht)@P_pre
    #return X_pre, P_pre
    return X_pre_new, P_pre_new


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    landmarks = X[3:3+2*k]
    cov_landmarks = P[3:3+2*k, 3:3+2*k]
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float).reshape((-1,1))
    err = l_true - landmarks
    Eucledian_dist = []#err.T@err
    Mahalanobis_dist = []#err.T@np.linalg.inv(cov_landmarks)@err
    for i in range(k):
        e = err[2*i:2*(i+1)]
        eucledian = e.T@e
        Eucledian_dist.append(eucledian)
        cov_step = cov_landmarks[2*i:2*(i+1),2*i:2*(i+1) ]
        mahalanobis = e.T@np.linalg.inv(cov_step)@e
        Mahalanobis_dist.append(mahalanobis)
    
    Eucledian_dist = np.array(Eucledian_dist).reshape((-1,1))
    Mahalanobis_dist = np.array(Mahalanobis_dist).reshape((-1,1))
    np.save("Eucledian_Dist", Eucledian_dist)
    np.save("Mahalanobis_Dist", Mahalanobis_dist)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.savefig("FinalPlot.png")
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01;
    sig_r = 0.08    ;


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])
    np.save("P_initial", P)
    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    np.save("Pfinal", P)
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
