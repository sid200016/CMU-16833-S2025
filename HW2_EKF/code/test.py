import numpy as np
def wrap_to_pi( angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

a = wrap_to_pi(5/6*np.pi)
print(a/np.pi)
def diag_from_block(K):
    return np.block([[K, np.zeros((2, 10))], 
                    [np.zeros((2, 2)),K, np.zeros((2, 8))], 
                    [np.zeros((2, 4)), K, np.zeros((2, 6))], 
                    [np.zeros((2, 6)), K, np.zeros((2, 4))], 
                    [np.zeros((2, 8)), K, np.zeros((2, 2))], 
                    [np.zeros((2, 10)), K ]])
K = np.array([[1, 0], [0, 1]])
K_2 = diag_from_block(K)
print(K_2.shape)
print(np.random.normal(0, 0.5))
A = np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12])).reshape((-1,1))
print(A)
print(A.reshape((6, 2)))