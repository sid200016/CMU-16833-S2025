'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse import csr_matrix
from scipy.sparse import csc_array
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
import sparseqr
#from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = inv(A.T @ A) @ A.T @ b
    #x = np.zeros((N, ))
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
   
    B = splu(A.T@A, permc_spec='NATURAL')

    x = B.solve(A.T@b)
    
    U = B.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    B = splu(A.T@A, permc_spec='COLAMD')

    x = B.solve(A.T@b)
    
    U = B.U
    return x, U
def solve_custom(A, b):
    N = A.shape[1]
    B = splu(A.T@A, permc_spec='COLAMD')
    L = B.L.toarray()
    U = B.U.toarray()
  
    r = B.perm_r
    c = B.perm_c
    Pr = csc_array((np.ones(N), (B.perm_r, np.arange(N))))
    Pc = csc_array((np.ones(N), (np.arange(N), B.perm_c)))
    z = np.zeros((N, ))
    #Solve Lz = y
    y = Pr.T@A.T@b@Pc.T
    for i in range(N):
        L_current = L[i, i]
        sum = 0
        for j in range(i):
            sum += L[i, j] * z[j]
        z[i] = (y[i] - sum) / L_current
    print(z)
    #Solve Ux = z
    x_perm = np.zeros((N, ))
    for i in range(N-1, -1, -1):
        U_current = U[i, i]
        sum = 0
        for j in range(i+1, N):
            sum += U[i, j] * x_perm[j]
        x_perm[i] = (z[i] - sum) / U_current
    x = np.zeros((N, ))
    #x = Pc @ x_perm
    #x[c] = x_perm
    print(x_perm)

    return Pc@x_perm, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]

    z, R, E, rank = sparseqr.rz(A, b, permc_spec = 'NATURAL')
    
    z = z.flatten()
    R = csr_matrix(R)
    x  =spsolve_triangular(R, z, lower = False)

    
    #R = eye(N)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    z, R, E, rank = sparseqr.rz(A, b, permc_spec = 'COLAMD')
    E = sparseqr.permutation_vector_to_matrix(E)
    R = csr_matrix(R)
    x  =spsolve_triangular(R, z.flatten(), lower = False)
    x = E@x
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'custom': solve_custom,
        'pinv': solve_pinv, 
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)
