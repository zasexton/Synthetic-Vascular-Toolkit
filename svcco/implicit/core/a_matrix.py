import numpy as np
import numba as nb
from .m_matrix import M
from .n_matrix import N

@nb.jit(nopython=True)
def A(surface):
    n = surface.shape[0]
    d = surface.shape[1]
    M_matrix = M(surface)
    N_matrix = N(surface)
    A_matrix = np.zeros(((n+1)*4,(n+1)*4))
    A_matrix[:n*4,:n*4] = M_matrix
    A_matrix[:n*4,n*4:] = N_matrix
    A_matrix[n*4:,:n*4] = N_matrix.T
    A_inv = np.linalg.inv(A_matrix)
    M_inv = A_inv[:n*4,:n*4]
    N_inv = A_inv[:n*4,n*4:]
    K00 = np.zeros((n,n))
    K01 = np.zeros((n,n*d))
    K11 = np.zeros((n*d,n*d))
    K00[:,:] = M_inv[:n,:n]
    K01[:,:] = M_inv[:n,n:]
    K11[:,:] = M_inv[n:,n:]
    return A_inv,K00,K01,K11
