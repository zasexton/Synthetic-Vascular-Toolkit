import numpy as np
import numba as nb

@nb.jit(nopython=True)
def H(K00,K01,K11,lam):
    if lam > 0:
        inv = np.linalg.inv(np.eye(K00.shape[0])+lam*K00)
        H_matrix = K11 - (lam*K01.T)@inv@K01
    else:
        H_matrix = K11
    return H_matrix
