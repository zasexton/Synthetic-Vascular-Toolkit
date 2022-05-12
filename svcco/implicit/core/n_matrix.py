import numpy as np
import numba as nb

@nb.jit(nopython=True)
def N(surface):
    n = surface.shape[0]
    d = surface.shape[1]
    N_matrix = np.zeros((4*n,4))
    for i in range(n):
        N_matrix[i,0] = 1
        for j in range(d):
            N_matrix[i,j+1] = surface[i,j]
    for i in range(n):
        for j in range(d):
            N_matrix[n+i+j*n,j+1] = -1
    return N_matrix
