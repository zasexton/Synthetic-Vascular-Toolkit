import numpy as np
import numba as nb

@nb.jit(nopython=True,fastmath=True)
def fitness(x,H_0,dim=3):
    n = H_0.shape[0]//dim
    a = np.ones(n*dim)
    for i in range(n):
        a[i] = np.cos(x[i*2+1])*np.sin(x[i*2])
        a[i+n] = np.sin(x[i*2+1])*np.sin(x[i*2])
        a[i+2*n] = np.cos(x[i*2])
    return [a.T@H_0@a]
