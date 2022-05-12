import numpy as np
import numba as nb

@nb.jit(nopython=True,parallel=True,fastmath=True,nogil=True)
def grad(x,H_0,n=0):
    n = H_0.shape[0]//3 #changing to test shape[0] should be the number of points n
    g = np.zeros(2*n)
    for i in nb.prange(n):
        for j in nb.prange(n):
            g[i*2] += (np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,j]-np.sin(x[i*2])*H_0[i+2*n,j]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,j])*\
                      (np.sin(x[j*2])*np.cos(x[j*2+1]))+\
                      (np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,j+n]-np.sin(x[i*2])*H_0[i+2*n,j+n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,j+n])*\
                      (np.sin(x[j*2+1])*np.sin(x[j*2]))+\
                      (np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,j+n*2]-np.sin(x[i*2])*H_0[i+2*n,j+n*2]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,j+n*2])*\
                      (np.cos(x[j*2]))+\
                      (np.sin(x[j*2+1])*np.sin(x[j*2])*H_0[j+n,i]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j,i]+np.cos(x[j*2])*H_0[j+2*n,i])*\
                      (np.cos(x[i*2+1])*np.cos(x[i*2]))+\
                      (np.sin(x[j*2+1])*np.sin(x[j*2])*H_0[j+n,i+n]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j,i+n]+np.cos(x[j*2])*H_0[j+2*n,i+n])*\
                      (np.sin(x[i*2+1])*np.cos(x[i*2]))-\
                      (np.sin(x[j*2+1])*np.sin(x[j*2])*H_0[j+n,i+n*2]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j,i+n*2]+np.cos(x[j*2])*H_0[j+2*n,i+n*2])*\
                      (np.sin(x[i*2]))
            g[i*2+1] += (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,j]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,j])*(np.sin(x[j*2])*np.cos(x[j*2+1]))+\
                        (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,j+n]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,j+n])*(np.sin(x[j*2])*np.sin(x[j*2+1]))+\
                        (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,j+n*2]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,j+n*2])*(np.cos(x[j*2]))-\
                        (np.sin(x[j*2+1])*np.sin(x[j*2])*H_0[j+n,i]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j,i]+np.cos(x[j*2])*H_0[j+2*n,i])*\
                        (np.sin(x[i*2+1])*np.sin(x[i*2]))+\
                        (np.sin(x[j*2+1])*np.sin(x[j*2])*H_0[j+n,i+n]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j,i+n]+np.cos(x[j*2])*H_0[j+2*n,i+n])*\
                        (np.sin(x[i*2])*np.cos(x[i*2+1]))
    return g


#GENERALIZE HYPERSPHERE GRADIENT

@nb.jit(nopython=True,parallel=True,fastmath=True,nogil=True,cache=True,boundscheck=False)
def nhs_grad(x,H_0):
    pass
