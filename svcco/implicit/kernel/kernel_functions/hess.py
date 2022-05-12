import numpy as np
import numba as nb

@nb.jit(nopython=True,parallel=True,fastmath=True,nogil=True)
def hess(x,H_0,n=0,dtype='float64'):
    n = H_0.shape[0]//3
    h = np.zeros((2*n,2*n),dtype=dtype)
    for i in nb.prange(n):
        for j in nb.prange(i,n):
            for k in nb.prange(n):
                if i==j:
                    # d^2/dti^2
                    h[i*2,j*2] += (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k]-np.cos(x[i*2])*H_0[i+2*n,k])*\
                                  (np.sin(x[k*2])*np.cos(x[k*2+1]))+\
                                  (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k+n]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k+n]-np.cos(x[i*2])*H_0[i+2*n,k+n])*\
                                  (np.sin(x[k*2+1])*np.sin(x[k*2]))+\
                                  (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k+2*n]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k+2*n]-np.cos(x[i*2])*H_0[i+2*n,k+2*n])*\
                                  (np.cos(x[k*2]))-\
                                  (np.sin(x[k*2+1])*np.sin(x[k*2])*H_0[k+n,i]+np.sin(x[k*2])*np.cos(x[k*2+1])*H_0[k,i]+np.cos(x[k*2])*H_0[k+2*n,i])*\
                                  (np.sin(x[i*2])*np.cos(x[i*2+1]))-\
                                  (np.sin(x[k*2+1])*np.sin(x[k*2])*H_0[k+n,i+n]+np.sin(x[k*2])*np.cos(x[k*2+1])*H_0[k,i+n]+np.cos(x[k*2])*H_0[k+2*n,i+n])*\
                                  (np.sin(x[i*2+1])*np.sin(x[i*2]))-\
                                  (np.sin(x[k*2+1])*np.sin(x[k*2])*H_0[k+n,i+2*n]+np.sin(x[k*2])*np.cos(x[k*2+1])*H_0[k,i+2*n]+np.cos(x[k*2])*H_0[k+2*n,i+2*n])*\
                                  (np.cos(x[i*2]))
                    if k==0:
                        h[i*2,j*2] += 2*(np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,i]-np.sin(x[i*2])*H_0[i+2*n,i]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,i])*\
                                      (np.cos(x[i*2+1])*np.cos(x[i*2]))+\
                                      2*(np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,i+n]-np.sin(x[i*2])*H_0[i+2*n,i+n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,i+n])*\
                                      (np.sin(x[i*2+1])*np.cos(x[i*2]))-\
                                      2*(np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,i+2*n]-np.sin(x[i*2])*H_0[i+2*n,i+2*n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,i+2*n])*\
                                      (np.sin(x[i*2]))
                    # d^2f/dpi^2
                    h[i*2+1,j*2+1] += (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k])*\
                                      (np.sin(x[2*k])*np.cos(x[2*k+1]))+\
                                      (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k+n]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k+n])*\
                                      (np.sin(x[k*2])*np.sin(x[k*2+1]))+\
                                      (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i+n,k+2*n]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i,k+2*n])*\
                                      (np.cos(x[k*2]))+\
                                      (-np.sin(x[2*k+1])*np.sin(x[2*k])*H_0[k+n,i]-np.sin(x[2*k])*np.cos(x[2*k+1])*H_0[k,i]-np.cos(x[2*k])*H_0[k+2*n,i])*\
                                      (np.sin(x[2*i])*np.cos(x[2*i+1]))+\
                                      (-np.sin(x[2*k+1])*np.sin(x[2*k])*H_0[k+n,i+n]-np.sin(x[2*k])*np.cos(x[2*k+1])*H_0[k,i+n]-np.cos(x[2*k])*H_0[k+2*n,i+n])*\
                                      (np.sin(x[i*2+1])*np.sin(x[i*2]))
                    if k==0:
                        h[i*2+1,j*2+1] += 2*(np.sin(x[2*i+1])*np.sin(x[2*i])*H_0[i,i]-np.sin(x[i*2])*np.cos(x[2*i+1])*H_0[i+n,i])*\
                                          (np.sin(x[i*2+1])*np.sin(x[i*2]))+\
                                          2*(-np.sin(x[2*i+1])*np.sin(x[2*i])*H_0[i,i+n]+np.sin(x[i*2])*np.cos(x[2*i+1])*H_0[i+n,i+n])*\
                                          (np.cos(x[i*2+1])*np.sin(x[i*2]))
                    # d^2f/dtidpi
                    h[i*2,j*2+1] += (-np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i,k]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,k])*\
                                    (np.sin(x[k*2])*np.cos(x[k*2+1]))+\
                                    (-np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i,k+n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,k+n])*\
                                    (np.sin(x[k*2+1])*np.sin(x[k*2]))+\
                                    (-np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i,k+2*n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,k+2*n])*\
                                    (np.cos(x[k*2]))-\
                                    (np.sin(x[k*2+1])*np.sin(x[k*2])*H_0[k+n,i]+np.sin(x[k*2])*np.cos(x[k*2+1])*H_0[k,i]+np.cos(x[k*2])*H_0[k+2*n,i])*\
                                    (np.sin(x[i*2+1])*np.cos(x[i*2]))+\
                                    (np.sin(x[k*2+1])*np.sin(x[k*2])*H_0[k+n,i+n]+np.sin(x[k*2])*np.cos(x[k*2+1])*H_0[k,i+n]+np.cos(x[k*2])*H_0[k+2*n,i+n])*\
                                    (np.cos(x[i*2+1])*np.cos(x[i*2]))
                    if k==0:
                        h[i*2,j*2+1] += (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,i]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,i])*\
                                        (np.cos(x[i*2+1])*np.cos(x[i*2]))+\
                                        (-np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,i+n]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,i+n])*\
                                        (np.sin(x[i*2+1])*np.cos(x[i*2]))+\
                                        (np.sin(x[i*2+1])*np.sin(x[i*2])*H_0[i,i+2*n]-np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,i+2*n])*\
                                        (np.sin(x[i*2]))-\
                                        (np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,i]-np.sin(x[i*2])*H_0[i+2*n,i]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,i])*\
                                        (np.sin(x[i*2+1])*np.sin(x[i*2]))+\
                                        (np.sin(x[i*2+1])*np.cos(x[i*2])*H_0[i+n,i+n]-np.sin(x[i*2])*H_0[i+2*n,i+n]+np.cos(x[i*2+1])*np.cos(x[i*2])*H_0[i,i+n])*\
                                        (np.sin(x[i*2])*np.cos(x[i*2+1]))
                elif j > i:
                    if k == 0:
                        # d^2f/dtidtk
                        h[i*2,j*2] += (np.sin(x[2*i+1])*np.cos(x[2*i])*H_0[i+n,j]-np.sin(x[2*i])*H_0[i+2*n,j]+np.cos(x[2*i+1])*np.cos(x[2*i])*H_0[i,j])*\
                                      (np.cos(x[2*j+1])*np.cos(x[2*j]))+\
                                      (np.sin(x[2*i+1])*np.cos(x[2*i])*H_0[i+n,j+n]-np.sin(x[2*i])*H_0[i+2*n,j+n]+np.cos(x[2*i+1])*np.cos(x[2*i])*H_0[i,j+n])*\
                                      (np.sin(x[2*j+1])*np.cos(x[2*j]))-\
                                      (np.sin(x[2*i+1])*np.cos(x[2*i])*H_0[i+n,j+2*n]-np.sin(x[2*i])*H_0[i+2*n,j+2*n]+np.cos(x[2*i+1])*np.cos(x[2*i])*H_0[i,j+2*n])*\
                                      (np.sin(x[2*j]))+\
                                      (np.sin(x[2*j+1])*np.cos(x[2*j])*H_0[j+n,i]-np.sin(x[2*j])*H_0[j+2*n,i]+np.cos(x[2*j+1])*np.cos(x[2*j])*H_0[j,i])*\
                                      (np.cos(x[2*i+1])*np.cos(x[2*i]))+\
                                      (np.sin(x[2*j+1])*np.cos(x[2*j])*H_0[j+n,i+n]-np.sin(x[2*j])*H_0[j+2*n,i+n]+np.cos(x[2*j+1])*np.cos(x[2*j])*H_0[j,i+n])*\
                                      (np.sin(x[2*i+1])*np.cos(x[i*2]))-\
                                      (np.sin(x[2*j+1])*np.cos(x[2*j])*H_0[j+n,i+2*n]-np.sin(x[2*j])*H_0[j+2*n,i+2*n]+np.cos(x[2*j+1])*np.cos(x[2*j])*H_0[j,i+2*n])*\
                                      (np.sin(x[2*i]))
                        # d^2f/dtidpk
                        h[i*2,j*2+1] += (-np.sin(x[2*j+1])*np.sin(x[2*j])*H_0[j,i]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j+n,i])*\
                                        (np.cos(x[2*i+1])*np.cos(x[2*i]))+\
                                        (-np.sin(x[2*j+1])*np.sin(x[2*j])*H_0[j,i+n]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j+n,i+n])*\
                                        (np.sin(x[2*i+1])*np.cos(x[2*i]))-\
                                        (-np.sin(x[2*j+1])*np.sin(x[2*j])*H_0[j,i+2*n]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j+n,i+2*n])*\
                                        (np.sin(x[2*i]))-\
                                        (np.sin(x[2*i+1])*np.cos(x[2*i])*H_0[i+n,j]-np.sin(x[2*i])*H_0[i+2*n,j]+np.cos(x[2*i+1])*np.cos(x[2*i])*H_0[i,j])*\
                                        (np.sin(x[2*j+1])*np.sin(x[2*j]))+\
                                        (np.sin(x[2*i+1])*np.cos(x[2*i])*H_0[i+n,j+n]-np.sin(x[2*i])*H_0[i+2*n,j+n]+np.cos(x[2*i+1])*np.cos(x[2*i])*H_0[i,j+n])*\
                                        (np.sin(x[2*j])*np.cos(x[2*j+1]))
                        # d^2f/dpidtk
                        h[i*2+1,j*2] += (-np.sin(x[2*i+1])*np.sin(x[2*i])*H_0[i,j]+np.sin(x[2*i])*np.cos(x[2*i+1])*H_0[i+n,j])*\
                                        (np.cos(x[2*j+1])*np.cos(x[2*j]))+\
                                        (-np.sin(x[2*i+1])*np.sin(x[2*i])*H_0[i,j+n]+np.sin(x[2*i])*np.cos(x[2*i+1])*H_0[i+n,j+n])*\
                                        (np.sin(x[2*j+1])*np.cos(x[2*j]))-\
                                        (-np.sin(x[2*i+1])*np.sin(x[2*i])*H_0[i,j+2*n]+np.sin(x[2*i])*np.cos(x[2*i+1])*H_0[i+n,j+2*n])*\
                                        (np.sin(x[j*2]))+\
                                        (-np.sin(x[2*j+1])*np.cos(x[2*j])*H_0[j+n,i]+np.sin(x[j*2])*H_0[j+2*n,i]-np.cos(x[j*2])*np.cos(x[j*2+1])*H_0[j,i])*\
                                        (np.sin(x[i*2+1])*np.sin(x[i*2]))+\
                                        (np.sin(x[2*j+1])*np.cos(x[2*j])*H_0[j+n,i+n]-np.sin(x[j*2])*H_0[j+2*n,i+n]+np.cos(x[j*2])*np.cos(x[j*2+1])*H_0[j,i+n])*\
                                        (np.sin(x[2*i])*np.cos(x[2*i+1]))
                        # d^2f/dpidpk
                        h[i*2+1,j*2+1] += -(-np.sin(x[2*i+1])*np.sin(x[i*2])*H_0[i,j]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,j])*\
                                          (np.sin(x[j*2+1])*np.sin(x[j*2]))+\
                                          (-np.sin(x[2*i+1])*np.sin(x[i*2])*H_0[i,j+n]+np.sin(x[i*2])*np.cos(x[i*2+1])*H_0[i+n,j+n])*\
                                          (np.sin(x[2*j])*np.cos(x[2*j+1]))+\
                                          (np.sin(x[2*j+1])*np.sin(x[j*2])*H_0[j,i]-np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j+n,i])*\
                                          (np.sin(x[2*i+1])*np.sin(x[i*2]))+\
                                          (-np.sin(x[2*j+1])*np.sin(x[j*2])*H_0[j,i+n]+np.sin(x[j*2])*np.cos(x[j*2+1])*H_0[j+n,i+n])*\
                                          (np.sin(x[i*2])*np.cos(x[i*2+1]))
                    else:
                        break
    l_tri = h.T.flatten()
    l_tri = l_tri[l_tri != 0]
    return [l_tri]
