import numpy as np
import numba as nb
from .utils import norm

@nb.jit(nopython=True)
def M(surface):
    N = surface.shape[0]
    D = surface.shape[1]
    M_matrix = np.zeros((N*4,N*4))
    for i in range(N):
        for j in range(i,N):
            M_matrix[i,j] = norm(surface[i,:] - surface[j,:])**3
            M_matrix[j,i] = norm(surface[i,:] - surface[j,:])**3

    for i in range(N):
        for j in range(N):
            for k in range(D):
               M_matrix[i,N+j+k*N] = 3*norm(surface[i,:] - surface[j,:])*\
                                           (surface[i,k] - surface[j,k])
               M_matrix[N+j+k*N,i] = 3*norm(surface[i,:] - surface[j,:])*\
                                           (surface[i,k] - surface[j,k])

    for i in range(N):
        for j in range(i,N):
            diff = surface[i,:] - surface[j,:]
            for k in range(D):
                for l in range(D):
                    if i==j:
                        M_matrix[N+j+l*N,N+i+k*N] = 0
                        M_matrix[N+i+k*N,N+j+l*N] = 0
                    elif k==l:
                        M_matrix[N+j+l*N,N+i+k*N] = -(3*(diff[l])**2/norm(surface[i,:] - surface[j,:]) +\
                                                    3*norm(surface[i,:] - surface[j,:]))
                        M_matrix[N+i+k*N,N+j+l*N] = -(3*(diff[l])**2/norm(surface[i,:] - surface[j,:]) +\
                                                    3*norm(surface[i,:] - surface[j,:]))
                    else:
                        M_matrix[N+j+l*N,N+i+k*N] = -3*diff[k]*diff[l]/norm(surface[i,:] - surface[j,:])
                        M_matrix[N+i+k*N,N+j+l*N] = -3*diff[k]*diff[l]/norm(surface[i,:] - surface[j,:])
    return M_matrix
