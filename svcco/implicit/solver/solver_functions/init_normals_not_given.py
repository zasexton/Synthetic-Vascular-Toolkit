import numpy as np

from ...core.h_matrix import H
from ...core.utils import norm

def init_normals_not_given(lam,K00,K01,K11):
    H_matrix = H(K00,K01,K11,lam)
    n = K00.shape[0]
    d = K11.shape[0]//n
    w,v = np.linalg.eig(H_matrix)
    rw = w.real
    eh = np.argmin(rw)
    g0 = np.zeros((n,d))
    for i in range(n):
        #for j in range(d):
        #    g0[i,j] = np.real(v[eh,i+n*j])
        #mag = np.linalg.norm(g0[i,:])
        #for j in range(d):
        #    g0[i,j] = g0[i,j]/mag
        g0[i,0] = np.real(v[eh,i])
        g0[i,1] = np.real(v[eh,i+n])
        g0[i,2] = np.real(v[eh,i+n*2])
        norms = norm(g0[i,:])
        g0[i,0] = g0[i,0]/norms
        g0[i,1] = g0[i,1]/norms
        g0[i,2] = g0[i,2]/norms
    init_norms = np.zeros(n*(d-1))
    #for i in range(n):
    #    for j in range(d-1):
    #        if j == d - 1:
    #            if ic_norms[i,j] >= 0:
    #                init_norms[i*(d-1)+j] = np.arccos(g0[i,j]/np.linalg.norm(g0[i,j:]))
    #            else:
    #                init_norms[i*(d-1)+j] = 2*np.pi - np.arccos(g0[i,j]/np.linalg.norm(g0[i,j:]))
    #        else:
    #            init_norms[i*(d-1)+j] = np.arccos(g0[i,j]/np.linalg.norm(g0[i,j:]))
    for i in range(K00.shape[0]):
        init_norms[i*2] = np.arctan2((g0[i,0]**2+g0[i,1]**2)**(1/2),g0[i,2])
        init_norms[i*2+1] = np.arctan2(g0[i,1],g0[i,0])
    return init_norms
