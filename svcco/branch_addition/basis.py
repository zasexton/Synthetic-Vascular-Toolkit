import numba as nb
import numpy as np

@nb.jit(nopython=True,cache=True,nogil=True)
def basis(data, edges):
    data[edges, 12:15] = (data[edges, 3:6] -
                          data[edges, 0:3]) / np.linalg.norm(data[edges, 3:6] -
                                                             data[edges, 0:3])
    if data[edges, 14] == -1:
        data[edges, 6:9] = np.array([-1,0,0])
        data[edges, 9:12] = np.array([0,-1,0])
    else:
        data[edges, 6:9] = np.array([1-data[edges, 12]**2/(1+data[edges, 14]),
                                     (-data[edges, 12]*data[edges, 13])/(1+data[edges, 14]),
                                     -data[edges, 12]])
        data[edges, 9:12] = np.array([(-data[edges, 12]*data[edges, 13])/(1+data[edges, 14]),
                                      1 - data[edges, 13]**2/(1+data[edges, 14]),
                                      -data[edges, 13]])

def tangent_basis(normal,pt):
    normal = normal/np.linalg.norm(normal)
    p = np.random.random(3)
    r = (p - pt)/np.linalg.norm(p - pt)
    t1 = r - np.dot(r,normal)*normal
    t1 = t1/np.linalg.norm(t1)
    t2 = np.cross(t1.T,normal)
    return t1.reshape(1,-1),t2.reshape(1,-1),normal