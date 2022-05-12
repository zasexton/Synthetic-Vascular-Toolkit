import numpy as np
import numba as nb

@nb.jit(nopython=True,cache=True,nogil=True)
def sphere_proximity(data,edge):
    C1 = (edge[0:3] + edge[3:6]) / 2
    C2 = (data[:,0:3] + data[:,3:6]) / 2
    r1 = np.sqrt(edge[21]**2+(edge[20]/2)**2)
    r2 = np.sqrt(data[:,21]**2+(data[:,20]/2)**2)
    D = np.sqrt(np.sum(np.square(C1 - C2), axis=1))
    return np.argwhere(D < (r1 + r2)).flatten()

@nb.jit(nopython=True,cache=True,nogil=True)
def sphere_point_query(point,distance,data):
    C = (data[:,0:3] + data[:,3:6]) / 2
    R = np.sqrt(data[:,21]**2+(data[:,20]/2)**2)
    D = np.sqrt(np.sum(np.square(C - point), axis=1))
    return np.argwhere(D<(distance+R)).flatten()
