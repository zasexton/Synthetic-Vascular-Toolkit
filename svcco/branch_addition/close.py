import numpy as np
import numba as nb

@nb.jit(nopython=True,cache=True,nogil=True)
def close(data,point,N=20):
    a = data[:, 0:3]
    b = data[:, 3:6]
    distances = np.sqrt(np.sum(np.square((a+b)/2-point),axis=1))
    if N >= len(distances):
        N = len(distances)
    close_edges = np.argsort(distances)[:N]
    distances=distances[close_edges]
    return close_edges, distances

#@nb.jit(nopython=True,cache=True,nogil=True)
def close_exact(data,point):
    line_direction = data[:,12:15]
    ss = np.array([np.dot(data[i,0:3] - point,line_direction[i,:]) for i in range(data.shape[0])])
    tt = np.array([np.dot(point - data[i,3:6],line_direction[i,:]) for i in range(data.shape[0])])
    decision = [[ss[i],tt[i],0] for i in range(len(ss))]
    hh = np.array([np.max(np.array(i)) for i in decision])
    cc = np.cross((point - data[:,0:3]),line_direction,axis=1)
    cd = np.linalg.norm(cc,axis=1)
    line_distances = np.hypot(hh,cd)
    vessel = np.argsort(line_distances)
    return vessel, line_distances
