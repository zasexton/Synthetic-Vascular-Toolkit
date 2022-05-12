import numba as nb
import numpy as np

@nb.jit(nopython=True,cache=True,nogil=True)
def check(data,point_set,threshold):
    centers = (data[:,0:3] + data[:,3:6])/2
    for i in range(point_set.shape[0]):
        distances = np.sqrt(np.sum(np.square(centers-point_set[i,:]),axis=1))
        if np.all(distances > threshold):
            return True, point_set[i,:]
    return False,None

@nb.jit(nopython=True,cache=True,nogil=True)
def check_point(data,point,threshold):
    centers = (data[:,0:3] + data[:,3:6])/2
    distances = np.sqrt(np.sum(np.square(centers-point),axis=1))
    if np.all(distances > threshold):
        return True
    return False