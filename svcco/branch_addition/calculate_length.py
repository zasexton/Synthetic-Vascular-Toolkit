import numba as nb
import numpy as np

@nb.jit(nopython=True,cache=True,nogil=True)
def length(data, edge):
    data[edge,20] = np.sqrt(np.sum(np.square(data[edge,0:3]-data[edge,3:6])))