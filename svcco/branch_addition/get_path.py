import numba as nb
import numpy as np

@nb.jit(nopython=True)
def get_path(D,Pr,i,j):
    path = [j]
    length = 0
    while Pr[i,k] != -9999:
        path.append(Pr[i,k])
        length += D[i,k]
        k = Pr[i,k]
    return path[::-1],length
