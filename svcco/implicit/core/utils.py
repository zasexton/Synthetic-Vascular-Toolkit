import numpy as np
import numba as nb

@nb.jit(nopython=True)
def norm(x):
    tmp = 0
    n = len(x)
    for i in range(n):
        tmp += abs(x[i])**2
    return tmp**(1/2)
