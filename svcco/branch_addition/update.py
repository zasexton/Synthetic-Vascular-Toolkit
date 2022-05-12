import numba as nb
import numpy as np

@nb.jit(nopython=True,cache=True,nogil=True)
def update(data,gamma,nu):
    depth_max = np.amax(data[:,26])
    start = 0
    while int(depth_max) >= 0:
        edges = np.argwhere(data[:,26] == depth_max)
        depth_max = depth_max - 1
        for idx in edges:
            if data[idx,15] < 0 or data[idx,16] < 0:
                data[idx,25] = (8*nu/np.pi)*data[idx,20]
                data[idx,23] = 0.0
                data[idx,24] = 0.0
                data[idx,27] = 0.0
            else:
                left = int(data[idx, 15].item())
                right = int(data[idx, 16].item())
                #print('right: '+str(right))
                #print(data[right,22])
                #print(data[left,25])
                #print('left:  '+str(left))
                #print(data[left,22])
                #print(data[left,25])
                LR = ((data[left, 22]*
                       data[left, 25])/
                      (data[right, 22]*
                       data[right, 25])) ** (1/4)
                #print('passed')
                lbif = (1+ LR ** (-gamma)) ** (-1/gamma)
                rbif = (1+ LR ** (gamma)) ** (-1/gamma)
                data[idx,25] = (8*nu/np.pi)*data[idx,20] + \
                               ((lbif ** 4 / data[left, 25]) +
                                (rbif ** 4 / data[right, 25])) ** -1
                data[idx,23] = lbif
                data[idx,24] = rbif
                data[idx,27] = lbif**2 * (data[left,20]+data[left,27])+\
                               rbif**2 * (data[right,20]+data[right,27])
