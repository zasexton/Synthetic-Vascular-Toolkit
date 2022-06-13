import numba as nb
import numpy as np

@nb.jit(nopython=True,cache=True,nogil=True)
def radii(data,Pperm,Pterm):
    updated_radii = [0]
    data[0,21] = ((data[0,25]*data[0,22])/(Pperm-Pterm)) ** (1/4)
    if data[0,15] < 0:
        return
    children_d = data[0,15:17].flatten()
    children_i = children_d.astype(np.int64)
    parents_d = data[children_i,17].flatten()
    parents_i = parents_d.astype(np.int64)
    bifs = data[parents_i[::2],23:25].flatten()
    radii = data[parents_i,21].flatten()
    while len(children_i) > 0:
        data[children_i,21] = bifs*radii
        updated_radii.extend(children_i)
        next_children = np.argwhere(data[children_i,15] > 0)
        next_children = children_i[next_children.flatten()]
        children_d = data[next_children,15:17].flatten()
        children_i = children_d.astype(np.int64)
        parents_d = data[children_i,17].flatten()
        parents_i = parents_d.astype(np.int64)
        bifs = data[parents_i[::2],23:25].flatten()
        radii = data[parents_i,21].flatten()
    return updated_radii

def update_radii(data,Pperm,Pterm):
    data[0,21] = ((data[0,25]*data[0,22])/(Pperm-Pterm)) ** (1/4)
    idx = [int(data[0,15]),int(data[0,16])]
    while len(idx) > 0:
        tmp = []
        for vessel in idx:
            parent = int(data[vessel,17])
            LR  = data[parent,15:17].astype(int)
            BIF = data[parent,23:25].flatten()
            LR_type = np.argwhere(LR == vessel).flatten()[0]
            BIF = BIF[LR_type]
            data[vessel,21] = data[parent,21]*BIF
            data[vessel,28] = data[parent,28]*BIF
            if data[vessel,15]>0:
                tmp.append(int(data[vessel,15]))
            if data[vessel,16]>0:
                tmp.append(int(data[vessel,16]))
        idx = tmp

##################################################################
# UNIT TEST
##################################################################
