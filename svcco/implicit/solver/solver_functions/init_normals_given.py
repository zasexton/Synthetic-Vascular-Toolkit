import numpy as np

def init_normals_given(ic_norms):
    init_norms = np.zeros(ic_norms.shape[0]*(ic_norms.shape[1]-1))
    for i in range(ic_norms.shape[0]):
        #for j in range(ic_norms.shape[1]-1):
        #    if j == ic_norms.shape[1] - 1:
        #        if ic_norms[i,j] >= 0:
        #            init_norms[i*(ic_norms.shape[1]-1)+j] = np.arccos(ic_norms[i,j]/np.linalg.norm(ic_norms[i,j:]))
        #        else:
        #            init_norms[i*(ic_norms.shape[1]-1)+j] = 2*np.pi - np.arccos(ic_norms[i,j]/np.linalg.norm(ic_norms[i,j:]))
        #    else:
        #        init_norms[i*(ic_norms.shape[1]-1)+j] = np.arccos(ic_norms[i,j]/np.linalg.norm(ic_norms[i,j:]))
        init_norms[i*2] = np.arctan2((ic_norms[i,0]**2+ic_norms[i,1]**2)**(1/2),ic_norms[i,2])
        init_norms[i*2+1] = np.arctan2(ic_norms[i,1],ic_norms[i,0])
    return init_norms
