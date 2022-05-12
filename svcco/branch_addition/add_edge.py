import numpy as np
import numba as nb

##################################################
# Adding Edge to Tree
##################################################

@nb.jit(nopython=True,cache=True,nogil=True)
def add_edge(data,proximal,distal,parent,flow,proximal_idx=-1.0,distal_idx=-1.0,
             left_child=-1.0,right_child=-1.0):
    edge = np.zeros(data.shape[1])
    # Set coordinates of new edge
    edge[0:3] = proximal
    edge[3:6] = distal
    # Normalized direction of added edge
    edge[12:15] = (distal - proximal) / np.linalg.norm(distal-proximal)
    # Get a random normal vector to the edge
    if edge[14] == -1.0:
        edge[6:9] = np.array([-1.0,0.0,0.0])
        edge[9:12] = np.array([0.0,-1.0,0.0])
    else:
        edge[6:9] = np.array([1.0-edge[12]**2/(1.0+edge[14]),
                              (-edge[12]*edge[13])/(1.0+edge[14]),
                              -edge[12]])
        edge[9:12] = np.array([(-edge[12]*edge[13])/(1.0+edge[14]),
                               1.0 - edge[13]**2/(1.0+edge[14]),
                               -edge[13]])
    # If this edge already had children then update them
    if left_child != -1.0:
        edge[15] = left_child.item()
    else:
        edge[15] = left_child
    if right_child != -1.0:
        edge[16] = right_child.item()
    else:
        edge[16] = right_child
    #edge[17] = parent
    # Determine node indicies for new edge
    if proximal_idx == -1.0:
        item1 = np.max(data[:,18])
        item2 = np.max(data[:,19])
        edge[18] = np.max(np.array([item1,item2])) + 1
    else:
        edge[18] = proximal_idx
    if distal_idx == -1.0:
        edge[19] = edge[18] + 1
    else:
        edge[19] = distal_idx
    # Length of new edge
    edge[20] = np.sqrt(np.sum(np.square(distal-proximal)))
    # index 21: radius, not determined yet
    edge[22] = flow
    # index 23: left bifurcation, not determined yet
    # index 24: right bifurcation, not determined yet
    # index 25: reduced resistance, not determined yet
    edge[26] = data[parent, 26].item() + 1
    if left_child != -1 and right_child != -1:
        edge[27] = data[parent, 27].item()
    # Actual edge
    edge[29] = -1
    # Index of edge
    edge[-1] = data.shape[0] - 1
    # Add edge to tree
    data[-1,:] = edge
