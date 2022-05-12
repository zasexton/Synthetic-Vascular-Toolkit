# Script for connecting trees of closed vascular networks
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

def connect(forest,network_id=-1,buffer=None):
    connections = []
    assignments = []
    if network_id == -1:
        for network in range(forest.number_of_networks):
            network_connections = []
            network_assignments = []
            tree_0 = forest.networks[network][0].data
            tree_1 = forest.networks[network][1].data
            terminals_0 = tree_0[np.argwhere(tree_0[:,15]==-1).flatten(),:]
            terminals_0 = terminals_0[np.argwhere(terminals_0[:,16]==-1).flatten(),:]
            terminals_0_ind = terminals_0[:,-1].astype(int).flatten()
            terminals_0_pts = terminals_0[:,3:6]
            terminals_1 = tree_1[np.argwhere(tree_1[:,15]==-1).flatten(),:]
            terminals_1 = terminals_1[np.argwhere(terminals_1[:,16]==-1).flatten(),:]
            terminals_1_ind = terminals_1[:,-1].astype(int).flatten()
            terminals_1_pts = terminals_1[:,3:6]
            network_assignments.append(terminals_0_ind)

            C = cdist(terminals_0_pts,terminals_1_pts)
            if not forest.convex:
                W = np.zeros(C.shape)
                penalty = np.max(C)
                for i in range(C.shape[0]):
                    for j in range(C.shape[1]):
                        mid = (terminals_0_pts[i] + terminals_1_pts[j])/2
                        mid = mid.flatten()
                        if not forest.boundary.within(mid[0],mid[1],mid[2],2):
                            W[i,j] = penalty
                C = C + W
            _,assignment = linear_sum_assignment(C)
            network_assignments.append(terminals_1_ind[assignment])
            midpoints  = (terminals_0_pts + terminals_1_pts[assignment])/2
            network_connections.append(midpoints)
            if forest.trees_per_network[network] > 2:
                for N in range(2,forest.trees_per_network[network]):
                    tree_n = forest.networks[network][N].data
                    terminals_n = tree_n[np.argwhere(tree_n[:,15]==-1).flatten(),:]
                    terminals_n = terminals_n[np.argwhere(terminals_n[:,16]==-1).flatten(),:]
                    terminals_n_ind = terminals_n[:,-1].astype(int).flatten()
                    terminals_n_pts = terminals_n[:,3:6]

                    C = cdist(midpoints,terminals_n_pts)
                    if not forest.convex:
                        W = np.zeros(c.shape)
                        penalty = np.max(C)
                        for i in range(C.shape[0]):
                            for j in range(C.shape[1]):
                                mid = (midpoints[i] + terminals_n_pts[j])/2
                                mid = mid.flatten()
                                if not forest.boundary.within(mid[0],mid[1],mid[2],2):
                                    W[i,j] = penalty
                        C = C + W
                    _,assignment = linear_sum_assignment(C)
                    network_assignments.append(terminals_n_ind[assignment])
            connections.append(network_connections)
            assignments.append(network_assignments)
    return connections, assignments
