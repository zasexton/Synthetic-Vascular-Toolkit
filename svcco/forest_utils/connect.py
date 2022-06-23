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

from copy import deepcopy
import pyvista as pv

def rotate(R0,R1,P0,P1,theta):
    theta = (theta/180)*np.pi
    Rvector = R1-R0
    Rvector = Rvector/np.linalg.norm(Rvector)
    vector = P1-P0
    vector = vector/np.linalg.norm(vector)
    vector_parallel = (np.dot(vector,Rvector.T)/np.dot(Rvector,Rvector.T))*Rvector
    vector_perp     = vector-vector_parallel
    w      = np.cross(Rvector,vector_perp)
    vector_perp_norm = np.linalg.norm(vector_perp)
    w_norm = np.linalg.norm(w)
    if vector_perp_norm > 0 and w_norm > 0:
        x1     = np.cos(theta)/np.linalg.norm(vector_perp)
        x2     = np.sin(theta)/np.linalg.norm(w)
        rotated_vector = np.linalg.norm(vector_perp)*(x1*vector_perp+x2*w)+vector_parallel
    else:
        #print(vector_perp)
        #print(w)
        rotated_vector = vector_parallel
    new_point = P0+rotated_vector*(np.linalg.norm(P1-P0))
    return new_point

def angle(P0,P1,P2):
    vector_1 = P0 - P1
    vector_2 = P2 - P1
    vector_1 = vector_1/np.linalg.norm(vector_1)
    vector_2 = vector_2/np.linalg.norm(vector_2)
    result   = (np.arccos(np.dot(vector_1,vector_2))/np.pi)*180
    return result


def parallel(P0,P1,P2):
    vector_1 = P0 - P1
    vector_2 = P2 - P1
    vector_1 = vector_1/np.linalg.norm(vector_1)
    vector_2 = vector_2/np.linalg.norm(vector_2)
    return np.all(np.isclose(vector_1,vector_2))

"""
def rotate_terminals(forest_copy):
    ANGLE_STEP = 5
    for ndx in range(forest_copy.number_of_networks):
        #for tdx in range(forest_copy.trees_per_network[ndx]):
        #A = forest_copy.assignments[ndx][tdx]
        #T = forest_copy.networks[ndx][tdx].data
        #C = forest_copy.connections[ndx][0]
        for edx in range(len(forest_copy.connections[ndx][0])):
            #CC = []
            #if T[A[edx],17] < 0:
            #    #Terminal root (cannot rotate)
            #    continue
            #elif parallel(T[A[edx],0:3],T[A[edx],3:6],TT[AA[edx],3:6]):
            #    continue
            #ANGLE = angle(T[A[edx],0:3],T[A[edx],3:6],TT[AA[edx],3:6])
            #attempt = 0
            #parent  = int(T[A[edx],17])
            #score  = [ANGLE]
            #center = (T[A[edx],3:6]+TT[AA[edx],3:6])/2
            #CC.append(center)
            #pts   = [deepcopy(T[A[edx],3:6])]
            max_attempt = int(360/ANGLE_STEP)
            CC = []
            C = forest_copy.connections[ndx][0]
            score  = [ANGLE]
            for kdx in range(forest_copy.trees_per_network[ndx]):
                A = forest_copy.assignments[ndx][kdx]
                T = forest_copy.networks[ndx][kdx].data
                #C = forest_copy.connections[ndx][0]
                ANGLE = angle(T[A[edx],0:3],T[A[edx],3:6],C[edx]])
                attempt = 0
                parent  = int(T[A[edx],17])
                score  = [ANGLE]
                CC.append(deepcopy(C[edx]))
                T = T = forest_copy.networks[ndx][tdx].data
                while attempt < max_attempt:
                    R0 = deepcopy(T[parent,0:3])
                    R1 = deepcopy(T[parent,3:6])
                    P0 = deepcopy(T[A[edx],0:3])
                    P1 = deepcopy(T[A[edx],3:6])
                    T[A[edx],3:6] = rotate(R0,R1,P0,P1,ANGLE_STEP)
                    T[A[edx],12:15] = (T[A[edx],3:6]-T[A[edx],0:3])/np.linalg.norm(T[A[edx],3:6]-T[A[edx],0:3])
                    center = np.zeros(3)
                    for tdx in range(forest_copy.trees_per_network[ndx]):
                        AA = forest_copy.assignments[ndx][tdx]
                        TT = forest_copy.networks[ndx][tdx].data
                        center += TT[AA[edx],3:6]
                    center = center/len(forest_copy.trees_per_network[ndx])
                    C[edx] = center
                    ANGLE = angle(T[A[edx],0:3],T[A[edx],3:6],C[edx]])
                    #center = (T[A[edx],3:6]+TT[AA[edx],3:6])/2
                    CC.append(deepcopy(C[edx]))
                    #view(forest_copy,highlight=[ndx,tdx,int(T[A[edx],-1])])
                    score.append(ANGLE)
                    pts.append(deepcopy(T[A[edx],3:6]))
                    attempt += 1
            if max(score) < 90:
                print("Fail")
                print(max(score))
            best = np.argmax(score)
            T[A[edx],3:6] = pts[best]
            T[A[edx],12:15] = (T[A[edx],3:6]-T[A[edx],0:3])/np.linalg.norm(T[A[edx],3:6]-T[A[edx],0:3])
            C[edx] = CC[best]

from itertools import product

def rotate_terminals(forest_copy):
    ANGLE_STEP = 5
    for ndx in range(forest_copy.number_of_networks):
        for edx in range(len(forest_copy.connections[ndx][0])):
            pts = []
            non_pts = []
            for tdx in range(forest_copy.trees_per_network[ndx]):
                rotated_pts = []
                non_rotated_pts = []
                T  = forest_copy.networks[ndx][tdx].data
                A  = forest_copy.assignments[ndx][tdx]
                parent = int(T[A[edx],17])
                R0 = deepcopy(T[parent,0:3])
                R1 = deepcopy(T[parent,3:6])
                P0 = deepcopy(T[A[edx],0:3])
                P1 = deepcopy(T[A[edx],3:6])
                for theta in np.arange(0,360,ANGLE_STEP):
                     non_rotated_pts.append(P0)
                     rotated_pts.append(rotate(R0,R1,P0,P1,theta))
                pts.append(np.array(rotated_pts))
                non_pts.append(np.array(non_rotated_pts))
            pts = np.array(pts)
            non_pts = np.array(non_pts)
            comb = np.array(list(product(*[range(len(rotated_pts))]*forest_copy.trees_per_network[ndx])))
            CENTERS = pts[0,comb[:,0],:]
            for tdx in range(1,forest_copy.trees_per_network[ndx]):
                CENTERS += pts[tdx,comb[:,tdx],:]
            CENTERS = CENTERS/pts.shape[0]
            ANGLES  = []
            for tdx in range(forest_copy.trees_per_network[ndx]):

                VECTOR_1 = non_pts[tdx,comb[:,tdx],:] - pts[tdx,comb[:,tdx],:]
                VECTOR_2 = CENTERS - pts[tdx,comb[:,tdx],:]
                VECTOR_1 = VECTOR_1/np.linalg.norm(VECTOR_1,axis=1).reshape(-1,1)
                VECTOR_2 = VECTOR_2/np.linalg.norm(VECTOR_2,axis=1).reshape(-1,1)
                print(VECTOR_1[0:3,:])
                print(VECTOR_2[0:3,:])
                DOT      = np.array([np.dot(VECTOR_1[i,:],VECTOR_2[i,:]) for i in range(VECTOR_1.shape[0])])
                ang      = (np.arccos(DOT)/np.pi)*180

                ANGLES.append(ang)
            ANGLES = np.array(ANGLES)
            return

def view(forest,highlight=None):
    model_networks = []
    for net_id,network in enumerate(forest.networks):
        model_trees = []
        for tree_id,network_tree in enumerate(network):
            model = []
            for edge in range(network_tree.data.shape[0]):
                if network_tree.data[edge,15]==-1 and network_tree.data[edge,16]==-1 and edge != 0:
                    dis_point = (network_tree.data[edge,0:3] + network_tree.data[edge,3:6])/2
                    vessel_length = network_tree.data[edge,20]/2
                else:
                    dis_point = network_tree.data[edge,3:6]
                    vessel_length = network_tree.data[edge,20]
                center = tuple((network_tree.data[edge,0:3] + dis_point)/2)
                radius = network_tree.data[edge,21]
                direction = tuple(network_tree.data[edge,12:15])
                model.append(pv.Cylinder(radius=radius,height=vessel_length,
                                         center=center,direction=direction,
                                         resolution=100))
            model_trees.append(model)
        model_networks.append(model_trees)
    plot = pv.Plotter()
    #colors = ['r','b','g','y']
    #plot.set_background(color=[253,250,219])
    for net_idx,model_network in enumerate(model_networks):
        for tree_idx, model_tree in enumerate(model_network):
            for vessel_idx,model in enumerate(model_tree):
                if net_idx == highlight[0] and tree_idx == highlight[1] and vessel_idx==highlight[2]:
                    plot.add_mesh(model,color='g')
                else:
                    plot.add_mesh(model,color='r')
    plot.show()
"""
