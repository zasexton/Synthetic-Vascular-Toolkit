# rotate connections to avoid gross collisions
import numpy as np
from copy import deepcopy
from ..collision.collision import *

def rotate(R0,R1,P0,P1,theta):
    theta = (theta/180)*np.pi
    Rvector = R1-R0
    Rvector = Rvector/np.linalg.norm(Rvector)
    vector = P1-P0
    vector = vector/np.linalg.norm(vector)
    vector_parallel = (np.dot(vector,Rvector.T)/np.dot(Rvector,Rvector.T))*Rvector
    vector_perp     = vector-vector_parallel
    pn = np.linalg.norm(vector_perp)
    wn = np.linalg.norm(w)
    if pn < 0.1 or wn < 0.1:
        print(pn)
        print(wn)
        print(vector_perp)
        print(w)
    w      = np.cross(Rvector,vector_perp)
    x1     = np.cos(theta)/np.linalg.norm(vector_perp)
    x2     = np.sin(theta)/np.linalg.norm(w)
    rotated_vector = np.linalg.norm(vector_perp)*(x1*vector_perp+x2*w)+vector_parallel
    new_point = P0+rotated_vector*(np.linalg.norm(P1-P0))
    return new_point

def angle(P0,P1,P2):
    vector_1 = P0 - P1
    vector_2 = P2 - P1
    vector_1 = vector_1/np.linalg.norm(vector_1)
    vector_2 = vector_2/np.linalg.norm(vector_2)
    result   = np.arccos(np.dot(vector_1,vector_2)/np.pi)*180
    return result

def rotate_terminals(forest_copy):
    ANGLE_STEP = 5
    for ndx in range(forest_copy.number_of_networks):
        for tdx in range(forest_copy.trees_per_network):
            A = forest_copy.assignments[ndx][tdx]
            T = forest_copy.networks[ndx][tdx].data
            C = forest_copy.connections[ndx][0]
            for edx in range(len(forest_copy.connections[ndx][0])):
                if T[edx,17] < 0:
                    #Terminal root (cannot rotate)
                    continue
                ANGLE = angle(T[A[edx],0:3],T[A[edx],3:6],C[edx])
                attempt = 0
                parent  = int(T[A[edx],17])
                score = [angle]
                pts   = [deepcopy(T[A[edx],3:6])]
                while ANGLE < 90 and attempt < 10:
                    R0 = T[parent,0:3]
                    R1 = T[parent,3:6]
                    P0 = T[A[edx],0:3]
                    P1 = T[A[edx],3:6]
                    T[A[edx],3:6] = rotate(R0,R1,P0,P1,ANGLE_STEP)
                    ANGLE = angle(T[A[edx],0:3],T[A[edx],3:6],C[edx])
                    score.append(ANGLE)
                    pts.append(deepcopy(T[A[edx],3:6]))
                    attempt += 1
                if attempt == 10:
                    print("Fail")
                best = np.argmax(score)
                T[A[edx],3:6] = pts[best]
                T[A[edx],12:15] = T[A[edx],3:6]-T[A[edx],0:3]/np.linalg.norm(T[A[edx],3:6]-T[A[edx],0:3])
