import numpy as np
import numba as nb
import math

@nb.jit(nopython=True,cache=True,nogil=True)
def close(data,point,N=20):
    a = data[:, 0:3]
    b = data[:, 3:6]
    distances = np.sqrt(np.sum(np.square((a+b)/2-point),axis=1))
    if N >= len(distances):
        N = len(distances)
    close_edges = np.argsort(distances)[:N]
    distances=distances[close_edges]
    return close_edges, distances

#@nb.jit(nopython=True,cache=True,nogil=True)
# this code can be optimized with reference to a working form of the
# code presented in optimize_connection_v2 module for close_exact
# the repeated list comprehensions likely slow down results
# ~should be at least 7 times faster~
def close_exact(data,point):
    line_direction = data[:,12:15]
    ss = np.array([np.dot(data[i,0:3] - point,line_direction[i,:]) for i in range(data.shape[0])])
    tt = np.array([np.dot(point - data[i,3:6],line_direction[i,:]) for i in range(data.shape[0])])
    decision = [[ss[i],tt[i],0] for i in range(len(ss))]
    hh = np.array([np.max(np.array(i)) for i in decision])
    cc = np.cross((point - data[:,0:3]),line_direction,axis=1)
    cd = np.linalg.norm(cc,axis=1)
    line_distances = np.hypot(hh,cd)
    vessel = np.argsort(line_distances)
    return vessel, line_distances

# Correct implementation of the JIT compiled exact distance test
@nb.jit(nopython=True)
def close_exact_v2(data,point):
    line_direction = np.zeros((data.shape[0],3))
    ss = np.zeros(data.shape[0])
    tt = np.zeros(data.shape[0])
    hh = np.zeros(data.shape[0])
    cc = np.zeros((data.shape[0],3))
    cd = np.zeros(data.shape[0])
    line_distances = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        line_direction[i,:] = (data[i,3:6] - data[i,0:3])/np.linalg.norm(data[i,3:6] - data[i,0:3])
        ss[i] = np.dot(data[i,0:3]-point,line_direction[i,:])
        tt[i] = np.dot(point-data[i,3:6],line_direction[i,:])
        d = np.array([ss[i],tt[i],0])
        hh[i] = np.max(d)
        diff = point - data[i,0:3]
        cc[i,:] = np.cross(diff,line_direction[i,:])
        cd[i] = np.linalg.norm(cc[i,:])
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2)
    return line_distances

"""
def close_exact(data,point):
    line_direction = data[:,12:15]
    ss = np.array([np.dot(data[i,0:3] - point,line_direction[i,:]) for i in range(data.shape[0])])
    tt = np.array([np.dot(point - data[i,3:6],line_direction[i,:]) for i in range(data.shape[0])])
    decision = [[ss[i],tt[i],0] for i in range(len(ss))]
    hh = np.array([np.max(np.array(i)) for i in decision])
    cc = np.cross((point - data[:,0:3]),line_direction,axis=1)
    cd = np.linalg.norm(cc,axis=1)
    line_distances = np.hypot(hh,cd)
    vessel = np.argsort(line_distances)
    return vessel, line_distances
"""
@nb.jit(nopython=True,nogil=True)
def close_binary_vectorize(data,point):
    best_dist = np.inf
    best      = -1.0
    for vessel in data:
        line_direction = vessel[12:15].reshape(-1,1)
        diff = vessel[0:3]-point
        ss = np.dot(diff,line_direction)
        tt = np.dot(point-vessel[3:6],line_direction)
        hh = max([ss[0].item(),tt[0].item(),0])
        diff2 = (point-vessel[0:3]) #.reshape(-1,1)
        line_direction2 = line_direction.T
        cc = np.cross(diff2,line_direction2)
        cd = np.linalg.norm(cc)
        ld = math.hypot(hh,cd).item()
        if ld < best_dist:
            best_dist = ld
            best = vessel[-1]
    return best,best_dist

@nb.jit(nopython=True,nogil=True)
def close_binary_vectorize2(data,point):
    best_dist = np.inf
    best      = -1.0
    for vessel in data:
        diff = [0,0,0]
        diff2 = [0,0,0]
        diff3 = [0,0,0]
        cc = [0,0,0]
        line_direction = vessel[12:15]
        diff[0] = vessel[0]-point[0]
        diff[1] = vessel[1]-point[1]
        diff[2] = vessel[2]-point[2]
        ss = diff[0]*line_direction[0]+diff[1]*line_direction[1]+diff[2]*line_direction[2]
        #ss = np.dot(diff,line_direction)
        diff2[0] = point[0]-vessel[3]
        diff2[1] = point[1]-vessel[4]
        diff2[2] = point[2]-vessel[5]
        #tt = np.dot(point-vessel[3:6],line_direction)
        tt = diff2[0]*line_direction[0]+diff2[1]*line_direction[1]+diff2[2]*line_direction[2]
        #hh = max([ss,tt,0])
        if ss > tt and ss > 0:
            hh = ss
        elif tt > ss and tt>0:
            hh=tt
        else:
            hh=0.0
        #diff2 = (point-vessel[0:3]) #.reshape(-1,1)
        diff3[0] = point[0]-vessel[0]
        diff3[1] = point[1]-vessel[1]
        diff3[2] = point[2]-vessel[2]
        #line_direction2 = line_direction.T
        #cc = np.cross(diff2,line_direction2)
        cc[0] = diff3[1]*line_direction[2] - diff3[2]*line_direction[1]
        cc[1] = diff3[2]*line_direction[0] - diff3[0]*line_direction[2]
        cc[2] = diff3[0]*line_direction[1] - diff3[1]*line_direction[0]
        #cd = np.linalg.norm(cc)
        cd = (cc[0]**2+cc[1]*2+cc[2]**2)**(1/2)
        #ld = math.hypot(hh,cd).item()
        ld = (hh**2+cd**2)**(1/2)
        if ld < best_dist:
            best_dist = ld
            best = vessel[-1]
    return best,best_dist
