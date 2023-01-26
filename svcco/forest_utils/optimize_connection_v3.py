# Optimization for Vessel Connections

import numpy as np
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
import numba as nb
from scipy import interpolate, optimize

@nb.jit(nopython=True)
def get_angle(V1,V2):
    return np.arccos(np.dot(-V1,V2))*(180/np.pi)

@nb.jit(nopython=True)
def get_vecs(vec):
    angles = np.zeros(vec.shape[0]-1)
    for i in range(vec.shape[0]-1):
        value = get_angle(vec[i,:],vec[i+1,:])
        angles[i] = 1/value
    return angles

@nb.jit(nopython=True)
def close_exact(data,point,radius_buffer):
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
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2) - data[i,6] - radius_buffer
    return line_distances

@nb.jit(nopython=True)
def get_collisions(collision_vessels,R,sample_pts,radius_buffer):
    collisions = 0
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:],radius_buffer)
        if np.any(dist<R):
            collisions += R - min(dist)
    return collisions

@nb.jit(nopython=True)
def check_bounds(bounds,sample_pts):
    bounds_score = 0
    for i in range(sample_pts.shape[0]):
        if sample_pts[i,0] < bounds[0,0]:
            bounds_score += (sample_pts[i,0]**2 - bounds[0,0]**2)**(1/2)
        elif sample_pts[i,0] > bounds[0,1]:
            bounds_score += (sample_pts[i,0]**2 - bounds[0,1]**2)**(1/2)
        if sample_pts[i,1] < bounds[1,0]:
            bounds_score += (sample_pts[i,1]**2 - bounds[1,0]**2)**(1/2)
        elif sample_pts[i,1] > bounds[1,1]:
            bounds_score += (sample_pts[i,1]**2 - bounds[1,1]**2)**(1/2)
        if sample_pts[i,2] < bounds[2,0]:
            bounds_score += (sample_pts[i,2]**2 - bounds[2,0]**2)**(1/2)
        elif sample_pts[i,2] > bounds[2,1]:
            bounds_score += (sample_pts[i,2]**2 - bounds[2,1]**2)**(1/2)
    return bounds_score


def connect_bezier(P1,P2,P3,P4):
    V1   = (P2 - P1)
    L1   = np.linalg.norm(V1)
    V1   = V1/np.linalg.norm(V1)
    V2   = (P4 - P3)
    L2   = np.linalg.norm(V2)
    V2   = V2/np.linalg.norm(V2)
    def create_bezier(data,V1=V1,V2=V2,P2=P2,P4=P4,L1=L1,L2=L2):
        CTR0 = P2 + (L1+data[0])*V1
        CTR1 = P4 + (L2+data[1])*V2
        CTR  = np.zeros(data.shape[0]-2+12)
        fill_CTR = np.zeros(data.shape[0]-2).reshape(-1,3)
        seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
        for i in range(fill_CTR.shape[0]):
            fill_CTR[i,:] = CTR0*(1-seg[i]) + CTR1*(seg[i])
        fill_CTR = fill_CTR.flatten()
        CTR[0:3]     = P2
        CTR[3:6]     = CTR0
        CTR[-6:-3]   = CTR1
        CTR[-3:]     = P4
        CTR[6:-6]    += fill_CTR + data[2:]
        CTR  = CTR.reshape(-1,3).tolist()
        curve = BSpline.Curve()
        curve.degree = len(CTR) - 1
        curve.ctrlpts = CTR
        curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
        curve.sample_size = 20
        curve.evaluate()
        return curve
    return create_bezier

def bezier_cost(data,grad,create_curve=None,R=None,
                P1=None,P3=None,collision_vessels=None,
                radius_buffer=0,bounds=None,sample_size=20):
    curve = create_curve(data)
    curve.sample_size = sample_size
    curve.evaluate()
    pts   = np.array(curve.evalpts)
    sample_pts = np.array(curve.evalpts)
    if collision_vessels is not None:
        collisions = get_collisions(collision_vessels,R,sample_pts,radius_buffer)
    else:
        collisions = 0
    bound_score = check_bounds(bounds,sample_pts)
    return bound_score+collisions

def find_optimum_connection(P1,P2,P3,P4,R,collision_vessels,bounds=None,radius_buffer=0):
    create_curve = connect_bezier(P1,P2,P3,P4)
    cost = lambda data: bezier_cost(data,None,create_curve=create_curve,R=R,P1=P1,P3=P3,collision_vessels=collision_vessels,radius_buffer=radius_buffer,bounds=bounds)
    success = False
    free_points = 1
    x0 = np.zeros(2+free_points*3)
    best = np.inf
    L = np.linalg.norm(P2-P4)
    while not success:
        lb = np.ones(x0.shape[0])*(-L)
        lb[0] = 0
        lb[1] = 0
        ub = np.ones(x0.shape[0])*L
        ctrl_bounds = []
        for b in range(len(lb)):
            ctrl_bounds.append([lb[b],ub[b]])
        res = optimize.shgo(cost,bounds=ctrl_bounds,options={'f_min':0})
        xopt = res.x
        if res.fun < 0.1:
            success = True
        else:
            free_points += 1
            x0 = np.zeros(2+free_points*3)
    curve = create_curve(xopt)
    return curve

def get_optimum_link_points(P1,P2,P3,P4,R,collision_vessels,bounds=None,radius_buffer=0):
    curve = find_optimum_connection(P1,P2,P3,P4,R,collision_vessels,bounds=bounds,radius_buffer=radius_buffer)
    pts   = np.array(curve.evalpts)
    #spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    num = 10
    curve.sample_size = num
    curve.evaluate()
    opt_pts = np.array(curve.evalpts)
    return opt_pts
