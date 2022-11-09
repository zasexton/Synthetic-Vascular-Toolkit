# Optimization of vessel connections

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from mpl_toolkits.mplot3d import Axes3D
import numba as nb
import nlopt
from math import tanh
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL


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

def get_radius(curve):
    num = curve.sample_size
    rad = []
    for i in np.linspace(0,1,num=num):
        _,d,dd = curve.derivatives(i,2)
        d  = np.array(d)
        dd = np.array(dd)
        value = np.linalg.norm(d)**3/(np.linalg.norm(np.cross(d,dd))+np.finfo(float).eps)
        rad.append(value)
    return rad


@nb.jit(nopython=True)
def get_collisions(collision_vessels,R,sample_pts):
    collisions = 0
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:])
        if np.any(dist<R):
            collisions += 10e8*tanh((R - min(dist))*(1/10))
    return collisions


@nb.jit(nopython=True)
def are_collisions(collision_vessels,R,sample_pts):
    collisions = False
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:])
        if np.any(dist<R):
            collisions = True
            break
    return collisions


@nb.jit(nopython=True)
def close_exact(data,point):
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
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2) - data[i,6]
    return line_distances

def connect_bezier(P1,P2,P3,P4,R,collision_vessels,degree=5):
    V1   = (P2 - P1)
    V1   = V1/np.linalg.norm(V1)
    V2   = (P4 - P3)
    V2   = V2/np.linalg.norm(V2)
    def create_bezier(data,V1=V1,V2=V2,R=R,P2=P2,P4=P4,degree=degree):
        CTR0 = P2 + (R+data[0])*V1
        CTR1 = P4 + (R+data[1])*V2
        CTR  = np.zeros(data.shape[0]-2+12)
        fill_CTR = np.zeros(data.shape[0]-2).reshape(-1,3)
        seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
        for i in range(fill_CTR.shape[0]):
            fill_CTR[i,:] = CTR0*(1-seg[i]) + CTR1*(seg[i]) #V1*(1-seg[i])*R + CTR1*(seg[i]) + V2*(seg[i])*R
        fill_CTR = fill_CTR.flatten()
        CTR[0:3]     = P2
        CTR[3:6]     = CTR0
        CTR[-6:-3]   = CTR1
        CTR[-3:]     = P4
        CTR[6:-6]    += fill_CTR + data[2:]
        CTR  = CTR.reshape(-1,3).tolist()
        curve = BSpline.Curve()
        curve.degree = degree
        curve.ctrlpts = CTR
        curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
        curve.sample_size = 40
        curve.evaluate()
        return curve
    return create_bezier


def bezier_cost(data,grad,create_curve=None,R=None,P1=None,P3=None,collision_vessels=None):
    curve = create_curve(data)
    pts   = np.array(curve.evalpts)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    #num = int(spline_length // R)
    #curve.sample_size = num
    #curve.evaluate()
    sample_pts = np.array(curve.evalpts)
    sample_pts_for_vec = np.vstack((P1,sample_pts,P3))
    vec = np.diff(sample_pts_for_vec,axis=0)
    if collision_vessels is not None:
        collisions = get_collisions(collision_vessels,R,sample_pts)
    else:
        collisions = 0
    #vec = vec/np.linalg.norm(vec,axis=0)
    #angles = get_vecs(vec)
    curve_rads = np.array(get_radius(curve))
    curve_rads = 1/(curve_rads[curve_rads<2*np.pi*R]+np.finfo(float).eps)
    a_sum = np.sum(curve_rads)
    vec = vec/np.linalg.norm(vec,axis=0)
    angles = 1/get_vecs(vec)
    if np.isclose(a_sum,0) or any(angles < 90):
        if np.any(angles < 90):
            a_sum = np.sum(1/angles)*10e8
    return collisions+a_sum*spline_length


def find_optimum_connection(P1,P2,P3,P4,R,collision_vessels):
    create_curve = connect_bezier(P1,P2,P3,P4,R,collision_vessels)
    cost = lambda d: bezier_cost(d,None,create_curve=create_curve,R=R,P1=P1,P3=P3,collision_vessels=collision_vessels)
    success = False
    x0 = np.zeros(8)
    max_time = 25
    count = 1
    L = 4*R
    level_attempts = 4
    attempt = 0
    best = np.inf
    while not success:
        #print('Linking Optimization Path Search {} Level {}'.format(count,attempt))
        lb = np.ones((2)*3+2)*(-L) + x0
        lb[0] = 0
        lb[1] = 0
        ub = np.ones((2)*3+2)*(L) + x0
        bounds = []
        for b in range(len(lb)):
            bounds.append([lb[b],ub[b]])
        #o = nlopt.opt(2,(2)*3+2)
        #o.set_min_objective(cost)
        #o.set_lower_bounds(lb)
        #o.set_upper_bounds(ub)
        #o.set_maxtime(max_time)
        #xopt = o.optimize(x0)
        res = optimize.shgo(cost,bounds=bounds,options={'maxtime':max_time})
        xopt = res.x
        curve = create_curve(xopt)
        vis_comp = VisMPL.VisCurve3D()
        curve.vis = vis_comp
        #if o.last_optimum_value() < 100 and attempt == level_attempts:
        if res.fun < 100 and attempt == level_attempts:
            L += L*0.25
            success = True
            #print(o.last_optimum_value())
        elif attempt < level_attempts:
            attempt += 1
            max_time += 5
            #if o.last_optimum_value() < best:
            if res.fun < best:
                x0 = xopt
                #best = o.last_optimum_value()
                best = res.fun
                #curve.render()
            #if np.isclose(o.last_optimum_value(),0):
            if np.isclose(res.fun,0):
                success = True
                #print('Constraints Satisfied: Exiting...')
            #curve.render()
            continue
        else:
            #print("Above link threshold {}".format(o.last_optimum_value()))
            L += L*0.25
            count += 1
            attempt = 0
            max_time = 25
            x0 = np.zeros((2)*3+2)
    curve = create_curve(xopt)
    #curve.vis  = vis_comp
    return curve


def get_optimum_link_points(P1,P2,P3,P4,R,collision_vessels):
    curve = find_optimum_connection(P1,P2,P3,P4,R,collision_vessels)
    pts   = np.array(curve.evalpts)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    num = int(spline_length // R)+2
    curve.sample_size = num
    curve.evaluate()
    opt_pts = np.array(curve.evalpts)
    #fig = plt.figure()
    #ax  = fig.add_subplot(111,projection='3d')
    #ax.plot(opt_pts[:,0],opt_pts[:,1],opt_pts[:,2],label='final')
    #plt.show()
    return opt_pts
