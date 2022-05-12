import numpy as np
from .basis import *
from .triangle import *

def to_point(start,end,grad,fraction,hess=None):
    path = []
    start = start.flatten()
    end = end.flatten()
    start_norm = grad(*start)
    t1,t2,n = tangent_basis(start_norm,start)
    theta = np.linspace(0,2*np.pi,num=40)
    potential_next_steps = start + t1*np.sin(theta).reshape(-1,1) + t2*np.cos(theta).reshape(-1,1)
    optimal_step = np.argmin(np.linalg.norm(potential_next_steps-end,axis=1)).flatten()[0]
    path.append(potential_next_steps[optimal_step])
    dist = np.linalg.norm(path[-1]-end)
    while dist > fraction:
        tmp_norm = grad(*path[-1])
        t1,t2,n = tangent_basis(tmp_norm,path[-1].flatten())
        potential_next_steps = path[-1] + t1*np.sin(theta).reshape(-1,1) + t2*np.cos(theta).reshape(-1,1)
        optimal_step = np.argmin(np.linalg.norm(potential_next_steps-end,axis=1)).flatten()[0]
        if dist - np.linalg.norm(potential_next_steps[optimal_step] - end) < 0.5*fraction:
            #direct_point = path[-1] + ((end - path[-1])/np.linalg.norm(end - path[-1]))*fraction
            #path.append(direct_point)
            break 
        else:
            path.append(potential_next_steps[optimal_step])
        dist = np.linalg.norm(path[-1] - end)
        #else:
        #    path.append(level_func(*potential_next_steps[pt_idx].flatten(),p0_target).x)
    path.append(end)
    path = np.array(path)
    return path

def sample_area(segment_data,term,grad,fraction):
    sides = []
    for i in range(segment_data.shape[0]):
        sides.append(to_point(segment_data[i,:],term,grad,fraction))
    points = np.vstack(sides)
    return points
    """
    side1 = to_point(p0,p1,grad,fraction)
    side2 = to_point(p0,p2,grad,fraction)
    side3 = to_point(p1,p2,grad,fraction)
    mid1 = side1[round(side1.shape[0]/2),:]
    mid2 = side2[round(side2.shape[0]/2),:]
    mid3 = side3[round(side3.shape[0]/2),:]
    mline1 = to_point(mid1,p2,grad,fraction)
    mline2 = to_point(mid2,p0,grad,fraction)
    mline3 = to_point(mid3,p1,grad,fraction)
    points = np.vstack((side1,side2,side3,mline1,mline2,mline3))
    """
    #return points
