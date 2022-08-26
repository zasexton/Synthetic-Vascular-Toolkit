"""
Code to evaluate computational cost for bifurcation optimization
"""
import svcco
import numpy as np
from time import time, perf_counter
from copy import deepcopy
import matplotlib.pyplot as plt

#####################################################
# CODE FOR BUILDING SVCCO Tree
#####################################################
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(10*cube.points,normals=cube.point_normals)
s.solve()
s.build()

t = svcco.tree()
t.set_boundary(s)
t.set_root()
t.convex = True
t.n_add(10)
#####################################################

def test_naive(t):
    rng_points,_ = t.boundary.pick(size=len(t.boundary.tet_verts),homogeneous=True,replacement=False)
    rng_points = rng_points.tolist()
    point = np.array(rng_points.pop(0))
    vessel, line_distances = svcco.branch_addition.close.close_exact(t.data,point)
    vessel = vessel[0]
    sampling=20
    points   = svcco.branch_addition.triangle.get_local_points(t.data,vessel,point.flatten(),sampling,t.clamped_root)
    mu                 = t.parameters['mu']
    lam                = t.parameters['lambda']
    gamma              = t.parameters['gamma']
    nu                 = t.parameters['nu']
    Qterm              = t.parameters['Qterm']
    Pperm              = t.parameters['Pperm']
    Pterm              = t.parameters['Pterm']
    data = deepcopy(t.data)
    start = perf_counter()
    svcco.branch_addition.finite_difference.finite_difference_naive(data,points,point.flatten(),vessel,gamma,nu,Qterm,Pperm,Pterm)
    elapsed = perf_counter() - start
    return elapsed

def test_binding(t):
    rng_points,_ = t.boundary.pick(size=len(t.boundary.tet_verts),homogeneous=True,replacement=False)
    rng_points = rng_points.tolist()
    point = np.array(rng_points.pop(0))
    vessel, line_distances = svcco.branch_addition.close.close_exact(t.data,point)
    vessel = vessel[0]
    sampling=20
    points = svcco.branch_addition.triangle.get_local_points(t.data,vessel,point.flatten(),sampling,t.clamped_root)
    mu                 = t.parameters['mu']
    lam                = t.parameters['lambda']
    gamma              = t.parameters['gamma']
    nu                 = t.parameters['nu']
    Qterm              = t.parameters['Qterm']
    Pperm              = t.parameters['Pperm']
    Pterm              = t.parameters['Pterm']
    data = deepcopy(t.data)
    data = np.vstack((data,np.zeros((2,data.shape[1]))))
    start = perf_counter()
    svcco.branch_addition.local_func_v7.fast_local_function(data,points,point.flatten(),vessel,gamma,nu,Qterm,Pperm,Pterm)
    elapsed = perf_counter() - start
    return elapsed

def test_bif(n_steps=100,step=1):
    t = svcco.tree()
    t.set_boundary(s)
    t.set_root()
    t.convex = True
    t.n_add(1)
    SIZE       = []
    TIME_NAIVE = []
    TIME_BIND  = []
    t.rng_points,_ = t.boundary.pick(size=40*n_steps,homogeneous=True)
    t.rng_points = t.rng_points.tolist()
    for i in range(n_steps):
        SIZE.append(t.parameters['edge_num'])
        #start = perf_counter()
        #t.add(-1,0)
        #elapsed = perf_counter() - start
        TIME_BIND.append(test_binding(t))
        TIME_NAIVE.append(test_naive(t))
        t.add(-1,0)
        #TIME_BIND.append(test_binding(t))
        #t.n_add(step)
    #t.show()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    #SIZE.insert(0,1)
    #TIME_BIND.insert(0,TIME_BIND[0]/SIZE[1])
    #TIME_NAIVE.insert(0,TIME_NAIVE[0]/SIZE[1])
    SIZE = np.array(SIZE)
    TIME_BIND = np.array(TIME_BIND)
    TIME_NAIVE = np.array(TIME_NAIVE)
    #ax.loglog(SIZE,(TIME_BIND[0]/SIZE[0])*SIZE**2*np.log(SIZE),label="Literature Implementation")
    ax.loglog(SIZE,np.cumsum(TIME_BIND),label="Partial Binding")
    ax.loglog(SIZE,np.cumsum(TIME_NAIVE),label="Naive Scaling")
    ax.loglog(SIZE,(TIME_BIND[0]/SIZE[0])*SIZE*np.log(SIZE),label="Guy Floor Scaling")
    #ax.loglog(SIZE,(TIME_BIND[0]/SIZE[0])*np.log(SIZE),label="Log(N) Scaling")
    ax.loglog(SIZE,(TIME_BIND[0]/SIZE[0])*SIZE**2,label="Quadratic Scaling")
    #ax.semilogy(np.log(SIZE),(TIME_NAIVE[0]/3)*SIZE**(4/3),label="Predicted Scaling")
    ax.legend(loc='upper left')
    plt.show()
