import svcco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
from tqdm import tqdm, trange
import tetgen
from scipy import stats
import pymeshfix
import pickle
import os
from scipy.spatial import cKDTree
from time import perf_counter

q = 4
resolution = 120

def get_cube():
    cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
    cube = svcco.surface()
    cube.set_data(cu.points,cu.point_normals)
    cube.solve()
    cube.build(q=q,resolution=60)
    print('cube constructed')
    return cube

def get_heart():
    heart = svcco.surface()
    heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
    heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
    heart.set_data(heart_points,heart_normals)
    heart.solve()
    heart.build(q=q,resolution=60,k=2,buffer=5)
    print('heart constructed')
    return heart

def get_disk():
    disk = pv.Disc(inner=2.8,outer=4,r_res=20,c_res=100)
    cyl  = disk.extrude([0,0,2],capping=True).triangulate()
    cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
    cyl  = cyl.subdivide(2)
    cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
    cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
    cylinder = svcco.surface()
    cylinder.set_data(cyl.points,cyl.point_normals)
    cylinder.solve()
    cylinder.build(q=q,resolution=60)
    print('cylinder constructed')
    return cylinder

def get_gyrus():
    left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
    gyrus_no_scale = pv.read(left_gyrus)
    heart = get_heart()
    sf = (heart.volume/gyrus_no_scale.volume)**(1/3)
    gyrus_scaled = gyrus_no_scale.scale([sf,sf,sf])
    left_gyrus_scaled = "left_gyrus_scaled.vtp"
    gyrus_scaled.save(left_gyrus_scaled)
    gyrus = svcco.surface()
    gyrus.load(left_gyrus_scaled)
    gyrus.solve()
    gyrus.build(q=q,resolution=60,buffer=5)
    print('gyrus constructed')
    return gyrus


################################################################################
# Time Analysis
################################################################################
def time_analysis(boundary,n_trees,tree_size,outdir,save_trees=True):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    times = np.zeros(n_trees)
    for n in range(n_trees):
        t = svcco.tree()
        t.set_boundary(boundary)
        t.set_root()
        start = perf_counter()
        t.n_add(tree_size)
        end   = perf_counter()
        elapsed = end-start
        times[n] = elapsed
        np.save(outdir+os.sep+'times',times)
        if save_trees:
            t.save(outdir+os.sep+'tree_{}'.format(n))
