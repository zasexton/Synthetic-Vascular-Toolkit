# File for generating vascular networks for
# cube, cylinder, heart, and gyrus

import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import tqdm

###########################################
# Code for Building Surfaces
###########################################
q = 4
resolution = 120


cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=q,resolution=resolution)
print('cube constructed')
"""
heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
print('heart constructed')

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
cylinder = svcco.surface()
cylinder.set_data(cyl.points,cyl.point_normals)
cylinder.solve()
cylinder.build(q=4,k=2,resolution=120)
print('cylinder constructed')

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
gyrus.build(q=4,k=2,resolution=120,buffer=5)
print('gyrus constructed')
"""
############################################
# Generate Vessel Networks for Domains
############################################

vessel_number = 1000

cube_tree = svcco.tree()
cube_tree.set_boundary(cube)
cube_tree.convex = True
cube_tree.set_root()
cube_tree.n_add(vessel_number)

cube_tree_models = cube_tree.show(show=False)
cube_tree_plot   = pv.Plotter()
cube_tree_plot.add_mesh(cube_tree.boundary.pv_polydata,opacity=0.5,color='red')
for model in cube_tree_models:
    tmp = model.GetOutput()
    data  = pv.PolyData(tmp)
    cube_tree_plot.add_mesh(data,color='red')
cube_tree_plot.set_background(color='white')
cube_tree_plot.camera_position = 'yz'
cube_tree_plot.camera.azimuth = 45
cube_tree_plot.save_graphic('cube_tree_vessels{}_q{}_res{}.svg'.format(vessel_number,q,resolution))

# Export SVG of Tree and surface

cylinder_tree = svcco.tree()
cylinder_tree.set_boundary(cylinder)
cylinder_tree.set_root()
cylinder_tree.n_add(1000)

# Export SVG of Tree and surface

heart_tree = svcco.tree()
heart_tree.set_boundary(heart)
heart_tree.set_root()
heart_tree.n_add(1000)

# Export SVG of Tree and surface

gyrus_tree = svcco.tree()
gyrus_tree.set_boundary(gyrus)
gyrus_tree.set_root()
gyrus_tree.n_add(1000)

# Export SVG of Tree and surface
