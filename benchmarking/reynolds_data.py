# File to estimate reynolds numbers and produce visualization
# and histogram quantification

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
from tqdm import trange

###########################################
# Code for Building Surfaces
###########################################
q = 4
resolution = 40


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


disk = pv.Disc(inner=2.8,outer=4,r_res=20,c_res=100)
cyl  = disk.extrude([0,0,2],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
cylinder = svcco.surface()
cylinder.set_data(cyl.points,cyl.point_normals)
cylinder.solve()
cylinder.build(q=q,resolution=resolution)
print('cylinder constructed')

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
gyrus_no_scale = pv.read(left_gyrus)
sf = (heart.volume/gyrus_no_scale.volume)**(1/3)
gyrus_scaled = gyrus_no_scale.scale([sf,sf,sf])
left_gyrus_scaled = "left_gyrus_scaled.vtp"
gyrus_scaled.save(left_gyrus_scaled)
gyrus = svcco.surface()
gyrus.load(left_gyrus_scaled)
gyrus.solve()
gyrus.build(q=q,resolution=resolution,buffer=5)
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
#cube_tree_plot.add_mesh(cube_tree.boundary.pv_polydata,opacity=0.25,color='red')
cube_reynolds    = svcco.reynolds.reynolds(cube_tree)
for i,model in enumerate(cube_tree_models):
    tmp = model.GetOutput()
    data  = pv.PolyData(tmp)
    data['reynolds'] = cube_reynolds[i]*np.ones(data.points.shape[0])
    _ = cube_tree_plot.add_mesh(data,scalars='reynolds')
cube_tree_plot.set_background(color='white')
cube_tree_plot.camera_position = 'yz'
cube_tree_plot.camera.azimuth = 45
cube_tree_plot.save_graphic('cube_tree_reynolds_vessels{}_q{}_res{}.svg'.format(vessel_number,q,resolution))
path = cube_tree_plot.generate_orbital_path(n_points=36,shift=cube.pv_polydata.length)
cube_tree_plot.open_gif("cube_reynolds_orbit.gif")
cube_tree_plot.orbit_on_path(path,write_frames=True)
# Export SVG of Tree and surface
"""
cylinder_tree = svcco.tree()
cylinder_tree.set_boundary(cylinder)
cylinder_tree.set_root()
cylinder_tree.n_add(vessel_number)

cylinder_tree_models = cylinder_tree.show(show=False)
cylinder_tree_plot   = pv.Plotter()
cylinder_tree_plot.add_mesh(cylinder_tree.boundary.pv_polydata,opacity=0.25,color='red')
for model in cylinder_tree_models:
    tmp = model.GetOutput()
    data = pv.PolyData(tmp)
    _ = cylinder_tree_plot.add_mesh(data,color='red')
cylinder_tree_plot.set_background(color='white')
cylinder_tree_plot.camera_position = 'yz'
cylinder_tree_plot.camera.roll = 0
cylinder_tree_plot.elevation = 25
cylinder_tree_plot.camera.azimuth = 140
cylinder_tree_plot.save_graphic('cylinder_tree_vessels{}_q{}_res{}.svg'.format(vessel_number,q,resolution))
path = cylinder_tree_plot.generate_orbital_path(n_points=36,shift=cylinder.pv_polydata.length)
cylinder_tree_plot.open_gif("cylinder_orbit.gif")
cylinder_tree_plot.orbit_on_path(path,write_frames=True)
# Export SVG of Tree and surface

heart_tree = svcco.tree()
heart_tree.set_boundary(heart)
heart_tree.set_root()
heart_tree.n_add(vessel_number)

# Export SVG of Tree and surface

heart_tree_models = heart_tree.show(show=False)
heart_tree_plot   = pv.Plotter()
heart_tree_plot.add_mesh(heart_tree.boundary.pv_polydata,opacity=0.25,color='red')
for model in heart_tree_models:
    tmp = model.GetOutput()
    data = pv.PolyData(tmp)
    _ = heart_tree_plot.add_mesh(data,color='red')
heart_tree_plot.set_background(color='white')
heart_tree_plot.camera_position = 'yz'
heart_tree_plot.camera.roll = 160
heart_tree_plot.save_graphic('heart_tree_vessels{}_q{}_res{}.svg'.format(vessel_number,q,resolution))
path = heart_tree_plot.generate_orbital_path(n_points=36,shift=heart.pv_polydata.length)
heart_tree_plot.open_gif("heart_orbit.gif")
heart_tree_plot.orbit_on_path(path,write_frames=True)


gyrus_tree = svcco.tree()
gyrus_tree.set_boundary(gyrus)
gyrus_tree.set_root()
gyrus_tree.n_add(vessel_number)

# Export SVG of Tree and surface

gyrus_tree_models = gyrus_tree.show(show=False)
gyrus_tree_plot   = pv.Plotter()
gyrus_tree_plot.add_mesh(gyrus_tree.boundary.pv_polydata,opacity=0.25,color='red')
for model in gyrus_tree_models:
    tmp = model.GetOutput()
    data = pv.PolyData(tmp)
    _ = gyrus_tree_plot.add_mesh(data,color='red')
gyrus_tree_plot.set_background(color='white')
gyrus_tree_plot.camera_position = 'yz'
gyrus_tree_plot.camera.azimuth = 45
gyrus_tree_plot.save_graphic('gyrus_tree_vessels{}_q{}_res{}.svg'.format(vessel_number,q,resolution))
path = gyrus_tree_plot.generate_orbital_path(n_points=36,shift=gyrus.pv_polydata.length)
gyrus_tree_plot.open_gif("gyrus_orbit.gif")
gyrus_tree_plot.orbit_on_path(path,write_frames=True)
"""
