# Reconstructed Implicit Surfaces
import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tetgen

###################################
# Setting up surface reconstruction
###################################

cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(3)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=2,resolution=30)
cube.pv_polydata_surf.save('cube.vtp')

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
heart.pv_polydata_surf.save('heart.vtp')

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
cylinder = svcco.surface()
cylinder.set_data(cyl.points,cyl.point_normals)
cylinder.solve()
cylinder.build(q=10,k=2,resolution=25,method='flying_edges')
cylinder.pv_polydata_surf.save('cylinder.vtp')


left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
gyrus.build(q=4,resolution=40,buffer=5)


refined_cube = cube.pv_polydata.subdivide(2)
tet = tetgen.TetGen(refined_cube)
total_patches = len(cube.patches)
surface_pts = cube.pv_polydata.points
surface_pts = [pt.tolist() for pt in surface_pts]
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
pts = mesh.points
for pt in pts:
    data = pt.tolist()
    data.append(total_patches//4)
    values.append(cube.DD[0](data)[0])
values = np.array(values)
mesh['implicit_data'] = values
mesh.save('cube.vtu')

refined_cyl = cylinder.pv_polydata.subdivide(2)
refined_cyl = svcco.utils.remeshing.remesh.remesh_surface(refined_cyl,hausd=0.005)
tet = tetgen.TetGen(refined_cyl)
tet.make_manifold(verbose=True)
total_patches = len(cylinder.patches)
surface_pts = cylinder.pv_polydata.points
surface_pts = [pt.tolist() for pt in surface_pts]
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
pts = mesh.points
for pt in pts:
    data = pt.tolist()
    data.append(total_patches//4)
    values.append(cylinder.DD[0](data)[0])

values = np.array(values)
mesh['implicit_data'] = values
mesh.save('cylinder.vtu')

refined_heart = heart.pv_polydata
refined_heart = svcco.utils.remeshing.remesh.remesh_surface(refined_heart,hausd=0.005)
tet = tetgen.TetGen(refined_heart)
tet.make_manifold(verbose=True)
total_patches = len(heart.patches)
surface_pts = heart.pv_polydata.points
surface_pts = [pt.tolist() for pt in surface_pts]
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
pts = mesh.points
for pt in pts:
    data = pt.tolist()
    data.append(total_patches//4)
    values.append(heart.DD[0](data)[0])

values = np.array(values)
mesh['implicit_data'] = values
mesh.save('cylinder.vtu')

refined_gyrus = gyrus.pv_polydata
refined_gyrus = svcco.utils.remeshing.remesh.remesh_surface(refined_gyrus,hausd=0.005)
tet = tetgen.TetGen(refined_gyrus)
tet.make_manifold(verbose=True)
total_patches = len(gyrus.patches)
surface_pts = gyrus.pv_polydata.points
surface_pts = [pt.tolist() for pt in surface_pts]
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
pts = mesh.points
for pt in pts:
    data = pt.tolist()
    data.append(total_patches//4)
    values.append(gyrus.DD[0](data)[0])

values = np.array(values)
mesh['implicit_data'] = values
mesh.save('gyrus.vtu')
"""

"""

# Convex shapes
"""
sphere = pv.Sphere().triangulate().subdivide(2)

surf = svcco.surface()
surf.set_data(sphere.points,normals=sphere.point_normals)
surf.solve()
surf.build()

refined = sphere.subdivide(2)
tet = tetgen.TetGen(refined)
total_patches = len(surf.patches)
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
centers = mesh.cell_centers()

for i in tqdm.tqdm(range(centers.n_cells)):
    data = centers.cell_points(i).flatten().tolist()
    data.append(total_patches)
    values.append(surf.DD[0](data))

values = np.array(values)
mesh.cell_data['implicit_data'] = values
mesh.save('sphere.vtu')
"""

disc = pv.Disc(r_res=10,c_res=20)
cyl  = disc.extrude([0,0,1],capping=True)
cyl.triangulate(inplace=True)
cyl = svcco.utils.remeshing.remesh.remesh_surface(cyl)

surf = svcco.surface()
surf.set_data(cyl.points,normals=cyl.point_normals)
surf.solve()
surf.build()

refined = cyl.subdivide(2)
tet = tetgen.TetGen(refined)
total_patches = len(surf.patches)
values = []
tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
mesh = tet.grid
centers = mesh.cell_centers()

for i in tqdm.tqdm(range(centers.n_cells)):
    data = centers.cell_points(i).flatten().tolist()
    data.append(total_patches)
    values.append(surf.DD[0](data))

values = np.array(values)
mesh.cell_data['implicit_data'] = values
mesh.save('cylinder.vtu')
