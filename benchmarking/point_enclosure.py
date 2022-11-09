import svcco
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from tqdm import tqdm

cube = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(3)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build(q=2) #,resolution=30)
res_cube = svcco.implicit.visualize.visualize.plot_mpu_boundaries(s,name="cube_point_enclosure",resolution=80,test_size=10000)

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=6) #,resolution=85,k=200,buffer=5)
res_heart = svcco.implicit.visualize.visualize.plot_mpu_boundaries(heart,name="heart_point_enclosure",resolution=80,plane_axis=0,test_size=10000)

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
c = svcco.surface()
c.set_data(cyl.points,cyl.point_normals)
c.solve()
c.build(q=8) #,k=200,resolution=100)
res_cyl = svcco.implicit.visualize.visualize.plot_mpu_boundaries(c,name="cylinder_point_enclosure",resolution=80,test_size=10000)

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"

gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
gyrus.build(q=4) #,resolution=40,buffer=5)
res_gyrus = svcco.implicit.visualize.visualize.plot_mpu_boundaries(gyrus,name="gyrus_point_enclosure",resolution=80,test_size=10000)
