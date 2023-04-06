import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import trange

q = 2
resolution = 120

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
print('anulus cfd')

t = svcco.tree()
t.set_boundary(cylinder)
t.set_root()
t.n_add(1000)

t.export_0d_simulation()
