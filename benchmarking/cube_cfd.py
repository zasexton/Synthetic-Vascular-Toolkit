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

cu = pv.Cube(x_length=3.7,y_length=3.7,z_length=3.7).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=q,resolution=resolution)
print('cube constructed')

############################################
# Tree Construction
############################################

t = svcco.tree()
t.set_boundary(cube)
#t.set_parameters(Pperm=10,Pterm=9)
#t.parameters['Qterm'] = 0.07
t.convex = True
t.set_root()
t.n_add(1000)
#t.export(gui=False,global_edge_size=0.1)

t.export_0d_simulation()
