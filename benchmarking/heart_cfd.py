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
resolution = 120

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
print('heart constructed')


############################################
# Tree Construction
############################################

t = svcco.tree()
t.set_boundary(heart)
t.set_root()
t.n_add(1000)

t.export_0d_simulation(steady=False)
