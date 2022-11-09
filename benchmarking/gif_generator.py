# File for building gifs of vascular construction

import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import trange

q = 4
resolution = 25


cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=q,resolution=resolution)

f = svcco.forest(boundary=cube,convex=True,number_of_networks=4,trees_per_network=[1,1,1,1],
                 compete=True)
f.set_roots()
if not os.path.exists('gif_folder_4_tree_compete'):
    os.mkdir('gif_folder_4_tree_compete')
for i in trange(200):
    f.add(1)
    p = f.show(off_screen=True)
    p.screenshot(os.getcwd()+os.sep+'gif_folder_4_tree_compete'+os.sep+'forest_4_trees_compete_iter_{}.png'.format(i))
    p.close()
