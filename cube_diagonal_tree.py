# File for printing a vascular tree that begins
# in the corner of a cube with an initial direction
# of growth toward the center of the volume

import svcco
import pyvista as pv
import numpy as np
# Assuming cm units

cu = pv.Cube(center=[1,1,1],x_length=2,y_length=2,z_length=2).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=4,resolution=50)

t = svcco.tree()
t.set_boundary(cube)
t.convex = True
direction = np.array([1,1,1])/np.linalg.norm(np.array([1,1,1]))
t.set_root(start=[0,0,0],direction=direction)
t.n_add(10)
