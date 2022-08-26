import svcco
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build()

# Warm up
t = svcco.tree()
t.convex = True
t.set_boundary(s)
t.set_root()
t.n_add(20)

#t.export(gui=False)
