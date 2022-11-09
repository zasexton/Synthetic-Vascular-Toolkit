import svcco
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(5)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build(q=4,resolution=25)

# Warm up
t = svcco.tree()
t.convex = True
t.set_boundary(s)
t.set_root()
t.n_add(10)

#t.export(gui=False)
