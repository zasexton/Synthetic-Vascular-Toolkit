import svcco
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(10*cube.points,normals=cube.point_normals)
s.solve()
s.build()

t = svcco.tree()
t.set_boundary(s)
t.set_root()
t.n_add(10)
