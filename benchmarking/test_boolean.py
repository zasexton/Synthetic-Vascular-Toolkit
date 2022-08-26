import svcco
import pyvista as pv

sph = svcco.bumpy_sphere()
cube = pv.Cube().triangulate().subdivide(3)

s1 = svcco.surface()
s2 = svcco.surface()

s1.set_data(sph)
s2.set_data(3*cube.points,normals=cube.point_normals)

s1.solve()
s2.solve()

s1.build()
s2.build()
s1.subtract(s2)

t = svcco.tree()
t.set_boundary(s1)
t.parameters['Qterm'] *= 40
t.set_root()
t.n_add(100)
t.show(surface=True)
