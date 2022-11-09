# Code for Vascular Shell Geometries Made on 11-01-2022
# for iCLIP Vascular Printing Project

# svcco version 0.5.52 (alpha release)

# NOTE: this code will produce networks of
#       similar connectivity Not topology

# NOTE: units are in centimeter-gram-seconds

import svcco
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(5)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build(q=4,resolution=25)

f = svcco.forest(boundary=s,convex=True)
f.set_roots()
f.add(2)
f.connect()

shell1 = f.forest_copy.export_solid(folder="100_micron_shell",shell=True,thickness=0.01)

shell2 = f.forest_copy.export_solid(folder="50_micron_shell",shell=True,thickness=0.005)

shell3 = f.forest_copy.export_solid(folder="25_micron_shell",shell=True,thickness=0.0025)
