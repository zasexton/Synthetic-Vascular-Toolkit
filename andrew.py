import svcco
import pyvista as pv
import numpy as np

inlet_point   = np.array([0,0,-0.5])
outlet_point  = np.array([0,0,0.5])
inlet_normal  = np.array([0,0,1])
outlet_normal = np.array([0,0,-1])
cube          = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(cube.points,normals=cube.point_normals)
s.solve()
s.build()


t = svcco.forest(boundary=s,number_of_networks=1,trees_per_network=[2],convex=True,
                 start_points=[[inlet_point,outlet_point]],directions=[[inlet_normal,outlet_normal]],
                 root_lengths_low=[[0.25,0.25]],root_lengths_high=[[0.5,0.5]])

#t.set_roots()

#p = t.show()
#p.show()

t.networks[0][0].set_parameters(Pperm=5.045,Pterm=5,edge_num=1,Qterm=(0.25/21))
t.networks[0][1].set_parameters(Pperm=5.045,Pterm=5,edge_num=1,Qterm=(0.25/21))

t.set_roots()

#print(t.networks[0][0].data[0,20])

#p = t.show()
#p.show()

t.add(1)
