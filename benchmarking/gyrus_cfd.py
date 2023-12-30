import svcco
import pyvista as pv
import numpy as np

q = 4
resolution = 120
"""
heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
print('heart constructed')
"""
#left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
#gyrus_no_scale = pv.read(left_gyrus)
#sf = (heart.volume/gyrus_no_scale.volume)**(1/3)
#gyrus_scaled = gyrus_no_scale.scale([sf,sf,sf])
left_gyrus_scaled = "left_gyrus_scaled.vtp"
gyrus_scaled = pv.read(left_gyrus_scaled)
gyrus = svcco.surface()
gyrus.load(left_gyrus_scaled)
gyrus.solve()
gyrus.build(q=q,resolution=resolution,buffer=5)
print('gyrus constructed')

# Tree for 0D Simulations
t = svcco.tree()
t.set_boundary(gyrus)
t.set_root()
t.n_add(1000)

t.export_0d_simulation()
