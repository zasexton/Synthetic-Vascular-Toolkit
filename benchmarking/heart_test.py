import svcco
import numpy as np

pts = np.genfromtxt('D:\\Tree\\Tree_7-0\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
n =  np.genfromtxt('D:\\Tree\\Tree_7-0\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')

s = svcco.surface()
s.set_data(pts,n)
s.solve()
s.build(q=4,resolution=120,buffer=2)

t = svcco.tree()
t.set_boundary(s)
t.set_root()
t.n_add(1000)
