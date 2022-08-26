import svcco
import numpy as np

pts = np.genfromtxt('D:\\Tree\\Tree_7-0\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
n =  np.genfromtxt('D:\\Tree\\Tree_7-0\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')

s = svcco.surface()
s.set_data(pts,normals=n)
s.solve()
s.build(q=2,resolution=40,k=20,buffer=2)
