# Code for result figure 1
#
# What are we trying to show?
# --Build time for models
# --The models themselves (connected but not lofted)
# --Diameters versus depth

import svcco
from copy import deepcopy
import numpy as np
import os
import pickle
import time
import pyvista as pv
cube = pv.Cube().triangulate().subdivide(3)

A = [1]#,2,3,4,5]
B = [1]#,1,1,1,1]
S = []
C = []
for i in range(len(A)):
    if i == 4:
        sph = svcco.bumpy_sphere(samples=35,scale=3,a=A[i],b=B[i])
    else:
        sph = svcco.bumpy_sphere(samples=20,scale=3,a=A[i],b=B[i])
    s = svcco.surface()
    #s.set_data(np.array(10*cube.points),normals=cube.point_normals)
    s.set_data(sph)
    s.solve(regularize=True)
    #s.solve()
    s.build()
    #svcco.plot_volume(s)
    #print(s.DD[0]((s.x_range[1]+0.1,s.y_range[1]+0.1,s.z_range[1]+0.1,len(s.patches))))
    if s.DD[0]((s.x_range[1]+0.1,s.y_range[1]+0.1,s.z_range[1]+0.1,len(s.patches))) < 0:
        s.normals = -s.normals
        s.solve(regularize=True)
        #s.solve()
        s.build()
        #svcco.plot_volume(s)
    if i == 0:
        parent_convexity = s.surface_area/s.volume
        convexity = 1
    else:
        convexity = (s.surface_area/s.volume)/parent_convexity
    S.append(s)
    C.append(convexity)

s = svcco.surface()
s.set_data(np.array(10*cube.points),normals=cube.point_normals)
s.solve()
s.build()
if s.DD[0]((s.x_range[1]+0.1,s.y_range[1]+0.1,s.z_range[1]+0.1,len(s.patches))) < 0:
    s.normals = -s.normals
    s.solve()
    s.build()
S.append(s)
C.append(1)

V = [100,900,1000]#,2000]
repeats = 2
DATA = []
T = []
t = svcco.tree()
t.set_boundary(S[0])
t.set_root()
t.n_add(10)
os.mkdir(os.getcwd()+os.sep+'figure1')
os.chdir(os.getcwd()+os.sep+'figure1')
for n,surf in enumerate(S):
    s_data = []
    t_data = []
    for j in range(repeats):
        tree_data = []
        time_data = []
        t = svcco.tree()
        t.set_boundary(surf)
        t.set_root()
        for i in range(len(V)):
            start = time.time()
            t.n_add(V[i])
            end = time.time() - start
            time_data.append(end)
            tree_data.append(deepcopy(t.data))
            t.show(surface=True,save=True,name='{}_{}_{}'.format('S'+str(n)+str(1),str(j),str(V[i])))
        s_data.append(tree_data)
        t_data.append(time_data)
    DATA.append(s_data)
    T.append(t_data)
w = open(os.getcwd()+os.sep+'figure_1_data.pkl','wb+')
pickle.dump(DATA,w)
w.close()
wc = open(os.getcwd()+os.sep+'convexity.pkl','wb+')
pickle.dump(C,wc)
wc.close()
wt = open(os.getcwd()+os.sep+'time.pkl','wb+')
pickle.dump(T,wt)
