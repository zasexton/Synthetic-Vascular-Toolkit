##########
#UNIT TEST - COLLIDING CYLINDERS (SPHERE PROXIMITY)
##########

import numpy as np
from sphere_proximity import sphere_proximity

point_0 = np.array([0,0,-1])
point_1 = np.array([0,0,1])
point_2 = np.array([-1,0,0])
point_3 = np.array([1,0,0])

r0 = 0.1
r1 = 0.1

DATA = np.zeros((1,31))
EDGE = np.zeros(31)

l0 = np.linalg.norm(point_1-point_0)
l1 = np.linalg.norm(point_3-point_2)

DATA[0,0:3] = point_0
DATA[0,3:6] = point_1
EDGE[0:3] = point_2
EDGE[3:6] = point_3

DATA[0,20]  = l0
EDGE[20]    = l1

result = sphere_proximity(DATA,EDGE)

assert len(result) == 1, "Sphere proximity did not catch collision"
