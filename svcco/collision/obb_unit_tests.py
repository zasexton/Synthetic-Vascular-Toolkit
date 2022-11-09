##########
#UNIT TEST - COLLIDING CYLINDERS (OBB PROXIMITY)
##########

import numpy as np
from .obb import obb

############################################
# Test 1: Perpendicular Intersecting Vessels
############################################

point_0 = np.array([0,0,-1])
point_1 = np.array([0,0,1])
point_2 = np.array([-1,0,0])
point_3 = np.array([1,0,0])

r0 = 0.2
r1 = 0.2

DATA = np.zeros((1,31))
EDGE = np.zeros(31)

l0 = np.linalg.norm(point_1-point_0)
l1 = np.linalg.norm(point_3-point_2)

DATA[0,0:3] = point_0
DATA[0,3:6] = point_1
EDGE[0:3] = point_2
EDGE[3:6] = point_3

DATA[0,12:15] = ((DATA[0,3:6] - DATA[0,0:3]) /
               np.linalg.norm(DATA[0,3:6] -
                              DATA[0,0:3]))
if DATA[0,14] == -1:
    DATA[0,6:9] = np.array([-1,0,0])
    DATA[0,9:12] = np.array([0,-1,0])
else:
    DATA[0,6:9] = np.array([1-DATA[0,12]**2/(1+DATA[0,14]),
                          (-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                          -DATA[0,12]])
    DATA[0,9:12] = np.array([(-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                           1 - DATA[0,13]**2/(1+DATA[0,14]),
                           -DATA[0,13]])

DATA[0,20]  = l0
EDGE[20]    = l1

DATA[0,21]  = r0
EDGE[21]    = r1
result = obb(DATA,EDGE)

assert result == True, "Obb did not catch collision"
print('PASS Test 1')

############################################
# Test 2: Parallel Intersecting Vessels
############################################

point_0 = np.array([0,0,-1])
point_1 = np.array([0,0,1])
point_2 = np.array([0,0.1,-1])
point_3 = np.array([0,0.1,1])

r0 = 0.2
r1 = 0.2

DATA = np.zeros((1,31))
EDGE = np.zeros(31)

l0 = np.linalg.norm(point_1-point_0)
l1 = np.linalg.norm(point_3-point_2)

DATA[0,0:3] = point_0
DATA[0,3:6] = point_1
EDGE[0:3] = point_2
EDGE[3:6] = point_3

DATA[0,12:15] = ((DATA[0,3:6] - DATA[0,0:3]) /
               np.linalg.norm(DATA[0,3:6] -
                              DATA[0,0:3]))
if DATA[0,14] == -1:
    DATA[0,6:9] = np.array([-1,0,0])
    DATA[0,9:12] = np.array([0,-1,0])
else:
    DATA[0,6:9] = np.array([1-DATA[0,12]**2/(1+DATA[0,14]),
                          (-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                          -DATA[0,12]])
    DATA[0,9:12] = np.array([(-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                           1 - DATA[0,13]**2/(1+DATA[0,14]),
                           -DATA[0,13]])

DATA[0,20]  = l0
EDGE[20]    = l1

DATA[0,21]  = r0
EDGE[21]    = r1
result = obb(DATA,EDGE)

assert result == True, "Obb did not catch collision"
print('PASS Test 2')

############################################
# Test 3: Parallel Intersecting Vessels
############################################

point_0 = np.array([0,0,-1])
point_1 = np.array([0,0,-0.5])
point_2 = np.array([0,0,1])
point_3 = np.array([0,0,0.5])

r0 = 0.2
r1 = 0.2

DATA = np.zeros((1,31))
EDGE = np.zeros(31)

l0 = np.linalg.norm(point_1-point_0)
l1 = np.linalg.norm(point_3-point_2)

DATA[0,0:3] = point_0
DATA[0,3:6] = point_1
EDGE[0:3] = point_2
EDGE[3:6] = point_3

DATA[0,12:15] = ((DATA[0,3:6] - DATA[0,0:3]) /
               np.linalg.norm(DATA[0,3:6] -
                              DATA[0,0:3]))
if DATA[0,14] == -1:
    DATA[0,6:9] = np.array([-1,0,0])
    DATA[0,9:12] = np.array([0,-1,0])
else:
    DATA[0,6:9] = np.array([1-DATA[0,12]**2/(1+DATA[0,14]),
                          (-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                          -DATA[0,12]])
    DATA[0,9:12] = np.array([(-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                           1 - DATA[0,13]**2/(1+DATA[0,14]),
                           -DATA[0,13]])

DATA[0,20]  = l0
EDGE[20]    = l1

DATA[0,21]  = r0
EDGE[21]    = r1
result = obb(DATA,EDGE)

assert result == False, "Obb did not catch non-collision"
print('PASS Test 3')

############################################
# Test 4: Close Parallel Intersecting Vessels
############################################

point_0 = np.array([0,0,-1])
point_1 = np.array([0,0,1.1])
point_2 = np.array([0,0,1])
point_3 = np.array([0,0,-0.2])

r0 = 0.2
r1 = 0.1

DATA = np.zeros((1,31))
EDGE = np.zeros(31)

l0 = np.linalg.norm(point_1-point_0)
l1 = np.linalg.norm(point_3-point_2)

DATA[0,0:3] = point_0
DATA[0,3:6] = point_1
EDGE[0:3] = point_2
EDGE[3:6] = point_3

DATA[0,12:15] = ((DATA[0,3:6] - DATA[0,0:3]) /
               np.linalg.norm(DATA[0,3:6] -
                              DATA[0,0:3]))
if DATA[0,14] == -1:
    DATA[0,6:9] = np.array([-1,0,0])
    DATA[0,9:12] = np.array([0,-1,0])
else:
    DATA[0,6:9] = np.array([1-DATA[0,12]**2/(1+DATA[0,14]),
                          (-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                          -DATA[0,12]])
    DATA[0,9:12] = np.array([(-DATA[0,12]*DATA[0,13])/(1+DATA[0,14]),
                           1 - DATA[0,13]**2/(1+DATA[0,14]),
                           -DATA[0,13]])

DATA[0,20]  = l0
EDGE[20]    = l1

DATA[0,21]  = r0
EDGE[21]    = r1
result = obb(DATA,EDGE)

assert result == True, "Obb did not catch collision"
print('PASS Test 4')
