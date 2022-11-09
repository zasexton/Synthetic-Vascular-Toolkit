import numpy as np
import numba as nb
import math

@nb.jit(nopython=True,nogil=True)
def close_binary_vectorize(data,point):
    best_dist = np.inf
    best      = -1.0
    print(len(data))
    for vessel in data:
        line_direction = vessel[12:15].reshape(-1,1)
        diff = vessel[0:3]-point
        ss = np.dot(diff,line_direction)
        tt = np.dot(point-vessel[3:6],line_direction)
        hh = max([ss[0].item(),tt[0].item(),0])
        diff2 = (point-vessel[0:3]) #.reshape(-1,1)
        line_direction2 = line_direction.T
        cc = np.cross(diff2,line_direction2)
        cd = np.linalg.norm(cc)
        ld = math.hypot(hh,cd).item()
        if ld < best_dist:
            best_dist = ld
            best = vessel[-1]
    return best,best_dist

@nb.jit(nopython=True,nogil=True)
def close_binary_vectorize2(data,point):
    best_dist = np.inf
    best      = -1.0
    for vessel in data:
        diff = [0,0,0]
        diff2 = [0,0,0]
        diff3 = [0,0,0]
        cc = [0,0,0]
        line_direction = vessel[12:15]
        diff[0] = vessel[0]-point[0]
        diff[1] = vessel[1]-point[1]
        diff[2] = vessel[2]-point[2]
        ss = diff[0]*line_direction[0]+diff[1]*line_direction[1]+diff[2]*line_direction[2]
        #ss = np.dot(diff,line_direction)
        diff2[0] = point[0]-vessel[3]
        diff2[1] = point[1]-vessel[4]
        diff2[2] = point[2]-vessel[5]
        #tt = np.dot(point-vessel[3:6],line_direction)
        tt = diff2[0]*line_direction[0]+diff2[1]*line_direction[1]+diff2[2]*line_direction[2]
        #hh = max([ss,tt,0])
        if ss > tt and ss > 0:
            hh = ss
        elif tt > ss and tt>0:
            hh=tt
        else:
            hh=0.0
        #diff2 = (point-vessel[0:3]) #.reshape(-1,1)
        diff3[0] = point[0]-vessel[0]
        diff3[1] = point[1]-vessel[1]
        diff3[2] = point[2]-vessel[2]
        #line_direction2 = line_direction.T
        #cc = np.cross(diff2,line_direction2)
        cc[0] = diff3[1]*line_direction[2] - diff3[2]*line_direction[1]
        cc[1] = diff3[2]*line_direction[0] - diff3[0]*line_direction[2]
        cc[2] = diff3[0]*line_direction[1] - diff3[1]*line_direction[0]
        #cd = np.linalg.norm(cc)
        cd = (cc[0]**2+cc[1]*2+cc[2]**2)**(1/2)
        #ld = math.hypot(hh,cd).item()
        ld = (hh**2+cd**2)**(1/2)
        if ld < best_dist:
            best_dist = ld
            best = vessel[-1]
    return best,best_dist
##########
# TESTING
##########

def close_exact(data,point):
    line_direction = data[:,12:15]
    ss = np.array([np.dot(data[i,0:3] - point,line_direction[i,:]) for i in range(data.shape[0])])
    tt = np.array([np.dot(point - data[i,3:6],line_direction[i,:]) for i in range(data.shape[0])])
    decision = [[ss[i],tt[i],0] for i in range(len(ss))]
    hh = np.array([np.max(np.array(i)) for i in decision])
    cc = np.cross((point - data[:,0:3]),line_direction,axis=1)
    cd = np.linalg.norm(cc,axis=1)
    line_distances = np.hypot(hh,cd)
    vessel = np.argsort(line_distances)
    return vessel, line_distances

data = np.array([[ 5.08243631e+00, -4.46177086e+00,  2.60644315e+00,
         1.98152091e+00, -2.72606725e+00,  7.54540997e-01,
        -1.13326408e-01,  6.23172325e-01,  7.73830330e-01,
         6.23172325e-01,  6.51186082e-01, -4.33143094e-01,
        -7.73830330e-01,  4.33143094e-01, -4.62140326e-01,
         2.00000000e+00,  1.00000000e+00, -1.00000000e+00,
         0.00000000e+00,  2.00000000e+00,  4.00722908e+00,
         1.79664092e-02,  4.16666667e-03,  8.23460155e-01,
         7.61524342e-01,  1.33357732e+00,  0.00000000e+00,
         1.06835182e+01,  1.00000000e+00, -1.00000000e+00,
         0.00000000e+00],
       [ 1.98152091e+00, -2.72606725e+00,  7.54540997e-01,
        -4.99766417e+00, -3.41363795e+00, -2.82500656e-01,
        -1.35265522e-01, -1.11843331e-01,  9.84476667e-01,
        -1.11843331e-01,  9.88981493e-01,  9.69880150e-02,
        -9.84476667e-01, -9.69880150e-02, -1.46284028e-01,
        -1.00000000e+00, -1.00000000e+00,  0.00000000e+00,
         2.00000000e+00,  3.00000000e+00,  7.08923362e+00,
         1.36818579e-02,  2.08333333e-03,  0.00000000e+00,
         0.00000000e+00,  6.49893066e-01,  1.00000000e+00,
         0.00000000e+00,  7.61524342e-01, -1.00000000e+00,
         1.00000000e+00],
       [ 1.98152091e+00, -2.72606725e+00,  7.54540997e-01,
        -8.06610736e-01,  4.65374414e+00, -4.87650598e+00,
         8.02524244e-01,  5.22691901e-01,  2.87659544e-01,
         5.22691901e-01, -3.83495523e-01, -7.61396322e-01,
        -2.87659544e-01,  7.61396322e-01, -5.80971279e-01,
        -1.00000000e+00, -1.00000000e+00,  0.00000000e+00,
         2.00000000e+00,  1.00000000e+00,  9.69247050e+00,
         1.47946221e-02,  2.08333333e-03,  0.00000000e+00,
         0.00000000e+00,  8.88540244e-01,  1.00000000e+00,
         0.00000000e+00,  8.23460155e-01, -1.00000000e+00,
         2.00000000e+00]])

point = np.array([0.2,0.2,0.2]).reshape(1,-1)
bin_data = []
for i in range(data.shape[0]):
    bin_data.append(data[i,:].tolist())
bin_data = tuple(bin_data)
point = point.tolist()
point = tuple(point[0])
print(point)
print(close_binary_vectorize2(bin_data,point))
