# File to test tangent constrained splines

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from mpl_toolkits.mplot3d import Axes3D
import numba as nb
import nlopt
"""
def test():
    P1 = np.random.random(3)
    P2 = np.random.random(3)
    V1 = P2-P1
    # Tangent Constraint 1 for First Spline
    V1 = V1/np.linalg.norm(V1)
    P3 = np.random.random(3)
    P4 = np.random.random(3)
    V2 = P4-P3
    # Tangent Constraint 1 for Second Spline
    V2 = V2/np.linalg.norm(V2)
    #MID = (P2 + P4)/2
    #V3  = MID - P2
    # Tangent Constraint 2 for First Spline
    V3  = V3/np.linalg.norm(V3)
    #V4  = MID - P4
    # Tangent Constraint 2 for Second Spline
    V4  = V4/np.linalg.norm(V4)
    CTR0 = P2 + V1*0.01
    CTR1 = P2 + V1*0.1
    CTR2 = P4 - V3*0.1
    #CTR3 = P4 - V3*0.01
    #CTR4 = MID + V3*0.25
    #CTR3 = P4 + V2*0.1
    # Create Cubic Spline 1
    SPT1 = np.vstack((P2,CTR0,CTR1,CTR2,CTR3,MID))
    tck1, u = interpolate.splprep([SPT1[:,0],SPT1[:,1],SPT1[:,2]],s=0,k=3,nest=-1)
    tck2, u = interpolate.splprep([SPT1[:,0],SPT1[:,1],SPT1[:,2]],s=0,k=3)
    l, r = [(1, (0, 0, 0))], [(1, (0, 0, 0))]
    clamped_spline = interpolate.make_interp_spline(u,SPT1,bc_type=(l,r))
    spline_f1 = lambda t: np.array(interpolate.splev(t,tck=tck1,der=0))
    spline_df1 = lambda t: np.array(interpolate.splev(t,tck=tck1,der=1)).flatten()/np.linalg.norm(np.array(interpolate.splev(t,tck=tck1,der=1)).flatten())
    print('True Vector Inlet:   {}'.format(V1))
    print('Spline Vector Inlet: {}'.format(spline_df1(0)))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    tt = np.linspace(0,1,num=400)
    spline1 = spline_f1(tt).T
    spline2 = clamped_spline(tt)
    ax.plot(spline1[:,0],spline1[:,1],spline1[:,2],color='blue',label='unclamped')
    ax.plot(spline2[:,0],spline2[:,1],spline2[:,2],color='red',label='clamped')
    ax.plot([P1[0],P2[0]],[P1[1],P2[1]],[P1[2],P2[2]],color='green',label='inlet')
    ax.plot([MID[0],CTR4[0]],[MID[1],CTR4[1]],[MID[2],CTR4[2]],color='green',label='outlet')
    ax.scatter(SPT1[:,0],SPT1[:,1],SPT1[:,2],color='black',label='data')
    ax.legend()
    plt.show()

@nb.jit(nopython=True)
def get_angle(V1,V2):
    return np.arccos(np.dot(-V1,V2))*(180/np.pi)


@nb.jit(nopython=True)
def get_vecs(vec):
    angles = np.zeros(vec.shape[0]-1)
    for i in range(vec.shape[0]-1):
        value = get_angle(vec[i,:],vec[i+1,:])
        angles[i] = 1/value
    return angles


def curvature(t,df=None,ddf=None):
    numerator = np.cross(df(t),ddf(t))
    kappa = np.linalg.norm(numerator)/np.linalg.norm(df(t))**3
    return kappa

collision_vessels = np.array([[0,0.5,0.25,1,0.5,0.25,0.01],
                              [0,0.7,0.25,1,0.7,0.25,0.01],
                              [0,0.3,0.25,1,0.3,0.25,0.01],
                              [0,0.9,0.25,1,0.9,0.25,0.01],
                              [0,0.1,0.25,1,0.1,0.25,0.01],
                              [0.25,0,0.25,0.25,1,0.25,0.01],
                              [0.75,0,0.25,0.75,1,0.25,0.01],
                              [0,0.5,0.75,1,0.5,0.75,0.01],
                              [0,0.7,0.75,1,0.7,0.75,0.01],
                              [0,0.3,0.75,1,0.3,0.75,0.01],
                              [0,0.9,0.75,1,0.9,0.75,0.01],
                              [0,0.1,0.75,1,0.1,0.75,0.01],
                              [0.25,0,0.75,0.25,1,0.75,0.01],
                              [0.75,0,0.75,0.75,1,0.75,0.01],
                              [1,0,0,1,0,1,0.1]])


pt = np.array([0,0,0])
pts= np.random.random((3,3))

from math import tanh

def get_radius(curve):
    num = curve.sample_size
    rad = []
    for i in np.linspace(0,1,num=num):
        _,d,dd = curve.derivatives(i,2)
        d  = np.array(d)
        dd = np.array(dd)
        value = np.linalg.norm(d)**3/(np.linalg.norm(np.cross(d,dd))+np.finfo(float).eps)
        rad.append(value)
    return rad


@nb.jit(nopython=True)
def get_collisions(collision_vessels,R,sample_pts):
    collisions = 0
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:])
        if np.any(dist<R):
            collisions += 10e8*tanh((R - min(dist))*(1/10))
    return collisions


@nb.jit(nopython=True)
def are_collisions(collision_vessels,R,sample_pts):
    collisions = False
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:])
        if np.any(dist<R):
            collisions = True
            break
    return collisions


@nb.jit(nopython=True)
def close_exact(data,point):
    line_direction = np.zeros((data.shape[0],3))
    ss = np.zeros(data.shape[0])
    tt = np.zeros(data.shape[0])
    hh = np.zeros(data.shape[0])
    cc = np.zeros((data.shape[0],3))
    cd = np.zeros(data.shape[0])
    line_distances = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        line_direction[i,:] = (data[i,3:6] - data[i,0:3])/np.linalg.norm(data[i,3:6] - data[i,0:3])
        ss[i] = np.dot(data[i,0:3]-point,line_direction[i,:])
        tt[i] = np.dot(point-data[i,3:6],line_direction[i,:])
        d = np.array([ss[i],tt[i],0])
        hh[i] = np.max(d)
        diff = point - data[i,0:3]
        cc[i,:] = np.cross(diff,line_direction[i,:])
        cd[i] = np.linalg.norm(cc[i,:])
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2) - data[i,6]
    return line_distances

def connect_deterministic(P1,P2,P3,P4,R,collision_vessels):
    MID = (P2 + P4)/2
    P2_P4_to_MID_dist = np.linalg.norm(P2 - MID)
    V1 = P2 - P1
    V1 = V1/np.linalg.norm(V1)
    V2 = P4 - P3
    V2 = V2/np.linalg.norm(V2)
    success = False
    points = [P2]
    vectors = [V1]
    while not success:
        # Check direct connection
        check = P4 - points[-1]
        check = check/np.linalg.norm(check)
        vectors_check = vectors
        vectors_check.append(check)
        vectors_check.append(V2)
        vectors_check = np.array(vectors_check)


from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL


def connect_bezier(P1,P2,P3,P4,R,collision_vessels,degree=2):
    V1   = (P2 - P1)
    V1   = V1/np.linalg.norm(V1)
    V2   = (P4 - P3)
    V2   = V2/np.linalg.norm(V2)
    def create_bezier(data,V1=V1,V2=V2,R=R,P2=P2,P4=P4,degree=degree):
        CTR0 = P2 + (R+data[0])*V1
        CTR1 = P4 + (R+data[1])*V2
        CTR  = np.zeros(data.shape[0]-2+12)
        fill_CTR = np.zeros(data.shape[0]-2).reshape(-1,3)
        seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
        for i in range(fill_CTR.shape[0]):
            fill_CTR[i,:] = CTR0*(1-seg[i]) + CTR1*(seg[i]) #V1*(1-seg[i])*R + CTR1*(seg[i]) + V2*(seg[i])*R
        fill_CTR = fill_CTR.flatten()
        CTR[0:3]     = P2
        CTR[3:6]     = CTR0
        CTR[-6:-3]   = CTR1
        CTR[-3:]     = P4
        CTR[6:-6]    += fill_CTR + data[2:]
        CTR  = CTR.reshape(-1,3).tolist()
        curve = BSpline.Curve()
        curve.degree = degree
        curve.ctrlpts = CTR
        curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
        curve.sample_size = 40
        curve.evaluate()
        return curve
    return create_bezier


def bezier_cost(data,grad,create_curve=None,R=None,P1=None,P3=None,collision_vessels=None):
    curve = create_curve(data)
    pts   = np.array(curve.evalpts)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    #num = int(spline_length // R)
    #curve.sample_size = num
    #curve.evaluate()
    sample_pts = np.array(curve.evalpts)
    sample_pts_for_vec = np.vstack((P1,sample_pts,P3))
    vec = np.diff(sample_pts_for_vec,axis=0)
    collisions = get_collisions(collision_vessels,R,sample_pts)
    #vec = vec/np.linalg.norm(vec,axis=0)
    #angles = get_vecs(vec)
    curve_rads = np.array(get_radius(curve))
    curve_rads = 1/(curve_rads[curve_rads<R]+np.finfo(float).eps)
    a_sum = np.sum(curve_rads)
    if np.isclose(a_sum,0):
        vec = vec/np.linalg.norm(vec,axis=0)
        angles = 1/get_vecs(vec)
        if np.any(angles < 120):
            a_sum = np.sum(1/angles)
    return collisions+a_sum*spline_length



def test_b(P1,P2,P3,P4,R,collision_vessels):
    create_curve = connect_bezier(P1,P2,P3,P4,R,collision_vessels)
    cost = lambda d,g: bezier_cost(d,g,create_curve=create_curve,R=R,P1=P1,P3=P3,collision_vessels=collision_vessels)
    vis_comp = VisMPL.VisCurve3D()
    success = False
    x0 = np.zeros(11)
    max_time = 25
    count = 1
    L = 4*R
    level_attempts = 4
    attempt = 0
    best = np.inf
    while not success:
        print('Linking Optimization Path Search {} Level {}'.format(count,attempt))
        lb = np.ones((count+2)*3+2)*(-L) + x0
        lb[0] = 0
        lb[1] = 0
        ub = np.ones((count+2)*3+2)*(L) + x0
        o = nlopt.opt(2,(count+2)*3+2)
        o.set_min_objective(cost)
        o.set_lower_bounds(lb)
        o.set_upper_bounds(ub)
        o.set_maxtime(max_time)
        xopt = o.optimize(x0)
        curve = create_curve(xopt)
        curve.vis = vis_comp
        if o.last_optimum_value() < 100 and attempt == level_attempts:
            L += L*0.25
            success = True
            print(o.last_optimum_value())
        elif attempt < level_attempts:
            attempt += 1
            max_time += 5
            if o.last_optimum_value() < best:
                x0 = xopt
                best = o.last_optimum_value()
                curve.render()
            if np.isclose(o.last_optimum_value(),0):
                success = True
                print('Constraints Satisfied: Exiting...')
            continue
        else:
            print("Above link threshold {}".format(o.last_optimum_value()))
            L += L*0.25
            count += 1
            attempt = 0
            max_time = 25
            x0 = np.zeros((count+2)*3+2)
    curve = create_curve(xopt)
    return curve


def get_connect_points(P1,P2,P3,P4,R,collision_vessels):
    curve = test_b(P1,P2,P3,P4,R,collision_vessels)
    pts   = np.array(curve.evalpts)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    num = int(spline_length // R)+2
    curve.sample_size = num
    curve.evaluate()
    opt_pts = np.array(curve.evalpts)
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    ax.plot(opt_pts[:,0],opt_pts[:,1],opt_pts[:,2],label='final')
    plt.show()
    return opt_pts

def connect_linear_search(P1,P2,P3,P4,R,collsiion_vessels):
    pass

def opt(P1,P2,P3,P4,R,collision_vessels): #P1,P2,P3,P4,R):
    #P1 = np.random.random(3)
    #P2 = np.random.random(3)

    collision_vessels = np.array([[0,0.5,0.25,1,0.5,0.25,0.01],
                                  [0,0.7,0.25,1,0.7,0.25,0.01],
                                  [0,0.3,0.25,1,0.3,0.25,0.01],
                                  [0,0.9,0.25,1,0.9,0.25,0.01],
                                  [0,0.1,0.25,1,0.1,0.25,0.01],
                                  [0.25,0,0.25,0.25,1,0.25,0.01],
                                  [0.75,0,0.25,0.75,1,0.25,0.01],
                                  [0,0.5,0.75,1,0.5,0.75,0.01],
                                  [0,0.7,0.75,1,0.7,0.75,0.01],
                                  [0,0.3,0.75,1,0.3,0.75,0.01],
                                  [0,0.9,0.75,1,0.9,0.75,0.01],
                                  [0,0.1,0.75,1,0.1,0.75,0.01],
                                  [0.25,0,0.75,0.25,1,0.75,0.01],
                                  [0.75,0,0.75,0.75,1,0.75,0.01],
                                  [1,0,0,1,0,1,0.1]])

    #print(collision_vessels.shape)
    #P1 = np.array([0.5,0,0.5])
    #print(close_exact(collision_vessels,P1))
    #P2 = np.array([0.5,0.5,0.5])
    V1 = P2-P1
    # Tangent Constraint 1 for First Spline
    V1 = V1/np.linalg.norm(V1)
    #P3 = np.random.random(3)
    #P4 = np.random.random(3)
    #P3 = np.array([0.5,1,0])
    #P4 = np.array([0.5,0.5,0])
    V2 = P4-P3
    # Tangent Constraint 1 for Second Spline
    V2 = V2/np.linalg.norm(V2)
    MID = (P2 + P4)/2
    V3  = MID - P2
    # Tangent Constraint 2 for First Spline
    V3  = V3/np.linalg.norm(V3)
    V4  = MID - P4
    # Tangent Constraint 2 for Second Spline
    V4  = V4/np.linalg.norm(V4)
    LENGTH = np.linalg.norm(P2-MID)
    CTR0 = P2 + V1*0.01*LENGTH
    CTR1 = P2 + V1*0.1*LENGTH
    CTR2 = P4 + V2*0.1*LENGTH
    CTR3 = (P2+P4)/2 + (V1+V2)/2*0.25*LENGTH
    CTR4 = P4 + V2*0.01*LENGTH
    #CTR3 = P4 + V2*0.1
    # Create Cubic Spline 1
    def get_spline(data,P2=P2,P4=P4,V1=V1,V2=V2,CTR0=CTR0,CTR1=CTR1,CTR2=CTR2,CTR3=CTR3,MID=MID,LENGTH=LENGTH):
        CTR = np.zeros(len(data)).reshape(-1,3)
        seg = np.linspace(0.1,0.9)
        for i in range(CTR.shape[0]):
            CTR[i,:] = P2*(1-seg[i]) + V1*(1-seg[i])*LENGTH + P4*(seg[i]) + V2*(seg[i])*LENGTH
        CTR = CTR + data.reshape(-1,3)
        SPT1 = np.vstack((P2,CTR0,CTR,CTR4,P4))
        try:
            tck1, u = interpolate.splprep([SPT1[:,0],SPT1[:,1],SPT1[:,2]],s=0,k=3)
        except:
            CTR1 = CTR1 - data[0:3]
            CTR2 = CTR2 - data[3:]
            SPT1 = np.vstack((P2,CTR0,CTR1,CTR2,CTR4,P4))
            tck1, u = interpolate.splprep([SPT1[:,0],SPT1[:,1],SPT1[:,2]],s=0,k=3)
        spline_f1 = lambda t: np.array(interpolate.splev(t,tck=tck1,der=0))
        spline_df1 = lambda t: np.array(interpolate.splev(t,tck=tck1,der=1))
        spline_ddf1 = lambda t: np.array(interpolate.splev(t,tck=tck1,der=2))
        return spline_f1,spline_df1,spline_ddf1
    def show(ax,spl,color,label,R,P1=P1,P2=P2,P4=P4,CTR4=CTR4,MID=MID):
        #fig = plt.figure()
        #ax = fig.add_subplot(111,projection='3d')
        tt = np.linspace(0,1,num=400)
        spline1 = spl(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(spline1,axis=0),axis=1))
        num = int(spline_length // R) #(R*2*np.pi))
        t = np.linspace(0,1,num=num+2)
        spline1 = spl(t).T
        ax.plot(spline1[:,0],spline1[:,1],spline1[:,2],color=color,label=label)
        ax.plot([P1[0],P2[0]],[P1[1],P2[1]],[P1[2],P2[2]],color='green',label='inlet')
        ax.plot([P3[0],P4[0]],[P3[1],P4[1]],[P3[2],P4[2]],color='green',label='outlet')
        #ax.scatter(SPT1[:,0],SPT1[:,1],SPT1[:,2],color='black',label='data')
        #ax.legend()
        #plt.show()
        return ax
    angle_threshold = 120
    init = np.zeros(9)
    #init[0:3] = CTR1
    #init[3:]  = CTR2
    def cost(data,grad,spline_gen=get_spline,angle_threshold=angle_threshold,R=R,P1=P1,P2=P2,P4=P4,CTR4=CTR4,collision_vessels=collision_vessels):
        spf,spdf,spddf = get_spline(data)
        tt = np.linspace(0,1,num=500)
        pts = spf(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
        num = int(spline_length // R) #(R*2*np.pi))
        sample_tt = np.linspace(0,1,num=4*num+2)
        sample_pts = spf(sample_tt).T
        sample_pts_for_vec = np.vstack((P1,P2,sample_pts,P4,P3))
        vec = np.diff(sample_pts_for_vec,axis=0)
        collisions = get_collisions(collision_vessels,R,sample_pts)
        vec = vec/np.linalg.norm(vec,axis=0)
        angles = []
        div = 0
        angles = get_vecs(vec)
        a_sum = np.exp(np.sum(angles))
        a_try = np.sum(angles)
        return collisions+a_sum+spline_length
    def cost_polish(data,grad,spline_gen=get_spline,angle_threshold=angle_threshold,R=R,P1=P1,CTR4=CTR4,collision_vessels=collision_vessels):
        spf,spdf,spddf = get_spline(data)
        tt = np.linspace(0,1,num=500)
        pts = spf(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
        num = int(spline_length // R) #(R*2*np.pi))
        sample_tt = np.linspace(0,1,num=4*num+2)
        sample_pts = spf(sample_tt).T
        sample_pts = np.vstack((P1,sample_pts,CTR4))
        vec = np.diff(sample_pts,axis=0)
        collisions = get_collisions(collision_vessels,R,sample_pts)
        vec = vec/np.linalg.norm(vec,axis=0)
        angles = []
        div = 0
        angles = get_vecs(vec)
        a_sum = np.exp(np.sum(angles))
        return collisions+a_sum
    def check(data,spline_gen=get_spline,R=R,P1=P1,CTR4=CTR4):
        spf,spdf,spddf = get_spline(data)
        tt = np.linspace(0,1,num=500)
        pts = spf(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
        num = int(spline_length // R) #(R*2*np.pi))
        #print(num)
        sample_tt = np.linspace(0,1,num=num+2)
        sample_pts = spf(sample_tt).T
        sample_pts = np.vstack((P1,sample_pts,CTR4))
        vec = np.diff(sample_pts,axis=0)
        print(np.linalg.norm(vec,axis=0))
        vec = vec/np.linalg.norm(vec,axis=0)
        #print(vec)
        angles = []
        div = 0
        for i in range(vec.shape[0]-1):
            value = get_angle(vec[i,:],vec[i+1,:])
            angles.append(value)
        return max(angles),min(angles)
    def get_points(data,spline_gen=get_spline):
        spf,spdf,spddf = get_spline(data)
        tt = np.linspace(0,1,num=500)
        pts = spf(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
        num = int(spline_length // R) #(R*2*np.pi))
        #print(num)
        sample_tt = np.linspace(0,1,num=num+2)
        sample_pts = spf(sample_tt).T
        return sample_pts
    def get_curvature(data):
        spf,spdf,spddf = get_spline(data)
        curve = lambda t: curvature(t,df=spdf,ddf=spddf)
        return curve
    def cost2(data,grad,R=R,get_spline=get_spline,collision_vessels=collision_vessels):
        spf,spdf,spddf = get_spline(data)
        tt = np.linspace(0,1,num=500)
        pts = spf(tt).T
        spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
        kappa = get_curvature(data)
        collisions = get_collisions(collision_vessels,R,pts)
        tt = np.linspace(0,1,num=100)
        k = []
        for t in tt:
            value = 1/kappa(t)
            if value > R:
                k.append(1)
            else:
                k.append(0)
        return collisions + np.sum(k)*spline_length
    def cost3(t_vector,grad,spline=None,angle_threshold=angle_threshold,P1=P1,P2=P2,P3=P3,MID=MID,CTR4=CTR4):
        tt = np.linspace(0,1,num=len(t_vector)+2)
        tt[1:-1] = t_vector
        sample_pts = spline(t_vector).T
        sample_pts = np.vstack((P1,P2,sample_pts,P4,P3))
        vec = np.diff(sample_pts,axis=0)
        vec = vec/np.linalg.norm(vec,axis=0)
        angles = []
        for i in range(vec.shape[0]-1):
            value = get_angle(vec[i,:],vec[i+1,:])
            if value < angle_threshold:
                angles.append(1/value)
            else:
                angles.append(0)
        a_sum = np.exp(100*np.sum(angles))
        return a_sum
    return get_spline,show,cost2,init,LENGTH,check,get_points,get_curvature,cost3,collision_vessels,cost_polish


def form_connection(R):
    f,s,c,i,L,ch,gp,cv = opt(P1,P2,P3,P4,R)
    spf,spdf = f(i)
    res1 = optimize.shgo(c,iters=3,bounds=[(-L,L),(-L,L),(-L,L),(-L,L),(-L,L),(-L,L)],minimizer_kwargs={"method":"Nelder-Mead","N":12000})
    spf,spdf = f(res1.x)
    pts = gp(res1.x)
    return pts

def test_nlopt(P1,P2,P3,P4,R,collision_vessels):
    f,s,c,i,L,ch,gp,gc,c3,cv,cp = opt(P1,P2,P3,P4,R,collision_vessels)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    #ax2 = fig.add_subplot(122,projection='3d')
    spf,spdf,spddf = f(i)
    ax1 = s(ax1,spf,'red','initial',R)
    for i in range(cv.shape[0]):
        ax1.plot([cv[i,0],cv[i,3]],[cv[i,1],cv[i,4]],[cv[i,2],cv[i,5]],color='black',label='other vessel {}'.format(i))
    success = False
    count = 0
    max_time = 25
    x0 = np.zeros((count+2)*3)
    while not success:
        print('Linking Optimization Path Search {}'.format(count))
        bounds = [(-L,L)]*((count+2)*3)
        lb = np.ones((count+2)*3)*(-L) + x0
        ub = np.ones((count+2)*3)*(L) + x0
        o = nlopt.opt(2,(count+2)*3)
        o.set_min_objective(c)
        o.set_lower_bounds(lb)
        o.set_upper_bounds(ub)
        o.set_maxtime(max_time)
        xopt = o.optimize(x0)
        if o.last_optimum_value() < 10:
            L += L*0.25
            success = True
        else:
            print("Above link threshold {}".format(o.last_optimum_value()))
            L += L*0.25
            count += 1
            max_time += 0
            x0 = np.zeros((count+2)*3)
            spfc,spdfc,spddfc = f(xopt)
            tj = np.linspace(0.1,0.9,num=count+2)
            for j in range(len(tj)):
                x0[j*3:j*3+3] = spfc(tj[j])
    spfc,spdfc,spddfc = f(xopt)
    print("Collision Optimum: {}".format(o.last_optimum_value()))
    old_best = cp(xopt,np.array([]))
    tt = np.linspace(0,1,num=500)
    pts = spfc(tt).T
    ax1.plot(pts[:,0],pts[:,1],pts[:,2],color='orange',label='no collision spline')
    for i in range(int(xopt.size//3)):
        def local_cost(point,grad,idx=i,data=xopt,lc=cp):
            data[i*3:i*3+3] = point
            return c(data,grad)
        o = nlopt.opt(2,3)
        o.set_min_objective(local_cost)
        o.set_lower_bounds(lb[i*3:i*3+3])
        o.set_upper_bounds(ub[i*3:i*3+3])
        o.set_maxtime(15)
        local_xopt = o.optimize(xopt[i*3:i*3+3])
        if o.last_optimum_value() < old_best:
            xopt[i*3:i*3+3] = local_xopt
            print('New Optimum Found')
            old_best = o.last_optimum_value()
    print('Finished Local Adjustments')
    spf,spdf,spddf = f(xopt)
    print("Curvature Optimum: {}".format(o.last_optimum_value()))
    cost3 = lambda t,grad: c3(t,grad,spline=spf)
    tt = np.linspace(0,1,num=500)
    pts = spf(tt).T
    ax1.plot(pts[:,0],pts[:,1],pts[:,2],color='yellow',label='curvature optimized spline')
    pts_xopt = xopt.reshape(-1,3)
    ax1.scatter(pts_xopt[:,0],pts_xopt[:,1],pts_xopt[:,2],color='red')
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    num = int(spline_length // R)
    t_vector = np.linspace(0,1,num=num+2)
    bounds = []
    lb = []
    ub = []
    for i in range(len(t_vector)-1):
        lb.append(t_vector[i])
        ub.append(t_vector[i+1])
        bounds.append(tuple([t_vector[i],t_vector[i+1]]))
    lb = np.array(lb)
    ub = np.array(ub)
    t0 = (lb + ub)/2
    #print("Performing Vessel Annealing Optimization")
    seg_opt = nlopt.opt(2,len(t0))
    seg_opt.set_min_objective(cost3)
    seg_opt.set_lower_bounds(lb)
    seg_opt.set_upper_bounds(ub)
    seg_opt.set_stopval(100)
    seg_opt.set_maxtime(100)
    #t_vector = seg_opt.optimize(t0)
    #tt = np.linspace(0,1,num=len(t_vector)+2)
    #tt[1:-1] = t_vector
    #spline1 = spf(tt).T
    #ax1.plot(spline1[:,0],spline1[:,1],spline1[:,2],color='blue',label='final')
    plt.show()
    return xopt,spf


def test(R):
    f,s,c,i,L,ch,gp,gc,c3,cv,cp = opt(R)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    #ax2 = fig.add_subplot(122,projection='3d')
    spf,spdf,spddf = f(i)
    ax1 = s(ax1,spf,'red','initial',R)
    for i in range(cv.shape[0]):
        ax1.plot([cv[i,0],cv[i,3]],[cv[i,1],cv[i,4]],[cv[i,2],cv[i,5]],color='black',label='other vessel {}'.format(i))
    #ax2 = s(ax2,spf,'red','initial',R)
    #res = optimize.minimize(c,i,method='Nelder-Mead',options={'disp':True})
    #res = optimize.differential_evolution(c,maxiter=8000,bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)])
    #print("Performing Brute Seed Selection")
    #x0 = optimize.brute(c,Ns=5,ranges=[(-L,L),(-L,L),(-L,L),(-L,L),(-L,L),(-L,L)],disp=True)
    #print("Basinhopping Optimization")
    #res1 = optimize.basinhopping(c,x0)
    #print("Spline Optimization Complete")
    success = False
    count = 0
    while not success:
        print('Linking Optimization Path Search {}'.format(count))
        bounds = [(-L,L)]*((count+2)*3)
        res1 = optimize.shgo(c,iters=1,bounds=bounds,options={'disp':True})#,minimizer_kwargs={"method":"Nelder-Mead","N":12000})
        #res1 = optimize.dual_annealing(c,bounds=bounds)
        #res1 = optimize.basinhopping(c,np.zeros((count+2)*3))
        #res1 = optimize.differential_evolution(c,bounds=bounds)
        #res1 = optimize.minimize(c,np.zeros((count+2)*3),method="L-BFGS-B",bounds=bounds)
        if res1.fun < 10:
            L += L*0.25
            bounds = [(-L,L)]*((count+2)*3)
            success = True
        else:
            print("Above link threshold {}".format(res1.fun))
            L += L*0.25
            count += 1
    res1 = optimize.minimize(cp,res1.x,method='L-BFGS-B',bounds=bounds,options={'disp':1})
    #res2 = optimize.basinhopping(c,np.zeros(6),niter=200,minimizer_kwargs={"method":"Nelder-Mead"})#,bounds=[(-L,L),(-L,L),(-L,L),(-L,L),(-L,L),(-L,L)])#,minimizer_kwargs={"method":"BFGS"})
    #res = optimize.basinhopping(c,i,minimizer_kwargs={"method": "L-BFGS-B","bounds":[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)]},T=0.25,niter=200,disp=True) #,bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)])#,i,minimizer_kwargs={"method": "Nelder-Mead"},disp=True)#bounds=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])
    spf,spdf,spddf = f(res1.x)
    print("Curvature Optimum: {}".format(res1.fun))
    cost3 = lambda t: c3(t,spline=spf)
    tt = np.linspace(0,1,num=500)
    pts = spf(tt).T
    ax1.plot(pts[:,0],pts[:,1],pts[:,2],color='yellow',label='intermediate')
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    num = int(spline_length // R)
    print(num)
    t_vector = np.linspace(0,1,num=num+2)
    bounds = []
    for i in range(len(t_vector)-1):
        bounds.append(tuple([t_vector[i],t_vector[i+1]]))
    t0 = t_vector[1:-1]
    print("Performing Vessel Annealing Optimization")
    res3 = optimize.dual_annealing(cost3,bounds=bounds)#,minimizer_kwargs={"method":"L-BFGS-B"})
    t_vector = res3.x
    tt = np.linspace(0,1,num=len(t_vector)+2)
    tt[1:-1] = t_vector
    spline1 = spf(tt).T
    ax1.plot(spline1[:,0],spline1[:,1],spline1[:,2],color='blue',label='final')
    #ax1 = s(ax1,spf,'blue','final',R)
    #ax1.set_title('results 1')
    #spf,spdf = f(res2.x)
    #ax2 = s(ax2,spf,'blue','final',R)
    #ax2.set_title('results 2')
    #plt.legend()
    plt.show()
    #print(res.fun)
    #ANGLES = ch(res1.x)
    #print('Min angle:\n{}'.format(ANGLES[1]))
    #print('Max angle:\n{}'.format(ANGLES[0]))
    #ANGLES = ch(res2.x)
    #print('Min angle:\n{}'.format(ANGLES[1]))
    #print('Max angle:\n{}'.format(ANGLES[0]))
    pts = gp(res1.x)
    return pts


def retest(data,f,s,c,i,L,ch):
    fig = plt.figure()
    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122,projection='3d')
    spf,spdf = f(i)
    ax1 = s(ax1,spf,'red','initial',R)
    ax2 = s(ax2,spf,'red','initial',R)
    #res = optimize.minimize(c,i,method='Nelder-Mead',options={'disp':True})
    #res = optimize.differential_evolution(c,maxiter=8000,bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)])
    res1 = optimize.shgo(c,iters=3,bounds=[(-L,L),(-L,L),(-L,L),(-L,L),(-L,L),(-L,L)],minimizer_kwargs={"method":"Nelder-Mead","N":12000})
    res2 = optimize.shgo(c,iters=3,bounds=[(-L,L),(-L,L),(-L,L),(-L,L),(-L,L),(-L,L)],minimizer_kwargs={"method":"BFGS"})
    #res = optimize.basinhopping(c,i,minimizer_kwargs={"method": "L-BFGS-B","bounds":[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)]},T=0.25,niter=200,disp=True) #,bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)])#,i,minimizer_kwargs={"method": "Nelder-Mead"},disp=True)#bounds=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])
    spf,spdf = f(res1.x)
    ax1 = s(ax1,spf,'blue','final',R)
    ax1.set_title('results 1')
    spf,spdf = f(res2.x)
    ax2 = s(ax2,spf,'blue','final',R)
    ax2.set_title('results 2')
    plt.show()
    #print(res.fun)
    ANGLES = ch(res1.x)
    print('Min angle:\n{}'.format(ANGLES[1]))
    print('Max angle:\n{}'.format(ANGLES[0]))
    ANGLES = ch(res2.x)
    print('Min angle:\n{}'.format(ANGLES[1]))
    print('Max angle:\n{}'.format(ANGLES[0]))



def show(spl):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    tt = np.linspace(0,1,num=400)
    spline1 = spl(tt).T
    ax.plot(spline1[:,0],spline1[:,1],spline1[:,2],color='blue',label='unclamped')
    #ax.plot([P1[0],P2[0]],[P1[1],P2[1]],[P1[2],P2[2]],color='green',label='inlet')
    #ax.plot([MID[0],CTR4[0]],[MID[1],CTR4[1]],[MID[2],CTR4[2]],color='green',label='outlet')
    #ax.scatter(SPT1[:,0],SPT1[:,1],SPT1[:,2],color='black',label='data')
    #ax.legend()
    plt.show()


##
# Test 1
##

P1 = np.array([0.5,0,0.5])
P2 = np.array([0.5,0.5,0.5])
P3 = np.array([0.5,1,0])
P4 = np.array([0.5,0.5,0])
R  = 0.1
collision_vessels = np.array([[1,0,0,1,0,1,0.1]])



P1 = np.array([0.5,0,0.5])
P2 = np.array([0.5,0.5,0.5])
P3 = np.array([0.5,0,0])
P4 = np.array([0.5,0.5,0])
R  = 0.1
collision_vessels = np.array([[1,0,0,1,0,1,0.1]])


P1 = np.array([0.5,0,0.5])
P2 = np.array([0,0.5,0.3])
P3 = np.array([0.5,1,0])
P4 = np.array([0.5,0.5,0])
R  = 0.1
collision_vessels = np.array([[1,0,0,1,0,1,0.1]])


P1 = np.array([0.5,0,0.5])
P2 = np.array([0,0.5,0.3])
P3 = np.array([1,0,0])
P4 = np.array([1,0.5,0])
R  = 0.1
collision_vessels = np.array([[1,0,0,1,0,1,0.1]])


P1 = np.array([0.5,0,0.5])
P2 = np.array([0.5,0.5,0.5])
P3 = np.array([0.5,1,0])
P4 = np.array([0.5,0.5,0])
R  = 0.1
collision_vessels = np.array([[0,0.5,0.25,1,0.5,0.25,0.01],
                              [0,0.7,0.25,1,0.7,0.25,0.01],
                              [0,0.3,0.25,1,0.3,0.25,0.01],
                              [0,0.9,0.25,1,0.9,0.25,0.01],
                              [0,0.1,0.25,1,0.1,0.25,0.01],
                              [0.25,0,0.25,0.25,1,0.25,0.01],
                              [0.75,0,0.25,0.75,1,0.25,0.01],
                              [0,0.5,0.75,1,0.5,0.75,0.01],
                              [0,0.7,0.75,1,0.7,0.75,0.01],
                              [0,0.3,0.75,1,0.3,0.75,0.01],
                              [0,0.9,0.75,1,0.9,0.75,0.01],
                              [0,0.1,0.75,1,0.1,0.75,0.01],
                              [0.25,0,0.75,0.25,1,0.75,0.01],
                              [0.75,0,0.75,0.75,1,0.75,0.01],
                              [1,0,0,1,0,1,0.1]])

"""
