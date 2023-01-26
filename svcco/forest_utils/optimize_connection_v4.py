# Optimization for Vessel Connections

import numpy as np
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
import numba as nb
from scipy import interpolate, optimize
import pyvista as pv

@nb.jit(nopython=True)
def get_angle(V1,V2):
    return np.arccos(np.dot(-V1,V2))*(180/np.pi)

@nb.jit(nopython=True)
def get_all_angles(vectors):
    angles = np.zeros(vectors.shape[0]-1)
    for i in range(vectors.shape[0]-1):
        angles[i] = get_angle(vectors[i,:],vectors[i+1,:])
    return angles

@nb.jit(nopython=True)
def close_exact(data,point,radius_buffer):
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
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2) - data[i,6] - radius_buffer
    return line_distances

def cost_angles(angles):
    #func = lambda x: np.tanh(-(x+np.arctanh(np.log(2))))+np.log(1+np.exp(-x))
    func = lambda x: np.log(1+np.exp(-x))
    return np.sum(func(angles))

def get_all_vectors(pts):
    vectors = pts[1:] - pts[:-1]
    vectors = vectors/np.linalg.norm(vectors,axis=1).reshape(-1,1)
    return vectors

def check_bounds(bounds,sample_pts):
    bounds_score = 0
    #func = lambda x: np.tanh((x-np.arctanh(np.log(2))))+np.log(1+np.exp(x))
    #func = lambda x: np.exp(x)
    func = lambda x: np.log(1+np.exp(x*100))
    for i in range(sample_pts.shape[0]):
        bounds_score += func(bounds[0,0] - sample_pts[i,0])
        bounds_score += func(sample_pts[i,0] - bounds[0,1])
        bounds_score += func(bounds[1,0] - sample_pts[i,1])
        bounds_score += func(sample_pts[i,1] - bounds[1,1])
        bounds_score += func(bounds[2,0] - sample_pts[i,2])
        bounds_score += func(sample_pts[i,2] - bounds[2,1])
    return bounds_score

def get_collisions(collision_vessels,R,sample_pts,radius_buffer):
    collisions = 0
    #func = lambda x: np.tanh((x-np.arctanh(np.log(2))))+np.log(1+np.exp(x))
    #func = lambda x: np.log(1+np.exp(x))
    func = lambda x: 0.5*np.tanh(x*100)+0.5
    for i in range(sample_pts.shape[0]):
        dist = close_exact(collision_vessels,sample_pts[i,:],radius_buffer)
        collisions += np.sum(func(R + radius_buffer - dist))
    return collisions

def connect_bezier(P1,P2,P3,P4,clamp_first=True,clamp_second=True,number_vessels=20):
    if clamp_first and clamp_second:
        V1   = (P2 - P1)
        L1   = np.linalg.norm(V1)
        V1   = V1/np.linalg.norm(V1)
        V2   = (P4 - P3)
        L2   = np.linalg.norm(V2)
        V2   = V2/np.linalg.norm(V2)
        def create_bezier(data,V1=V1,V2=V2,P2=P2,P4=P4,L1=L1,L2=L2):
            CTR0 = P2 + (L1/2+data[2])*V1
            CTR1 = P4 + (L2/2+data[3])*V2
            CTR  = np.zeros(data.shape[0]-4+12)
            fill_CTR = np.zeros(data.shape[0]-4).reshape(-1,3)
            seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
            for i in range(fill_CTR.shape[0]):
                fill_CTR[i,:] = CTR0*(1-seg[i]) + CTR1*(seg[i])
            fill_CTR = fill_CTR.flatten()
            CTR[0:3]     = P1 + (L1/4+data[0])*V1
            CTR[3:6]     = CTR0
            CTR[-6:-3]   = CTR1
            CTR[-3:]     = P3 + (L2/4+data[1])*V2
            CTR[6:-6]    += fill_CTR + data[4:]
            CTR  = CTR.reshape(-1,3).tolist()
            curve = BSpline.Curve()
            curve.degree = len(CTR) - 1
            curve.ctrlpts = CTR
            curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
            curve.sample_size = number_vessels
            curve.evaluate()
            return curve
    elif not clamp_first and clamp_second:
        V2   = (P4 - P3)
        L2   = np.linalg.norm(V2)
        V2   = V2/np.linalg.norm(V2)
        def create_bezier(data,V2=V2,P2=P2,P4=P4,L2=L2):
            CTR1 = P4 + (L2/2+data[1])*V2
            CTR  = np.zeros(data.shape[0]-2+9)
            fill_CTR = np.zeros(data.shape[0]-2).reshape(-1,3)
            seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
            for i in range(fill_CTR.shape[0]):
                fill_CTR[i,:] = P2*(1-seg[i]) + CTR1*(seg[i])
            fill_CTR = fill_CTR.flatten()
            CTR[0:3]     = P2
            CTR[-6:-3]   = CTR1
            CTR[-3:]     = P3 + (L2/4+data[0])*V2
            CTR[3:-6]    += fill_CTR + data[2:]
            CTR  = CTR.reshape(-1,3).tolist()
            curve = BSpline.Curve()
            curve.degree = len(CTR) - 1
            curve.ctrlpts = CTR
            curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
            curve.sample_size = 20
            curve.evaluate()
            return curve
    elif clamp_first and not clamp_second:
        V1   = (P2 - P1)
        L1   = np.linalg.norm(V1)
        V1   = V1/np.linalg.norm(V1)
        def create_bezier(data,V1=V1,P2=P2,P4=P4,L1=L1):
            CTR0 = P2 + (L1/2+data[1])*V1
            CTR  = np.zeros(data.shape[0]-2+9)
            fill_CTR = np.zeros(data.shape[0]-2).reshape(-1,3)
            seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
            for i in range(fill_CTR.shape[0]):
                fill_CTR[i,:] = CTR0*(1-seg[i]) + P4*(seg[i])
            fill_CTR = fill_CTR.flatten()
            CTR[0:3]     = P1 + (L1/4+data[0])*V1
            CTR[3:6]     = CTR0
            CTR[-3:]     = P4
            CTR[6:-3]    += fill_CTR + data[2:]
            CTR  = CTR.reshape(-1,3).tolist()
            curve = BSpline.Curve()
            curve.degree = len(CTR) - 1
            curve.ctrlpts = CTR
            curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
            curve.sample_size = 20
            curve.evaluate()
            return curve
    else:
        def create_bezier(data,P2=P2,P4=P4):
            CTR  = np.zeros(data.shape[0]+6)
            fill_CTR = np.zeros(data.shape[0]).reshape(-1,3)
            seg = np.linspace(0.1,0.9,num=fill_CTR.shape[0])
            for i in range(fill_CTR.shape[0]):
                fill_CTR[i,:] = P2*(1-seg[i]) + P4*(seg[i])
            fill_CTR = fill_CTR.flatten()
            CTR[0:3]     = P2
            CTR[-3:]     = P4
            CTR[3:-3]    += fill_CTR + data
            CTR  = CTR.reshape(-1,3).tolist()
            curve = BSpline.Curve()
            curve.degree = len(CTR) - 1
            curve.ctrlpts = CTR
            curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
            curve.sample_size = 20
            curve.evaluate()
            return curve
    return create_bezier

def bezier_cost(data,create_curve=None,R=None,
                P1=None,P3=None,collision_vessels=None,
                radius_buffer=0,bounds=None,sample_size=20,
                clamp_first=True,clamp_second=True):
    curve = create_curve(data)
    curve.sample_size = sample_size
    curve.evaluate()
    pts   = np.array(curve.evalpts)
    sample_pts = np.array(curve.evalpts)
    if collision_vessels is not None:
        collisions = get_collisions(collision_vessels,R,sample_pts,radius_buffer)
    else:
        collisions = 0
    bound_score = check_bounds(bounds,sample_pts)
    if clamp_first and clamp_second:
        pts = np.vstack((P1,pts,P3))
    elif clamp_first and not clamp_second:
        pts = np.vstack((P1,pts))
    elif not clamp_first and clamp_second:
        pts = np.vstack((pts,P3))
    else:
        pass
    vectors = get_all_vectors(pts)
    angles  = get_all_angles(vectors)
    angles  = cost_angles(angles-110)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    connection_length_min = np.linalg.norm(P3-P1)
    spline_length_norm = spline_length/connection_length_min
    return (bound_score+angles+collisions)*spline_length_norm

class connection:
    def __init__(self,forest,network,tree_idx,tree_jdx,terminal_index,radius_buffer,
                 clamp_first=True,clamp_second=True,seperate_connection=True,
                 number_vessels=20):
        self.network  = network
        self.tree_idx = tree_idx
        self.tree_jdx = tree_jdx
        self.edge_idx = forest.assignments[self.network][self.tree_idx][terminal_index]
        self.edge_jdx = forest.assignments[self.network][self.tree_jdx][terminal_index]
        self.edge_1   = forest.networks[self.network][self.tree_idx].data[self.edge_idx,:]
        self.edge_2   = forest.networks[self.network][self.tree_jdx].data[self.edge_jdx,:]
        self.tree_idx_last_index = int(np.max(forest.networks[self.network][self.tree_idx].data[:,-1]))+1
        self.tree_idx_last_index_conn = int(np.max(forest.connection))
        self.tree_jdx_last_index = int(np.max(forest.networks[self.network][self.tree_jdx].data[:,-1]))+1
        self.P1       = self.edge_1[0:3]
        self.P2       = self.edge_1[3:6]
        self.P3       = self.edge_2[0:3]
        self.P4       = self.edge_2[3:6]
        self.R1       = self.edge_1[21]
        self.R2       = self.edge_2[21]
        self.bounds   = np.array([forest.boundary.x_range,forest.boundary.y_range,forest.boundary.z_range])
        self.best     = np.inf
        self.xopt     = None
        self.number_vessels = number_vessels
        self.generator = connect_bezier(self.P1,self.P2,self.P3,self.P4,clamp_first=clamp_first,clamp_second=clamp_second,number_vessels=number_vessels)
        self.boundary = forest.boundary.pv_polydata
        tree_idx_data = forest.networks[self.network][self.tree_idx].data
        tree_jdx_data = forest.networks[self.network][self.tree_jdx].data
        self.upstream_tree_idx = [self.edge_1]
        self.upstream_tree_jdx = [self.edge_2]
        parent_tree_idx = int(self.edge_1[17])
        parent_tree_jdx = int(self.edge_2[17])
        while parent_tree_idx > -1:
            parent = tree_idx_data[parent_tree_idx,:]
            self.upstream_tree_idx.append(parent)
            parent_tree_idx = int(parent[17])
        while parent_tree_jdx > -1:
            parent = tree_jdx_data[parent_tree_jdx,:]
            self.upstream_tree_jdx.append(parent)
            parent_tree_jdx = int(parent[17])
        self.collision_vessels = np.zeros((tree_idx_data.shape[0]+tree_jdx_data.shape[0]-2,7))
        first_chunk  = tree_idx_data[:self.edge_idx,:]
        second_chunk = tree_idx_data[self.edge_idx+1:,:]
        third_chunk  = tree_jdx_data[:self.edge_jdx,:]
        fourth_chunk = tree_jdx_data[self.edge_jdx+1:,:]
        proximal_points = np.vstack((first_chunk[:,0:3],second_chunk[:,0:3],third_chunk[:,0:3],fourth_chunk[:,0:3]))
        distal_points   = np.vstack((first_chunk[:,3:6],second_chunk[:,3:6],third_chunk[:,3:6],fourth_chunk[:,3:6]))
        radii           = np.vstack((first_chunk[:,21].reshape(-1,1),second_chunk[:,21].reshape(-1,1),third_chunk[:,21].reshape(-1,1),fourth_chunk[:,21].reshape(-1,1)))
        self.collision_vessels[:,0:3] = proximal_points
        self.collision_vessels[:,3:6] = distal_points
        self.collision_vessels[:,6] = radii.T
        self.radius_buffer = radius_buffer
        self.clamp_first = clamp_first
        self.clamp_second = clamp_second
        self.seperate_connection = seperate_connection
    def solve(self,number_free_points=2,iters=1,maxtime=None,f_min=None):
        self.create_curve = connect_bezier(self.P1,self.P2,self.P3,self.P4,
                                           clamp_first=self.clamp_first,
                                           clamp_second=self.clamp_second,
                                           number_vessels=self.number_vessels)
        cost = lambda data: bezier_cost(data,create_curve=self.create_curve,
                                        R=self.R1,P1=self.P1,P3=self.P3,
                                        collision_vessels=self.collision_vessels,
                                        sample_size=self.number_vessels,
                                        radius_buffer=self.radius_buffer,bounds=self.bounds,
                                        clamp_first=self.clamp_first,clamp_second=self.clamp_second)
        if self.clamp_first and self.clamp_second:
            clamp_number = 4
        elif self.clamp_first or self.clamp_second:
            clamp_number = 2
        else:
            clamp_number = 0
        if self.xopt is None:
            self.xopt = np.zeros(clamp_number+3*number_free_points)
        if not self.xopt.shape[0] == clamp_number+3*number_free_points:
            self.xopt = np.zeros(clamp_number+3*number_free_points)
        L = np.linalg.norm(self.P2-self.P4)
        lb = np.ones(self.xopt.shape[0])*(-L)
        for i in range(clamp_number):
            lb[i] = 0
        ub = np.ones(self.xopt.shape[0])*L
        ctrl_bounds = []
        for b in range(len(lb)):
            ctrl_bounds.append([lb[b],ub[b]])
        if number_free_points == 1:
            if maxtime is None and f_min is None:
                res = optimize.shgo(cost,bounds=ctrl_bounds,iters=iters)
            elif not maxtime is None:
                res = optimize.shgo(cost,bounds=ctrl_bounds,iters=iters,options={'maxtime':maxtime})
            elif not f_min is None:
                res = optimize.shgo(cost,bounds=ctrl_bounds,iters=iters,options={'f_min':f_min})
        else:
            res = optimize.basinhopping(cost,self.xopt,niter=iters)
        self.xopt = res.x
        self.best = res.fun
        return res
    def check_angles(self):
        curve = self.create_curve(self.xopt)
        curve.sample_size = self.number_vessels
        curve.evaluate()
        pts   = np.array(curve.evalpts)
        pts = np.vstack((self.P1,pts,self.P3))
        vectors = get_all_vectors(pts)
        angles  = get_all_angles(vectors)
        return cost_angles(angles)
    def check_bounds(self):
        curve = self.create_curve(self.xopt)
        curve.sample_size = self.number_vessels
        curve.evaluate()
        pts   = np.array(curve.evalpts)
        return check_bounds(self.bounds,pts)
    def check_collisions(self):
        curve = self.create_curve(self.xopt)
        curve.sample_size = self.number_vessels
        curve.evaluate()
        pts   = np.array(curve.evalpts)
        return get_collisions(self.collision_vessels,max(self.R1,self.R2),pts,self.radius_buffer)
    def show(self):
        tree_1_vessels = []
        tree_2_vessels = []
        connection_vessels = []
        other_vessels = []
        for vessel in range(len(self.upstream_tree_idx)):
            center    = (self.upstream_tree_idx[vessel][0:3] + self.upstream_tree_idx[vessel][3:6])/2
            direction = self.upstream_tree_idx[vessel][3:6] - self.upstream_tree_idx[vessel][0:3]
            direction = direction/np.linalg.norm(direction)
            radius    = self.upstream_tree_idx[vessel][21]
            length    = self.upstream_tree_idx[vessel][20]
            cylinder  = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            tree_1_vessels.append(cylinder)
        for vessel in range(len(self.upstream_tree_jdx)):
            center    = (self.upstream_tree_jdx[vessel][0:3] + self.upstream_tree_jdx[vessel][3:6])/2
            direction = self.upstream_tree_jdx[vessel][3:6] - self.upstream_tree_jdx[vessel][0:3]
            direction = direction/np.linalg.norm(direction)
            radius    = self.upstream_tree_jdx[vessel][21]
            length    = self.upstream_tree_jdx[vessel][20]
            cylinder  = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            tree_2_vessels.append(cylinder)
        if not self.xopt is None:
            curve = self.generator(self.xopt)
            pts   = np.array(curve.evalpts)
            print(pts.shape)
            centers    = (pts[1:,:] + pts[:-1,:])/2
            directions = (pts[1:,:] - pts[:-1,:])
            print('directions done')
            lengths    = np.linalg.norm(directions,axis=1).reshape(-1,1)
            print(lengths.shape)
            print('lengths done')
            directions = directions/lengths
            print('division done')
            if self.seperate_connection:
                num = len(lengths)//2
            for i in range(len(lengths)):
                if self.seperate_connection:
                    if i < num:
                        cylinder = pv.Cylinder(center=centers[i,:],direction=directions[i,:],radius=self.R1,height=lengths[i])
                    else:
                        cylinder = pv.Cylinder(center=centers[i,:],direction=directions[i,:],radius=self.R2,height=lengths[i])
                connection_vessels.append(cylinder)
        for vessel in range(self.collision_vessels.shape[0]):
            center = (self.collision_vessels[vessel,0:3] + self.collision_vessels[vessel,3:6])/2
            direction = self.collision_vessels[vessel,3:6] - self.collision_vessels[vessel,0:3]
            length = np.linalg.norm(direction)
            direction = direction/length
            radius = self.collision_vessels[vessel,-1]
            cylinder  = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            other_vessels.append(cylinder)
        plotter = pv.Plotter()
        for vessel in tree_1_vessels:
            plotter.add_mesh(vessel,color='red')
        for vessel in tree_2_vessels:
            plotter.add_mesh(vessel,color='blue')
        for vessel in connection_vessels:
            plotter.add_mesh(vessel,color='green')
        for vessel in other_vessels:
            plotter.add_mesh(vessel,color='black')
        plotter.add_mesh(self.boundary,opacity=0.25)
        plotter.show()
    def vessels(self):
        curve = self.create_curve(self.xopt)
        curve.sample_size = self.number_vessels
        curve.evaluate()
        pts   = np.array(curve.evalpts)
        pts = np.vstack((self.P1,pts,self.P3))
        vessels = np.zeros((pts.shape[0]-1,7))
        vessels[:,0:3] = pts[:-1,:]
        vessels[:,3:6] = pts[1:,:]
        vessels[:,6]  = max(self.R1,self.R2)
        # Now construct the real vessels
        real_vessels = np.zeros((pts.shape[0]-1,self.edge_1.shape))
        real_vessels[:,0:3] = pts[:-1,:]
        real_vessels[:,3:6] = pts[1:,:]

        return vessels
    def add_collision_vessels(self,connection_object):
        other_connection_vessels = connection_object.vessels()
        self.collision_vessels = np.vstack((self.collision_vessels,other_connection_vessels))
    def replace_terminals(self,forest):
        pass
