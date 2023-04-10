# Optimization for Vessel Connections

import numpy as np
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
import numba as nb
from scipy import interpolate, optimize
import pyvista as pv
from tqdm import tqdm,trange
from time import perf_counter
from copy import deepcopy
# Function Factory for BSpline Generation
def bezier_factory(p1, p2, p3, p4, clamp_first=True, clamp_second=True):
    def create_bezier(data, p1_=p1, p2_=p2, p3_=p3, p4_=p4, c1=clamp_first, c2=clamp_second):
        v1 = p2_ - p1_
        v2 = p4_ - p3_
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        v1 = v1 / l1
        v2 = v2 / l2
        if c1 and c2:
            if data[0] < 0:
                data[0] = 0
            if data[1] < 0:
                data[1] = 0
            if data[0] > 1:
                data[0] = 1
            if data[1] > 1:
                data[1] = 1
            fill_number = data.shape[0] - 2
            ctr_number = data.shape[0] - 2 + 18
            data_offset = 2
            ctr0 = p1_
            ctr1 = p1_ + v1 * (l1 / 2)
            ctr2 = p1_ + (1 + 3 * data[0]) * v1 * (l1 / 4)
            ctr3 = p3_ + (1 + 3 * data[1]) * v2 * (l2 / 4)
            ctr4 = p3_ + v2 * (l2 / 2)
            ctr5 = p3_
            ctr = np.zeros(ctr_number)
            fill_ctr = np.zeros(fill_number).reshape(-1, 3)
            t = np.linspace(0.25, 0.75, fill_ctr.shape[0])
            for idx in range(fill_ctr.shape[0]):
                fill_ctr[idx, :] = ctr2 * (1 - t[idx]) + ctr3 * (t[idx])
            fill_ctr = fill_ctr.flatten()
            ctr[0:3] = ctr0
            ctr[3:6] = ctr1
            ctr[6:9] = ctr2
            ctr[-9:-6] = ctr3
            ctr[-6:-3] = ctr4
            ctr[-3:] = ctr5
            ctr[9:-9] += fill_ctr + data[data_offset:]
        elif c1 and not c2:
            if data[0] < 0:
                data[0] = 0
            if data[0] > 1:
                data[1] = 1
            fill_number = data.shape[0] - 1
            ctr_number = data.shape[0] - 1 + 9
            data_offset = 1
            ctr1 = p1_
            ctr2 = p1_ + (1 + 3 * data[0]) * v1 * (l1 / 4)
            ctr3 = p3_
            ctr = np.zeros(ctr_number)
            fill_ctr = np.zeros(fill_number).reshape(-1, 3)
            t = np.linspace(0.1, 0.9, fill_ctr.shape[0])
            for idx in range(fill_ctr.shape[0]):
                fill_ctr[idx, :] = ctr2 * (1 - t[idx]) + ctr3 * (t[idx])
            fill_ctr = fill_ctr.flatten()
            ctr[0:3] = ctr1
            ctr[3:6] = ctr2
            ctr[-3:] = ctr3
            ctr[6:-3] += fill_ctr + data[data_offset:]
        elif not c1 and c2:
            if data[0] < 0:
                data[0] = 0
            if data[0] > 1:
                data[1] = 1
            fill_number = data.shape[0] - 1
            ctr_number = data.shape[0] - 1 + 9
            data_offset = 1
            ctr1 = p1_
            ctr2 = p3_ + (1 + 3 * data[2]) * v2 * (l2 / 4)
            ctr3 = p3_
            ctr = np.zeros(ctr_number)
            fill_ctr = np.zeros(fill_number).reshape(-1, 3)
            t = np.linspace(0.1, 0.9, fill_ctr.shape[0])
            for idx in range(fill_ctr.shape[0]):
                fill_ctr[idx, :] = ctr2 * (1 - t[idx]) + ctr3 * (t[idx])
            fill_ctr = fill_ctr.flatten()
            ctr[0:3] = ctr1
            ctr[-6:-3] = ctr2
            ctr[-3:] = ctr3
            ctr[6:-3] += fill_ctr + data[data_offset:]
        else:
            fill_number = data.shape[0]
            ctr_number = data.shape[0] + 6
            data_offset = 0
            ctr1 = p1_
            ctr2 = p3_
            ctr = np.zeros(ctr_number)
            fill_ctr = np.zeros(fill_number).reshape(-1, 3)
            t = np.linspace(0.1, 0.9, fill_ctr.shape[0])
            for idx in range(fill_ctr.shape[0]):
                fill_ctr[idx, :] = ctr1 * (1 - t[idx]) + ctr2 * (t[idx])
            fill_ctr = fill_ctr.flatten()
            ctr[0:3] = ctr1
            ctr[-3:] = ctr2
            ctr[3:-3] += fill_ctr + data[data_offset:]
        ctr = ctr.reshape(-1, 3).tolist()
        curve = BSpline.Curve()
        curve.degree = len(ctr) - 1
        curve.ctrlpts = ctr
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.vis = VisMPL.VisCurve3D()
        return curve
    return create_bezier


# Code for calculating angles
@nb.jit(nopython=True)
def get_angle(v1,v2):
    return np.arccos(np.dot(-v1,v2))*(180/np.pi)

@nb.jit(nopython=True)
def get_all_angles(vectors):
    angles = np.zeros(vectors.shape[0]-1)
    for i in range(vectors.shape[0]-1):
        angles[i] = get_angle(vectors[i,:],vectors[i+1,:])
    return angles

def get_all_vectors(pts):
    vectors = pts[1:] - pts[:-1]
    vectors = vectors/np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    return vectors

def cost_angles(angles, func=lambda x:-x*(x<0)):
    return np.sum(func(angles))

def cost_bounds(bounds, pts, func=lambda x:max(0,x)):
    bounds_score = 0.0
    for i in range(pts.shape[0]):
        bounds_score += func(bounds[0, 0] - pts[i, 0])
        bounds_score += func(pts[i, 0] - bounds[0, 1])
        bounds_score += func(bounds[1, 0] - pts[i, 1])
        bounds_score += func(pts[i, 1] - bounds[1, 1])
        bounds_score += func(bounds[2, 0] - pts[i, 2])
        bounds_score += func(pts[i, 2] - bounds[2, 1])
    return bounds_score

@nb.jit(nopython=True)
def close_exact(data, point, radius_buffer):
    line_direction = np.zeros((data.shape[0], 3))
    ss = np.zeros(data.shape[0])
    tt = np.zeros(data.shape[0])
    hh = np.zeros(data.shape[0])
    cc = np.zeros((data.shape[0], 3))
    cd = np.zeros(data.shape[0])
    line_distances = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        line_direction[i, :] = (data[i, 3:6] - data[i, 0:3])/np.linalg.norm(data[i, 3:6] - data[i, 0:3])
        ss[i] = np.dot(data[i, 0:3]-point, line_direction[i, :])
        tt[i] = np.dot(point-data[i, 3:6], line_direction[i, :])
        d = np.array([ss[i], tt[i], 0])
        hh[i] = np.max(d)
        diff = point - data[i, 0:3]
        cc[i, :] = np.cross(diff, line_direction[i, :])
        cd[i] = np.linalg.norm(cc[i, :])
        line_distances[i] = np.sqrt(hh[i]**2+cd[i]**2) - data[i, 6] - radius_buffer
    return line_distances

@nb.jit(nopython=True)
def cost_collisions_fast(collision_vessels, collision_centers, collision_lengths, r, pts, radius_buffer):
    collisions = 0
    for i in range(pts.shape[0]):
        tmp = np.argwhere((np.linalg.norm(collision_centers-pts[i,:])-collision_lengths-r-collision_vessels[:,6])<0).flatten()
        dist = close_exact(collision_vessels[tmp,:], pts[i, :], radius_buffer)
        collisions += np.sum((r+radius_buffer-dist)*((r+radius_buffer-dist)>0))
    return collisions

@nb.jit(nopython=True)
def cost_collisions(collision_vessels, r, pts, radius_buffer):
    collisions = 0
    #tmp_c = (collision_vessels[:,0:3] + collision_vessels[:,3:6])/2
    #tmp_l = np.linalg.norm(collision_vessels[:,0:3].T-collision_vessels[:,3:6].T)
    for i in range(pts.shape[0]):
        #tmp = np.argwhere(np.linalg.norm(tmp_c-pts[i,:])-tmp_l-r-collision_vessels[:,6]<0).flatten()
        dist = close_exact(collision_vessels, pts[i, :], radius_buffer)
        collisions += np.sum((r+radius_buffer-dist)*((r+radius_buffer-dist)>0))
    return collisions

def cost(data,curve_generator=None,r=None,p1=None,p2=None,p3=None,p4=None,collision_vessels=None,
         radius_buffer=None,bounds=None,sample_size=None,clamp_first=None,
         clamp_second=None,length_threshold=1.1,angle_threshold=110,collision_centers=None,
         collision_lengths=None):
    curve = curve_generator(data, c1=clamp_first, c2=clamp_second)
    curve.sample_size = sample_size
    curve.evaluate()
    pts = np.array(curve.evalpts)
    cpts = pts[2:-2,:]
    if collision_vessels is not None:
        collision_cost = cost_collisions_fast(collision_vessels, collision_centers, collision_lengths, r, cpts, radius_buffer)
        #collision_cost = cost_collisions(collision_vessels, r, cpts, radius_buffer)
    else:
        collision_cost = 0
    bounds_cost = cost_bounds(bounds, pts)
    vectors = get_all_vectors(pts)
    angles = get_all_angles(vectors)
    angle_cost = cost_angles(angles-angle_threshold)
    spline_length = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    conn_length = np.linalg.norm(p2-p4)
    vessel_1_length = np.linalg.norm(p2-p1)
    vessel_2_length = np.linalg.norm(p4-p3)
    spline_length = spline_length/(conn_length+vessel_1_length+vessel_2_length)
    return (bounds_cost+angle_cost+collision_cost+max(spline_length-length_threshold,0))*spline_length

class connection:
    def __init__(self):
        self.P1 = None
        self.P2 = None
        self.P3 = None
        self.P4 = None
        self.R1 = None
        self.R2 = None
        self.bounds = None
        self.xopt = None
        self.plotting_vessels = None
    def set_solver(self, method='basinhopping'):
        if method == 'basinhopping':
            self.solver = optimize.basinhopping
        elif method == 'shgo':
            self.solver = optimize.shgo
        elif method == 'differential_evolution':
            self.solver = optimize.differential_evolution
        else:
            print('Solver method not implemented')
        return
    def set_vessel(self,id,p1,p2,r):
        if id == 1:
            self.P1 = p1
            self.P2 = p2
            self.R1 = r
        elif id == 2:
            self.P3 = p1
            self.P4 = p2
            self.R2 = r
        return
    def set_bounds(self,bounds):
        self.bounds = bounds
        return
    def set_collision_vessels(self,collision_vessels):
        self.plotting_vessels = collision_vessels
        filter_1_x = np.argwhere(collision_vessels[:, 0] != self.P1[0]).flatten().tolist()
        filter_1_y = np.argwhere(collision_vessels[:, 1] != self.P1[1]).flatten().tolist()
        filter_1_z = np.argwhere(collision_vessels[:, 2] != self.P1[2]).flatten().tolist()
        filter_1 = []
        filter_1.extend(filter_1_x)
        filter_1.extend(filter_1_y)
        filter_1.extend(filter_1_z)
        filter_1 = np.array(list(set(filter_1)))
        if len(filter_1) == 0:
            self.collision_vessels = None
            self.collision_centers = None
            self.collision_lengths = None
            return
        collision_vessels = collision_vessels[filter_1,:]
        filter_2_x = np.argwhere(collision_vessels[:, 0] != self.P3[0]).flatten().tolist()
        filter_2_y = np.argwhere(collision_vessels[:, 1] != self.P3[1]).flatten().tolist()
        filter_2_z = np.argwhere(collision_vessels[:, 2] != self.P3[2]).flatten().tolist()
        filter_2 = []
        filter_2.extend(filter_2_x)
        filter_2.extend(filter_2_y)
        filter_2.extend(filter_2_z)
        filter_2 = np.array(list(set(filter_2)))
        if len(filter_2) == 0:
            self.collision_vessels = None
            self.collision_centers = None
            self.collision_lengths = None
            return
        collision_vessels = collision_vessels[filter_2,:]
        filter_3_x = np.argwhere(collision_vessels[:, 3] != self.P1[0]).flatten().tolist()
        filter_3_y = np.argwhere(collision_vessels[:, 4] != self.P1[1]).flatten().tolist()
        filter_3_z = np.argwhere(collision_vessels[:, 5] != self.P1[2]).flatten().tolist()
        filter_3 = []
        filter_3.extend(filter_3_x)
        filter_3.extend(filter_3_y)
        filter_3.extend(filter_3_z)
        filter_3 = np.array(list(set(filter_3)))
        if len(filter_3) == 0:
            self.collision_vessels = None
            self.collision_centers = None
            self.collision_lengths = None
            return
        collision_vessels = collision_vessels[filter_3,:]
        filter_4_x = np.argwhere(collision_vessels[:, 3] != self.P3[0]).flatten().tolist()
        filter_4_y = np.argwhere(collision_vessels[:, 4] != self.P3[1]).flatten().tolist()
        filter_4_z = np.argwhere(collision_vessels[:, 5] != self.P3[2]).flatten().tolist()
        filter_4 = []
        filter_4.extend(filter_4_x)
        filter_4.extend(filter_4_y)
        filter_4.extend(filter_4_z)
        filter_4 = np.array(list(set(filter_4)))
        if len(filter_4) == 0:
            self.collision_vessels = None
            self.collision_centers = None
            self.collision_lengths = None
        else:
            collision_vessels = collision_vessels[filter_4,:]
            self.collision_vessels = collision_vessels
            self.collision_centers = (collision_vessels[:,0:3]+collision_vessels[:,3:6])/2
            self.collision_lengths = np.linalg.norm(collision_vessels[:,3:6]-collision_vessels[:,0:3],axis=1)
        return
    def solve(self,*args,clamp_first=True,clamp_second=True,sample_size=20,radius_buffer=0.01):
        start = perf_counter()
        if len(args) == 0:
            number_free_points = 2
        else:
            number_free_points = args[0]
        self.radius_buffer = radius_buffer
        self.generator = bezier_factory(self.P1,self.P2,self.P3,self.P4,clamp_first=clamp_first,clamp_second=clamp_second)
        cost_function = lambda data: cost(data,curve_generator=self.generator,r=self.R1,p1=self.P1,p2=self.P2,
                                          p3=self.P3,p4=self.P4,
                                          collision_vessels=self.collision_vessels,radius_buffer=radius_buffer,
                                          sample_size=sample_size,bounds=self.bounds,clamp_first=clamp_first,
                                          clamp_second=clamp_second,collision_centers=self.collision_centers,
                                          collision_lengths=self.collision_lengths)
        if clamp_first and clamp_second:
            clamp_number = 2
        elif clamp_first or clamp_second:
            clamp_number = 1
        else:
            clamp_number = 0
        #if self.xopt is None:
        self.xopt = np.zeros(clamp_number+3*number_free_points)
        #if not self.xopt.shape[0] == clamp_number+3*number_free_points:
        #self.xopt = np.zeros(clamp_number+3*number_free_points)
        shortest_length = np.linalg.norm(self.P2-self.P4)
        lb = np.ones(self.xopt.shape[0])*(-shortest_length/number_free_points)
        for i in range(clamp_number):
            lb[i] = 0
            self.xopt[i] = 1
        ub = np.ones(self.xopt.shape[0])*(shortest_length/number_free_points)
        for i in range(clamp_number):
            ub[i] = 1
        solver_bounds = []
        for b in range(len(lb)):
            solver_bounds.append([lb[b],ub[b]])
        res = self.solver(cost_function, self.xopt)#,bounds=solver_bounds)
        self.xopt = res.x
        self.best = res.fun
        self.curve = self.generator(self.xopt)
        self.curve.sample_size = sample_size
        self.curve.evaluate()
        res.elapsed_time = perf_counter() - start
        return res
    def build_vessels(self,seperate):
        pts = np.array(self.curve.evalpts)
        if seperate:
            sep = (pts.shape[0]-1)//2
            vessels_1 = np.zeros((sep,7))
            vessels_2 = np.zeros((pts.shape[0]-1-sep,7))
            vessels_1_pts = pts[:sep+1,:]
            vessels_2_pts = np.flip(pts[sep:,:],axis=0)
            vessels_1[:,0:3] = vessels_1_pts[:-1,:]
            vessels_1[:,3:6] = vessels_1_pts[1:,:]
            vessels_2[:,0:3] = vessels_2_pts[:-1,:]
            vessels_2[:,3:6] = vessels_2_pts[1:,:]
            vessels_1[:,6] = self.R1
            vessels_2[:,6] = self.R2
        else:
            vessels_1 = np.zeros((pts.shape[0]-1,7))
            vessels_1[:,0:3] = pts[:-1,:]
            vessels_1[:,3:6] = pts[1:,:]
            vessels_1[:,6] = self.R1
            pts_reverse = np.flip(pts,axis=0)
            vessels_2 = np.zeros((pts.shape[0]-1,7))
            vessels_2[:,0:3] = pts_reverse[:-1,:]
            vessels_2[:,3:6] = pts_reverse[1:,:]
            vessels_2[:,6] = self.R2
        self.vessels_1 = vessels_1
        self.vessels_2 = vessels_2
        self.seperate = seperate
        return vessels_1,vessels_2
    def show(self):
        plotter = pv.Plotter()
        if self.seperate:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = np.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i,0:3]+self.vessels_2[i,3:6])/2
                direction = self.vessels_2[i,3:6] - self.vessels_2[i,0:3]
                length = np.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_2[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='blue')
        else:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = np.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
        if self.plotting_vessels is not None:
            for i in range(self.plotting_vessels.shape[0]):
                center = (self.plotting_vessels[i,0:3]+self.plotting_vessels[i,3:6])/2
                direction = self.plotting_vessels[i,3:6] - self.plotting_vessels[i,0:3]
                length = np.linalg.norm(direction)
                direction = direction/length
                radius = self.plotting_vessels[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='black')
        center = (self.P1+self.P2)/2
        direction = (self.P2-self.P1)
        length = np.linalg.norm(direction)
        direction = direction/length
        radius = self.R1
        cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
        plotter.add_mesh(cylinder,color='yellow')
        center = (self.P3+self.P4)/2
        direction = (self.P4-self.P3)
        length = np.linalg.norm(direction)
        direction = direction/length
        radius = self.R1
        cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
        plotter.add_mesh(cylinder,color='yellow')
        return plotter
    def check_bounds(self):
        start = perf_counter()
        pts = np.array(self.curve.evalpts)
        cost_result = cost_bounds(self.bounds,pts)
        print(perf_counter()-start)
        return cost_result
    def check_angles(self):
        start = perf_counter()
        pts = np.array(self.curve.evalpts)
        vectors = get_all_vectors(pts)
        angles = get_all_angles(vectors)
        cost_result = cost_angles(angles)
        print(perf_counter()-start)
        return cost_result
    def check_collisions(self):
        start = perf_counter()
        pts = np.array(self.curve.evalpts)
        pts = pts[2:-2,:]
        if self.collision_vessels is not None:
            collision_cost = cost_collisions_fast(self.collision_vessels, self.collision_centers,self.collision_lengths,self.R1, pts, self.radius_buffer)
            #collision_cost = cost_collisions(self.collision_vessels, self.R1, pts, self.radius_buffer)
        else:
            collision_cost = 0
        print(perf_counter()-start)
        return collision_cost

class simple_connection:
    def __init__(self,forest,network,tree_1,tree_2,connection_id,radius_buffer,
                 clamp_first,clamp_second,seperate):
        self.conn = connection()
        self.forest = forest
        self.network = network
        self.tree_1 = tree_1
        self.tree_2 = tree_2
        self.vessel_1_id = forest.assignments[network][tree_1][connection_id]
        self.vessel_2_id = forest.assignments[network][tree_2][connection_id]
        vessel_1 = forest.networks[network][tree_1].data[self.vessel_1_id]
        vessel_2 = forest.networks[network][tree_2].data[self.vessel_2_id]
        self.conn.set_vessel(1,vessel_1[0:3],vessel_1[3:6],vessel_1[21])
        self.conn.set_vessel(2,vessel_2[0:3],vessel_2[3:6],vessel_2[21])
        tree_1_ids = np.argwhere(forest.networks[network][tree_1].data[:, 15] > -1).flatten()
        tree_2_ids = np.argwhere(forest.networks[network][tree_2].data[:, 15] > -1).flatten()
        #tree_1_ids = np.delete(tree_1_ids,int(vessel_1[17]))
        #tree_2_ids = np.delete(tree_2_ids,int(vessel_2[17]))
        tree_1_data = forest.networks[network][tree_1].data[tree_1_ids, :]
        tree_2_data = forest.networks[network][tree_2].data[tree_2_ids, :]
        collision_vessels = np.zeros((tree_1_data.shape[0]+tree_2_data.shape[0],7))
        collision_vessels[:tree_1_data.shape[0],0:3] = tree_1_data[:,0:3]
        collision_vessels[:tree_1_data.shape[0],3:6] = tree_1_data[:,3:6]
        collision_vessels[:tree_1_data.shape[0],6] = tree_1_data[:,21]
        collision_vessels[tree_1_data.shape[0]:,0:3] = tree_2_data[:,0:3]
        collision_vessels[tree_1_data.shape[0]:,3:6] = tree_2_data[:,3:6]
        collision_vessels[tree_1_data.shape[0]:,6] = tree_2_data[:,21]
        self.radius_buffer = radius_buffer
        self.clamp_first = clamp_first
        self.clamp_second = clamp_second
        self.conn.set_collision_vessels(collision_vessels)
        self.bounds = np.array([forest.boundary.x_range, forest.boundary.y_range, forest.boundary.z_range])
        self.seperate = seperate
        self.edge_1 = vessel_1
        self.edge_2 = vessel_2
    def set_solver(self,*args,method='basinhopping'):
        self.conn.set_solver(method=method)
        self.conn.set_bounds(self.bounds)
    def solve(self,*args):
        if len(args) > 0:
            number_ctl_points = args[0]
        else:
            number_ctl_points = 2
        self.res = self.conn.solve(number_ctl_points)
        return self.res
    def build(self):
        self.vessels_1,self.vessels_2 = self.conn.build_vessels(self.seperate)
    def show(self):
        plotter = self.conn.show()
        plotter.show()
    def export_vessels(self):
        if self.seperate:
            vessels_tree_1 = np.ones((self.vessels_1.shape[0],self.forest.networks[self.network][self.tree_1].data.shape[1]))*-1
            vessels_tree_2 = np.ones((self.vessels_2.shape[0],self.forest.networks[self.network][self.tree_2].data.shape[1]))*-1
            vessels_tree_1[:,0:3] = self.vessels_1[:,0:3]
            vessels_tree_1[:,3:6] = self.vessels_1[:,3:6]
            vessels_tree_2[:,0:3] = self.vessels_2[:,0:3]
            vessels_tree_2[:,3:6] = self.vessels_2[:,3:6]
            tree_1_dirs = vessels_tree_1[:,3:6] - vessels_tree_1[:,0:3]
            tree_2_dirs = vessels_tree_2[:,3:6] - vessels_tree_2[:,0:3]
            vessels_tree_1[:,20] = np.linalg.norm(tree_1_dirs,axis=1)
            vessels_tree_2[:,20] = np.linalg.norm(tree_2_dirs,axis=1)
            vessels_tree_1[:,12:15] = tree_1_dirs/vessels_tree_1[:,20].reshape(-1,1)
            vessels_tree_2[:,12:15] = tree_2_dirs/vessels_tree_2[:,20].reshape(-1,1)
            vessels_tree_1[:,21] = self.vessels_1[:,6]
            vessels_tree_2[:,21] = self.vessels_2[:,6]
            vessels_tree_1[:,22] = self.edge_1[22]
            vessels_tree_2[:,22] = self.edge_2[22]
            self.vessels = [vessels_tree_1,vessels_tree_2]
        else:
            print('not implemented yet')
class tree_connections:
    def __init__(self,forest,network,tree_1,tree_2,radius_buffer,clamp_first=True,clamp_second=True,seperate=True):
        self.forest = forest
        self.network = network
        self.tree_1 = tree_1
        self.tree_2 = tree_2
        self.radius_buffer = radius_buffer
        self.extra_collision_vessels = None
        self.connection_solutions = []
        for i in range(self.forest.assignments[self.network][self.tree_1].shape[0]):
            tmp_connection = simple_connection(self.forest,self.network,self.tree_1,self.tree_2,i,self.radius_buffer,clamp_first,clamp_second,seperate)
            self.connection_solutions.append(tmp_connection)
    def solve(self,clamp_first=True,clamp_second=True,seperate=True,max_ctrl_points=5):
        connection_solutions = []
        continue_optimize = []
        num_ctrl_points = 1
        for i in trange(self.forest.assignments[self.network][self.tree_1].shape[0]):
            tmp_connection = self.connection_solutions[i]
            tmp_connection.clamp_first = clamp_first
            tmp_connection.clamp_second = clamp_second
            tmp_connection.seperate = seperate
            tmp_connection.set_solver()
            for j in range(len(connection_solutions)):
                if connection_solutions[j].seperate:
                    collision_vessels = tmp_connection.conn.collision_vessels
                    collision_vessels = np.vstack((collision_vessels,connection_solutions[j].vessels_1,
                                                   connection_solutions[j].vessels_2))
                    tmp_connection.conn.set_collision_vessels(collision_vessels)
                else:
                    collision_vessels = tmp_connection.conn.collision_vessels
                    collision_vessels = np.vstack((collision_vessels,connection_solutions[j].vessels_1))
                    tmp_connection.conn.set_collision_vessels(collision_vessels)
            if self.extra_collision_vessels is not None:
                collision_vessels = tmp_connection.conn.collision_vessels
                collision_vessels = np.vstack((collision_vessels,self.extra_collision_vessels))
                tmp_connection.conn.set_collision_vessels(collision_vessels)
            tmp_connection.solve(num_ctrl_points)
            if tmp_connection.res.fun > 0:
                continue_optimize.append(i)
            tmp_connection.build()
            connection_solutions.append(tmp_connection)
        while len(continue_optimize) > 0 and num_ctrl_points <= max_ctrl_points:
            num_ctrl_points += 1
            new_continue_optimize = []
            print("Design space increased to {} control points for connections {}".format(num_ctrl_points,continue_optimize))
            while len(continue_optimize) > 0:
                i = continue_optimize.pop(0)
                solved_vessels = list(set(list(range(len(connection_solutions)))) - set(continue_optimize))
                tmp_connection = self.connection_solutions[i]
                for j in solved_vessels:
                    if connection_solutions[j].seperate:
                        collision_vessels = tmp_connection.conn.collision_vessels
                        collision_vessels = np.unique(np.vstack((collision_vessels,connection_solutions[j].vessels_1,
                                                       connection_solutions[j].vessels_2)),axis=0)
                        tmp_connection.conn.set_collision_vessels(collision_vessels)
                    else:
                        collision_vessels = tmp_connection.conn.collision_vessels
                        collision_vessels = np.unique(np.vstack((collision_vessels,connection_solutions[j].vessels_1)),axis=0)
                        tmp_connection.conn.set_collision_vessels(collision_vessels)
                if self.extra_collision_vessels is not None:
                    collision_vessels = tmp_connection.conn.collision_vessels
                    collision_vessels = np.vstack((collision_vessels, self.extra_collision_vessels))
                    tmp_connection.conn.set_collision_vessels(collision_vessels)
                tmp_connection.solve(num_ctrl_points)
                if tmp_connection.res.fun > 0:
                    new_continue_optimize.append(i)
                tmp_connection.build()
                if tmp_connection.res.fun < connection_solutions[i].res.fun:
                    connection_solutions[i] = tmp_connection
            continue_optimize = new_continue_optimize
        self.connection_solutions = connection_solutions
        return
    def add_collision_vessels(self,vessels):
        if self.extra_collision_vessels is not None:
            self.extra_collision_vessels = np.unique(np.vstack((self.extra_collision_vessels,vessels)),axis=0)
        else:
            self.extra_collision_vessels = vessels
    def build_vessels(self):
        for i in range(len(self.connection_solutions)):
            self.connection_solutions[i].export_vessels()

class network_connections:
    def __init__(self,forest,network,radius_buffer):
        self.forest = forest
        self.network = network
        self.radius_buffer = radius_buffer
    def solve(self,order=None):
        solutions = []
        if order is None:
            order = list(range(len(self.forest.trees_per_network[self.network])))
        for i in range(1,len(order)):
            if i == 1:
                tmp_tree_connections = tree_connections(self.forest,self.network,order[0],order[i],self.radius_buffer)
                for k in range(len(tmp_tree_connections.connection_solutions)):
                    tmp_tree_connections.connection_solutions[k].conn.P1 = solutions[0].connection_solutions[k].vessels_1[-1,0:3]
                    tmp_tree_connections.connection_solutions[k].conn.P2 = solutions[0].connection_solutions[k].vessels_1[-1,3:6]
                    tmp_tree_connections.connection_solutions[k].conn.R1 = solutions[0].connection_solutions[k].vessels_1[-1,6]
                remaining_trees = set(order)-set([order[0],order[i]])
                for j in remaining_trees:
                    vessels = np.zeros((self.forest.networks[self.network][j].data.shape[0],7))
                    vessels[:,0:3] = self.forest.networks[self.network][j].data[:,0:3]
                    vessels[:,3:6] = self.forest.networks[self.network][j].data[:,3:6]
                    vessels[:,6] = self.forest.networks[self.network][j].data[:,21]
                    tmp_tree_connections.add_collision_vessels(vessels)
                tmp_tree_connections.solve()
            else:
                solved_trees = list(range(1,i))
                remaining_trees = set(list(range(len(order)))) - set(solved_trees)
                tmp_tree_connections = tree_connections(self.forest,self.network,order[0],order[i],self.radius_buffer,
                                                        clamp_first=False,clamp_second=True,seperate=False)
                for k in range(len(tmp_tree_connections.connection_solutions)):
                    tmp_tree_connections.connection_solutions[k].conn.P1 = solutions[0].connection_solutions[k].vessels_1[-1,0:3]
                    tmp_tree_connections.connection_solutions[k].conn.P2 = solutions[0].connection_solutions[k].vessels_1[-1,3:6]
                    tmp_tree_connections.connection_solutions[k].conn.R1 = solutions[0].connection_solutions[k].vessels_1[-1,6]
                for j in solved_trees:
                    for conn in solutions[j].connection_solutions:
                        if conn.seperate:
                            vessels = np.zeros((conn.vessels_1.shape[0]+conn.vessels_2.shape[0],7))
                            vessels[0:conn.vessels_1.shape[0],:] = conn.vessels_1
                            vessels[conn.vessels_1.shape[0]:,:] = conn.vessels_2
                        else:
                            vessels = conn.vessels_1
                        tmp_tree_connections.add_collision_vessels(vessels)
                for j in remaining_trees:
                    vessels = np.zeros((self.forest.networks[self.network][j].data.shape[0],7))
                    vessels[:,0:3] = self.forest.networks[self.network][j].data[:,0:3]
                    vessels[:,3:6] = self.forest.networks[self.network][j].data[:,3:6]
                    vessels[:,6] = self.forest.networks[self.network][j].data[:,21]
                    tmp_tree_connections.add_collision_vessels(vessels)
                tmp_tree_connections.solve()
            solutions.append(tmp_tree_connections)

"""
pts = np.array([[-2,0,0],[-1,0,0],[2,0,0],[1,0,0]])
bounds = np.array([[-15,15],[-15,15],[-15,15]])
gen1 = bezier_factory(pts[0,:],pts[1,:],pts[2,:],pts[3,:])
gen2 = bezier_factory(pts[0,:],pts[1,:],pts[2,:],pts[3,:],clamp_first=False)
gen3 = bezier_factory(pts[0,:],pts[1,:],pts[2,:],pts[3,:],clamp_first=False,clamp_second=False)

c = connection()
c.set_vessel(1,pts[0,:],pts[1,:],0.1)
c.set_vessel(2,pts[2,:],pts[3,:],0.1)
c.set_solver()
c.set_bounds(bounds)
c.set_collision_vessels(None)

pts1 = 10*np.array([[0,0.25,0.5],[0.5,0.25,0.5],[0.25,1,0.5],[0.25,0.5,0.5]])
c1 = connection()
c1.set_vessel(1,pts1[0,:],pts1[1,:],0.1)
c1.set_vessel(2,pts1[2,:],pts1[3,:],0.1)
c1.set_solver()
c1.set_bounds(bounds)
collision_vessels = 15*np.random.random((15,7))
collision_vessels[:,6] = 0.05
c1.set_collision_vessels(collision_vessels)

pts2 = 10*np.array([[0.25,0,0],[0.5,0,0],[-0.25,0,0],[-0.5,0,0]])
c1 = connection()
c1.set_vessel(1,pts2[0,:],pts2[1,:],0.1)
c1.set_vessel(2,pts2[2,:],pts2[3,:],0.1)
c1.set_solver()
c1.set_bounds(bounds)
c1.set_collision_vessels(None)
"""