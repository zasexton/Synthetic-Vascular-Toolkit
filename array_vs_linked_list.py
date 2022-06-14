"""
Code to evaluate cache hits and misses for svcco
"""
import svcco
import numpy as np
from binarytree import Node
from time import time, perf_counter
from functools import lru_cache
import matplotlib.pyplot as plt
import os
import cProfile
import pstats
from memory_profiler import memory_usage
from copy import deepcopy

os.environ["PATH"] += ';C:\\Program Files (x86)\\Graphviz\\bin'
os.environ["PATH"] += ';C:\\Program Files (x86)\\Graphviz'
################################################
# CODE FOR LINKED LIST CONTAINER
################################################

def Vessel(data):
    node                = Node(int(data[-1]))
    node.data           = np.array(deepcopy(data.reshape(1,-1).tolist()))
    node.proximal_point = data[0:3]
    node.distal_point   = data[3:6]
    node.radius         = data[21]
    node.length         = data[20]
    node.left_bif       = data[23]
    node.right_bif      = data[24]
    node.daughter_0     = int(data[15])
    node.daughter_1     = int(data[16])
    node.parent         = int(data[17])
    node.id             = int(data[-1])
    return node

class Tree:
    # Constructor
    def __init__(self):
        self.root = None
        self.all_nodes = []
        self.all_data  = []
        self.all_data_list = []
    def insert(self,data):
        new_vessel = Vessel(data)
        if new_vessel.id == 0:
            new_vessel.parent_node = None
            self.root = new_vessel
            self.all_nodes.append(new_vessel)
            self.all_data.append(np.array(deepcopy(new_vessel.data[0,:].tolist())))
            self.all_data_list.append(new_vessel.data[0,:].tolist())
        else:
            queue = [self.root]
            while queue:
                vessel = queue.pop(0)
                if vessel.id == new_vessel.parent:
                    if vessel.daughter_0 == new_vessel.id:
                        new_vessel.parent_node = vessel
                        vessel.left = new_vessel
                        self.all_nodes.append(new_vessel)
                        self.all_data.append(np.array(deepcopy(new_vessel.data[0,:].tolist())))
                        self.all_data_list.append(new_vessel.data[0,:].tolist())
                        break
                    if vessel.daughter_1 == new_vessel.id:
                        new_vessel.parent_node = vessel
                        vessel.right = new_vessel
                        self.all_nodes.append(new_vessel)
                        self.all_data.append(np.array(deepcopy(new_vessel.data[0,:].tolist())))
                        self.all_data_list.append(new_vessel.data[0,:].tolist())
                        break
                if vessel.daughter_0 > 0:
                    queue.append(vessel.left)
                if vessel.daughter_1 > 0:
                    queue.append(vessel.right)

    def find_closest(self,point,method='fbs'):
        best_value = np.inf
        best_id    = None
        if method == 'fbs':
            queue = [self.root]
            while queue:
                vessel = queue.pop(0)
                _,dist = svcco.branch_addition.close.close_exact(vessel.data,point)
                if dist < best_value:
                    best_value = dist
                    best_id    = vessel.id
                if vessel.daughter_0 > 0:
                    queue.append(vessel.left)
                if vessel.daughter_1 > 0:
                    queue.append(vessel.right)
        elif method == 'all':
            for vessel in self.all_nodes:
                _,dist = svcco.branch_addition.close.close_exact(vessel.data,point)
                if dist < best_value:
                    best_value = dist
                    best_id    = vessel.id
        elif method == 'vec':
            #data = [v.data[0,:] for v in self.root.preorder]
            best_id,best_value = svcco.branch_addition.close.close_binary_vectorize(self.all_data,point)
            best_id = int(best_id)
        return best_id,best_value

    def update_radii(self,factor):
        self.root.radius *= factor
        queue = [self.root]
        while queue:
            vessel = queue.pop(0)
            if vessel.daughter_0 > 0:
                vessel.left.radius = vessel.radius*vessel.left_bif
                queue.append(vessel.left)
            elif vessel.daughter_1 > 0:
                vessel.right.radius = vessel.radius*vessel.right_bif
                queue.append(vessel.right)

    def find_collision(self,new_vessel,method='fbs'):
        if method == 'fbs':
            collisions = False
            queue = [self.root]
            while queue:
                vessel = queue.pop(0)
                if len( svcco.collision.sphere_proximity.sphere_proximity_testing(vessel.data,new_vessel)) > 0:
                    collisions = svcco.collision.obb.obb(vessel.data,new_vessel)
                    if collisions:
                        return True
                if vessel.daughter_0 > 0:
                    queue.append(vessel.left)
                if vessel.daughter_1 > 0:
                    queue.append(vessel.right)
        elif method == 'all':
            collisions = []
            for vessel in self.all_nodes:
                if len(svcco.collision.sphere_proximity.sphere_proximity_testing(vessel.data,new_vessel)) > 0:
                    collisions.append(svcco.collision.obb.obb(vessel.data,new_vessel))
            return any(collisions)
        #elif method == 'vec':
        #    #data = [v.data[0,:] for v in self.root.preorder]
        #    best_id,best_value = svcco.branch_addition.close.close_binary_vectorize(self.all_data,point)
        #    best_id = int(best_id)
        #return collision

def build(data):
    t = Tree()
    t.insert(data[0,:])
    queue = []
    if data[0,15] > 0:
        queue.append(int(data[0,15]))
    if data[0,16] > 0:
        queue.append(int(data[0,16]))
    while queue:
        id = queue.pop(0)
        t.insert(data[id,:])
        if data[id,15] > 0:
            queue.append(int(data[id,15]))
        if data[id,16] > 0:
            queue.append(int(data[id,16]))
    return t
#####################################################
# CODE FOR BUILDING SVCCO Tree
#####################################################
import pyvista as pv

cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(10*cube.points,normals=cube.point_normals)
s.solve()
s.build()

t = svcco.tree()
t.set_boundary(s)
t.set_root()
t.convex = True
#####################################################
# WRAPPERS FOR TESTING CALLS
#####################################################

# FIND CLOSEST VESSEL
def test_sv_tree(sv_tree,point):
    start = perf_counter()
    best_id,best_dist = svcco.branch_addition.close.close_exact(sv_tree.data,point)
    elapsed = perf_counter() - start
    best_id = best_id[0]
    return elapsed,best_id

"""
def test_sv_tree_calls(t,point):
    cProfile.run("svcco.branch_addition.close.close_exact(t.data,point)","sv_stats")
    p = pstats.Stats("sv_stats")
    return p.total_calls
"""

def test_sv_tree_mem(t):
    mem = t.data.nbytes
    return mem*10e-9


def test_binary_tree(binary,point,meth='fbs'):
    start = perf_counter()
    best_id,best_dist = binary.find_closest(point,method=meth)
    elapsed = perf_counter() - start
    return elapsed,best_id

def test_binary_mem(binary):
    locs = []
    for vessel in binary.root.preorder:
        #locs.append(vessel.data.__array_interface__['data'][0])
        locs.append(vessel.data.ctypes.data)
    mem = max(locs)-min(locs)
    if mem == 0:
        mem = binary.root.data.nbytes
    return mem*10e-9
"""
def test_binary_calls(binary,point,meth='fbs'):
    cProfile.run("binary.find_closest(point,method='{}')".format(meth),"bin_stats")
    p = pstats.Stats("bin_stats")
    return p.total_calls

def test_binary_mem(binary,point,meth='fbs'):
    mem = memory_usage((binary.find_closest,(point,),{'method':meth}))
    return np.mean(mem)
"""

# UPDATE TREE RADII (rescale)

def test_sv_radii_time(t,factor):
    start = perf_counter()
    t.data[:,21] = t.data[0,21]*factor*t.data[:,28]
    elapsed = perf_counter()-start
    return elapsed

def test_binary_radii_time(binary,factor):
    start = perf_counter()
    binary.update_radii(factor)
    elapsed = perf_counter()-start
    return elapsed

# TEST TREE COLLISIONS
def test_sv_collision_time(t,vessel,meth='fbs'):
    if meth == 'fbs':
        start = perf_counter()
        n = svcco.collision.sphere_proximity.sphere_proximity_testing(t.data,vessel)
        collision = svcco.collision.obb.obb(t.data[n,:],vessel)
        elapsed = perf_counter()-start
    else:
        start = perf_counter()
        n = svcco.collision.sphere_proximity.sphere_proximity_testing(t.data,vessel)
        collision = svcco.collision.obb.obbc(t.data[n,:],vessel)
        elapsed = perf_counter()-start
    return elapsed

def test_binary_collision_time(binary,vessel,meth='fbs'):
    start = perf_counter()
    collision = binary.find_collision(vessel,method=meth)
    elapsed = perf_counter() - start
    return elapsed


#####################################################
# CODE FOR TESTING PERFORMANCE
#####################################################

binary = build(t.data)
point = np.array([0.2,0.2,0.2])

def test(t,binary,point,reps=20):
    sv_tree_perf = []
    binary_tree_perf_fbs = []
    binary_tree_perf_all = []
    binary_tree_perf_vec = []
    #sv_mem = test_sv_tree_mem(t,point)
    #bin_mem = test_binary_mem(binary,point,meth='vec')
    elapsed_binary_vec,best_binary = test_binary_tree(binary,point,meth='vec')
    for i in range(reps):
        point = np.random.random(3)*10-5
        elapsed_sv,best_sv = test_sv_tree(t,point)
        sv_tree_perf.append(elapsed_sv)
        elapsed_binary_fbs,best_binary = test_binary_tree(binary,point)
        binary_tree_perf_fbs.append(elapsed_binary_fbs)
        elapsed_binary_all,best_binary = test_binary_tree(binary,point,meth='all')
        binary_tree_perf_all.append(elapsed_binary_all)
        elapsed_binary_vec,best_binary = test_binary_tree(binary,point,meth='vec')
        binary_tree_perf_vec.append(elapsed_binary_vec)
        if best_sv != best_binary:
            print('ERROR: ANSWERS DO NOT MATCH {} != {}'.format(best_sv,best_binary))
            break
    #sv_mem = test_sv_tree_mem(t,point)
    #bin_mem = test_binary_mem(binary,point,meth='vec')
    #sv_calls = test_sv_tree_calls(t,point)
    #bin_calls = test_binary_calls(binary,point,meth='vec')
    print('PERFORMANCE, REPITITIONS = {}'.format(reps))
    print('-------------------------------------------------------')
    print('SV TREE PERFORMANCE             : {}'.format(np.mean(sv_tree_perf)))
    print('BINARY TREE PERFORMANCE METH=FBS: {}'.format(np.mean(binary_tree_perf_fbs)))
    print('BINARY TREE PERFORMANCE METH=ALL: {}'.format(np.mean(binary_tree_perf_all)))
    print('BINARY TREE PERFORMANCE METH=VEC: {}'.format(np.mean(binary_tree_perf_vec)))
    print('-------------------------------------------------------')
    #print('INSTRUCTION PERFORMANCE')
    #print('-------------------------------------------------------')
    #print('SV TREE PERFORMANCE    : {}'.format(sv_calls))
    #print('BINARY TREE PERFORMANCE: {}'.format(bin_calls))
    #print('-------------------------------------------------------')
    #print('MEMORY PERFORMANCE')
    #print('-------------------------------------------------------')
    #print('SV TREE PERFORMANCE    : {}'.format(sv_mem))
    #print('BINARY TREE PERFORMANCE: {}'.format(bin_mem))
    #print('-------------------------------------------------------')
    return np.mean(sv_tree_perf),np.mean(binary_tree_perf_fbs),np.mean(binary_tree_perf_all),np.mean(binary_tree_perf_vec) #,sv_calls,bin_calls,sv_mem,bin_mem

def test_radii(t,binary,reps=20):
    sv_time = []
    binary_time = []
    for rep in range(reps):
        factor = np.random.random(1)[0]
        sv_time.append(test_sv_radii_time(t,factor))
        binary_time.append(test_binary_radii_time(binary,factor))
    print('PERFORMANCE, REPITITIONS = {}'.format(reps))
    print('-------------------------------------------------------')
    print('SV TREE PERFORMANCE    : {}'.format(np.mean(sv_time)))
    print('BINARY TREE PERFORMANCE: {}'.format(np.mean(binary_time)))
    print('-------------------------------------------------------')
    return np.mean(sv_time), np.mean(binary_time)

def test_collision(t,binary,reps=20):
    sv_time = []
    sv_all = []
    binary_time_fbs = []
    binary_time_all = []
    for rep in range(reps):
        P0 = np.random.random(3)*10-5
        P1 = np.random.random(3)*10-5
        L  = np.linalg.norm(P0-P1)
        R  = np.random.random(1)*2
        vessel = np.ones(30)*-1
        vessel[0:3] = P0
        vessel[3:6] = P1
        vessel[20]  = L
        vessel[21]  = R
        test_sv_collision_time(t,vessel)
        test_sv_collision_time(t,vessel,meth='all')
        test_binary_collision_time(binary,vessel)
        test_binary_collision_time(binary,vessel,meth='all')
        sv_time.append(test_sv_collision_time(t,vessel))
        sv_all.append(test_sv_collision_time(t,vessel,meth='all'))
        binary_time_fbs.append(test_binary_collision_time(binary,vessel))
        binary_time_all.append(test_binary_collision_time(binary,vessel,meth='all'))
    print('PERFORMANCE, REPITITIONS = {}'.format(reps))
    print('-------------------------------------------------------')
    print('SV TREE PERFORMANCE         : {}'.format(np.mean(sv_time)))
    print('SV TREE PERFORMANCE ALL     : {}'.format(np.mean(sv_all)))
    print('BINARY TREE PERFORMANCE  FBS: {}'.format(np.mean(binary_time_fbs)))
    print('BINARY TREE PERFORMANCE  ALL: {}'.format(np.mean(binary_time_all)))
    print('-------------------------------------------------------')
    return np.mean(sv_time), np.mean(sv_all), np.mean(binary_time_fbs), np.mean(binary_time_all)

def test_range(start=1,stop=1000,incr=10,show=False):
    t = svcco.tree()
    t.set_boundary(s)
    t.set_root()
    t.convex = True
    binary = build(t.data)
    SIZE     = [start]
    SV       = []
    BIN_FBS  = []
    BIN_ALL  = []
    BIN_VEC  = []
    #SV_CALLS = []
    #BIN_CALLS= []
    #SV_MEM   = []
    #BIN_MEM  = []
    while SIZE[-1] < stop:
        sv_perf,bin_perf_fbs,bin_perf_all,bin_perf_vec = test(t,binary,point)
        SV.append(sv_perf)
        BIN_FBS.append(bin_perf_fbs)
        BIN_ALL.append(bin_perf_all)
        BIN_VEC.append(bin_perf_vec)
        #SV_CALLS.append(sv_calls)
        #BIN_CALLS.append(bin_calls)
        #SV_MEM.append(sv_mem)
        #BIN_MEM.append(bin_mem)
        t.n_add(incr)
        binary = build(t.data)
        SIZE.append(SIZE[-1]+incr)
    SIZE.pop(-1)
    if show:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(SIZE,SV,label="SV continuous")
        ax.plot(SIZE,BIN_FBS,label="Binary Tree FBS")
        #ax.plot(SIZE,BIN_ALL,label="Binary Tree Precollected")
        ax.plot(SIZE,BIN_VEC,label="Binary Tree Vectorized FBS")
        ax.set_xlabel('Tree Size')
        ax.set_ylabel('Time (seconds)')
        plt.legend()
        plt.show()
    return SIZE,SV,BIN_FBS,BIN_ALL,BIN_VEC

def test_range_radii(start=1,stop=1000,incr=10,show=False):
    t = svcco.tree()
    t.set_boundary(s)
    t.set_root()
    t.convex = True
    binary = build(t.data)
    SIZE = [start]
    SV = []
    BIN = []
    while SIZE[-1] < stop:
        sv_t,bin_t = test_radii(t,binary)
        SV.append(sv_t)
        BIN.append(bin_t)
        t.n_add(incr)
        SIZE.append(SIZE[-1]+incr)
        binary = build(t.data)
    SIZE.pop(-1)
    if show:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(SIZE,SV,label="SV")
        ax.plot(SIZE,BIN,label="Binary Tree")
        ax.set_xlabel("Tree Size")
        ax.set_ylabel("Time (seconds)")
        plt.legend()
        plt.show()
    return SIZE,SV,BIN

def test_range_collision(start=1,stop=1000,incr=10,show=False):
    t = svcco.tree()
    t.set_boundary(s)
    t.set_root()
    t.convex = True
    binary = build(t.data)
    SIZE = [start]
    SV = []
    SV_ALL = []
    BIN_FBS = []
    BIN_ALL = []
    SV_MEM  = []
    BIN_MEM = []
    while SIZE[-1] < stop:
        sv_t,sv_all,bin_fbs,bin_all = test_collision(t,binary)
        SV.append(sv_t)
        SV_ALL.append(sv_all)
        BIN_FBS.append(bin_fbs)
        BIN_ALL.append(bin_all)
        SV_MEM.append(test_sv_tree_mem(t))
        BIN_MEM.append(test_binary_mem(binary))
        t.n_add(incr)
        SIZE.append(SIZE[-1]+incr)
        binary = build(t.data)
    SIZE.pop(-1)
    if show:
        fig = plt.figure()
        ax  = fig.add_subplot(121)
        ax.plot(SIZE,SV,label="SV")
        ax.plot(SIZE,SV_ALL,label="SV all")
        ax.plot(SIZE,BIN_FBS,label="Binary Tree FBS")
        ax.plot(SIZE,BIN_ALL,label="Binary Tree ALL")
        ax.set_xlabel("Tree Size")
        ax.set_ylabel("Time (seconds)")
        ax1  = fig.add_subplot(122)
        ax1.plot(SIZE,SV_MEM,label="SV Memory")
        ax1.plot(SIZE,BIN_MEM,label="Binary Memory")
        ax1.set_xlabel("Tree Size")
        ax1.set_ylabel("Memory Span Locality(Gb)")
        plt.legend()
        plt.show()
    return SIZE,SV,SV_ALL,BIN_FBS,BIN_ALL,SV_MEM,BIN_MEM

def test_meta(num,stop=1000,incr=10):

    size,sv,bin_fbs,bin_all,bin_vec = test_range(stop=stop,incr=incr)
    SIZE = np.array(size)
    SV   = np.array(sv)
    BIN_FBS = np.array(bin_fbs)
    BIN_ALL = np.array(bin_all)
    BIN_VEC = np.array(bin_vec)
    for i in range(1,num):
        size,sv,bin_fbs,bin_all,bin_vec = test_range(stop=stop,incr=incr)
        SV      = np.vstack((SV,np.array(sv)))
        BIN_FBS = np.vstack((BIN_FBS,np.array(bin_fbs)))
        BIN_ALL = np.vstack((BIN_ALL,np.array(bin_all)))
        BIN_VEC = np.vstack((BIN_VEC,np.array(bin_vec)))
    SV_MEAN      = np.mean(SV,axis=0)
    BIN_FBS_MEAN = np.mean(BIN_FBS,axis=0)
    BIN_ALL_MEAN = np.mean(BIN_ALL,axis=0)
    BIN_VEC_MEAN = np.mean(BIN_VEC,axis=0)
    SV_STD       = np.std(SV,axis=0)
    BIN_FBS_STD  = np.std(BIN_FBS,axis=0)
    BIN_ALL_STD  = np.std(BIN_ALL,axis=0)
    BIN_VEC_STD  = np.std(BIN_VEC,axis=0)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(SIZE,SV_MEAN,label="SV")
    ax.fill_between(SIZE,SV_MEAN-SV_STD,SV_MEAN+SV_STD,alpha=0.5)
    ax.plot(SIZE,BIN_FBS_MEAN,label="Binary Tree FBS")
    ax.fill_between(SIZE,BIN_FBS_MEAN-BIN_FBS_STD,BIN_FBS_MEAN+BIN_FBS_STD,alpha=0.5)
    ax.plot(SIZE,BIN_VEC_MEAN,label="Binary Tree Vectorized FBS")
    ax.fill_between(SIZE,BIN_VEC_MEAN-BIN_VEC_STD,BIN_VEC_MEAN+BIN_VEC_STD,alpha=0.5)
    ax.set_xlabel('Tree Size')
    ax.set_ylabel('Time (seconds)')
    plt.legend(loc='upper left')
    plt.savefig('C:\\Users\\Zack\\Downloads\\FIGURE 1 PART C.png')

    size,sv,bin = test_range_radii(stop=stop,incr=incr)
    SIZE = np.array(size)
    SV   = np.array(sv)
    BIN = np.array(bin)
    for i in range(1,num):
        size,sv,bin = test_range_radii(stop=stop,incr=incr)
        SV      = np.vstack((SV,np.array(sv)))
        BIN     = np.vstack((BIN,np.array(bin)))
    SV_MEAN      = np.mean(SV,axis=0)
    BIN_MEAN     = np.mean(BIN,axis=0)
    SV_STD       = np.std(SV,axis=0)
    BIN_STD      = np.std(BIN,axis=0)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(SIZE,SV_MEAN,label="SV")
    ax.fill_between(SIZE,SV_MEAN-SV_STD,SV_MEAN+SV_STD,alpha=0.5)
    ax.plot(SIZE,BIN_MEAN,label="Binary Tree")
    ax.fill_between(SIZE,BIN_MEAN-BIN_STD,BIN_MEAN+BIN_STD,alpha=0.5)
    ax.set_xlabel('Tree Size')
    ax.set_ylabel('Time (seconds)')
    plt.legend(loc='upper left')
    plt.savefig('C:\\Users\\Zack\\Downloads\\FIGURE 1 PART D.png')

    size,sv,sv_all,bin_fbs,bin_all,sv_mem,bin_mem = test_range_collision(stop=stop,incr=incr)
    SIZE = np.array(size)
    SV   = np.array(sv)
    SV_ALL = np.array(sv_all)
    BIN_FBS = np.array(bin_fbs)
    BIN_ALL = np.array(bin_all)
    SV_MEM  = np.array(sv_mem)
    BIN_MEM = np.array(bin_mem)
    for i in range(1,num):
        size,sv,sv_all,bin_fbs,bin_all,sv_mem,bin_mem = test_range_collision(stop=stop,incr=incr)
        SV      = np.vstack((SV,np.array(sv)))
        SV_ALL  = np.vstack((SV_ALL,np.array(sv)))
        BIN_FBS = np.vstack((BIN_FBS,np.array(bin_fbs)))
        BIN_ALL = np.vstack((BIN_ALL,np.array(bin_all)))
        SV_MEM  = np.vstack((SV_MEM,np.array(sv_mem)))
        BIN_MEM = np.vstack((BIN_MEM,np.array(bin_mem)))
    SV_MEAN      = np.mean(SV,axis=0)
    SV_ALL_MEAN  = np.mean(SV_ALL,axis=0)
    BIN_FBS_MEAN = np.mean(BIN_FBS,axis=0)
    BIN_ALL_MEAN = np.mean(BIN_ALL,axis=0)
    SV_MEM_MEAN  = np.mean(SV_MEM,axis=0)
    BIN_MEM_MEAN = np.mean(BIN_MEM,axis=0)
    SV_STD       = np.std(SV,axis=0)
    SV_ALL_STD   = np.std(SV_ALL,axis=0)
    BIN_FBS_STD  = np.std(BIN_FBS,axis=0)
    BIN_ALL_STD  = np.std(BIN_ALL,axis=0)
    SV_MEM_STD   = np.std(SV_MEM,axis=0)
    BIN_MEM_STD  = np.std(BIN_MEM,axis=0)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(SIZE,SV_MEAN,label="SV FBS")
    ax.fill_between(SIZE,SV_MEAN-SV_STD,SV_MEAN+SV_STD,alpha=0.5)
    ax.plot(SIZE,SV_ALL_MEAN,label="SV all")
    ax.fill_between(SIZE,SV_ALL_MEAN-SV_ALL_STD,SV_ALL_MEAN+SV_ALL_STD,alpha=0.5)
    ax.plot(SIZE,BIN_FBS_MEAN,label="Binary Tree FBS")
    ax.fill_between(SIZE,BIN_FBS_MEAN-BIN_FBS_STD,BIN_FBS_MEAN+BIN_FBS_STD,alpha=0.5)
    ax.plot(SIZE,BIN_ALL_MEAN,label="Binary Tree ALL")
    ax.fill_between(SIZE,BIN_ALL_MEAN-BIN_ALL_STD,BIN_ALL_MEAN+BIN_ALL_STD,alpha=0.5)
    ax.set_xlabel('Tree Size')
    ax.set_ylabel('Time (seconds)')
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)
    plt.savefig('C:\\Users\\Zack\\Downloads\\FIGURE 1 PART E.png')

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(SIZE,SV_MEM_MEAN,label="SV Memory")
    ax.fill_between(SIZE,SV_MEM_MEAN-SV_MEM_STD,SV_MEM_MEAN+SV_MEM_STD,alpha=0.5)
    ax.plot(SIZE,BIN_MEM_MEAN,label="Binary Memory")
    ax.fill_between(SIZE,BIN_MEM_MEAN-BIN_MEM_STD,BIN_MEM_MEAN+BIN_MEM_STD,alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_xlabel('Tree Size')
    ax.set_ylabel('Span of Memory Locality (Gbs)')
    ax.set_ylim(bottom=0)
    plt.savefig('C:\\Users\\Zack\\Downloads\\FIGURE 1 PART F.png')

"""
import networkx
import pydotplus

def to_networkx(graph):
    dotplus =  pydotplus.graph_from_dot_data(graph.source)
    nx_graph = networkx.nx_pydot.from_pydot(dotplus)
    return nx_graph
"""
