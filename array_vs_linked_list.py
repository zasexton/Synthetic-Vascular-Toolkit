"""
Code to evaluate cache hits and misses for svcco
"""
import svcco
import numpy as np
from binarytree import Node
from time import time
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
    node.data           = data.reshape(1,-1)
    node.proximal_point = data[0:3]
    node.distal_point   = data[3:6]
    node.radius         = data[21]
    node.length         = data[20]
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
                        vessel.left = new_vessel
                        self.all_nodes.append(new_vessel)
                        self.all_data.append(np.array(deepcopy(new_vessel.data[0,:].tolist())))
                        self.all_data_list.append(new_vessel.data[0,:].tolist())
                        break
                    if vessel.daughter_1 == new_vessel.id:
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

def test_sv_tree(sv_tree,point):
    start = time()
    best_id,best_dist = svcco.branch_addition.close.close_exact(sv_tree.data,point)
    elapsed = time() - start
    best_id = best_id[0]
    return elapsed,best_id

def test_sv_tree_calls(t,point):
    cProfile.run("svcco.branch_addition.close.close_exact(t.data,point)","sv_stats")
    p = pstats.Stats("sv_stats")
    return p.total_calls

def test_sv_tree_mem(t,point):
    mem = memory_usage((svcco.branch_addition.close.close_exact,(t.data,point)))
    return np.mean(mem)

def test_binary_tree(binary,point,meth='fbs'):
    start = time()
    best_id,best_dist = binary.find_closest(point,method=meth)
    elapsed = time() - start
    return elapsed,best_id

def test_binary_calls(binary,point,meth='fbs'):
    cProfile.run("binary.find_closest(point,method='{}')".format(meth),"bin_stats")
    p = pstats.Stats("bin_stats")
    return p.total_calls

def test_binary_mem(binary,point,meth='fbs'):
    mem = memory_usage((binary.find_closest,(point,),{'method':meth}))
    return np.mean(mem)
#####################################################
# CODE FOR TESTING PERFORMANCE
#####################################################

binary = build(t.data)
point = np.array([0.2,0.2,0.2])

def test(t,binary,point,reps=100):
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

def test_range(t,binary,point,start=1,stop=1000,incr=10):
    #t = svcco.tree()
    #t.set_boundary(s)
    #t.set_root()
    #t.convex = True
    binary = build(t.data)
    SIZE     = [start]
    SV       = []
    BIN_FBS  = []
    BIN_ALL  = []
    BIN_VEC  = []
    SV_CALLS = []
    BIN_CALLS= []
    SV_MEM   = []
    BIN_MEM  = []
    while SIZE[-1] < stop:
        sv_perf,bin_perf_fbs,bin_perf_all,bin_perf_vec = test(t,binary,point)
        SV.append(sv_perf)
        BIN_FBS.append(bin_perf_fbs)
        BIN_ALL.append(bin_perf_all)
        BIN_VEC.append(bin_perf_vec)
        SV_CALLS.append(sv_calls)
        BIN_CALLS.append(bin_calls)
        SV_MEM.append(sv_mem)
        BIN_MEM.append(bin_mem)
        t.n_add(incr)
        binary = build(t.data)
        SIZE.append(SIZE[-1]+incr)
    SIZE.pop(-1)
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




"""
import networkx
import pydotplus

def to_networkx(graph):
    dotplus =  pydotplus.graph_from_dot_data(graph.source)
    nx_graph = networkx.nx_pydot.from_pydot(dotplus)
    return nx_graph
"""
