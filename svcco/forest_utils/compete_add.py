# Competitive growth of Multiple Vascular Networks within a Shared Domain
import numpy as np
import os #remove before final
import time #not neccessary?
from ..branch_addition.check import *
from ..branch_addition.close import *
from ..branch_addition.local_func_v7 import *
from ..collision.sphere_proximity import *
from ..collision.collision import *
from ..branch_addition.add_bifurcation import *
from ..branch_addition.sample_triad import *
from ..branch_addition.triangle import * #might not need
from ..branch_addition.basis import *
#from scipy import interpolate
#from scipy.spatial import KDTree
from ..branch_addition.get_point import *
from copy import deepcopy
#from .geodesic import extract_surface,geodesic
#from scipy.sparse.csgraph import shortest_path
#from .add_geodesic_path import *
import pyvista as pv
import random
from tqdm import tqdm

def compete_add(forest,network_ids=-1,radius_buffer=0.05,exact=True):
    #Find a viable new point to add within
    all_network_ids = list(range(forest.number_of_networks))
    if network_ids == -1:
        network_ids = all_network_ids
    #points,_ = forest.boundary.pick(size=prefetch,homogeneous=True,replacement=False)
    #points = points.tolist()
    active_network_ids = deepcopy(network_ids)
    active_network_tree_ids = [list(range(forest.trees_per_network[net])) for net in all_network_ids]
    scale = len(network_ids)
    volume = deepcopy(forest.boundary.volume)
    forest.boundary.volume = forest.boundary.volume/scale
    #networks_copy = deepcopy(forest.networks)
    unused_networks = list(set(all_network_ids) - set(active_network_ids))
    while len(active_network_ids) > 0:
        for nid in active_network_ids:
            for njd in active_network_tree_ids[nid]:
                ## New
                if  forest.networks[nid][njd].data.shape[0] < 100:
                    if len(forest.networks[nid][njd].rng_points) <= 1:
                        forest.networks[nid][njd].rng_points,_ = forest.boundary.pick(size=min(len(forest.boundary.tet_verts),10000),homogeneous=True,replacement=False)
                        forest.networks[nid][njd].rng_points = forest.networks[nid][njd].rng_points.tolist()
                    for n in range(len(forest.networks[nid][njd].rng_points)):
                        pt = np.array(forest.networks[nid][njd].rng_points.pop(0))
                        if exact:
                            other_vessels,distances = close_exact(forest.networks[nid][njd].data,pt)
                        else:
                            other_vessels,distances = close(forest.networks[nid][njd].data,pt)
                        minimum_distance = min(distances)
                        retry = False
                        for idx in network_ids:
                            for jdx in list(range(forest.trees_per_network[idx])):
                                if idx == nid:
                                    continue
                                if forest.networks[nid][njd].data.shape[0] < 100:
                                    other_vessels,distances = close_exact(forest.networks[idx][jdx].data,pt)
                                else:
                                    other_vessels,distances = close(forest.networks[idx][jdx].data,pt)
                                if min(distances) < minimum_distance:
                                    retry = True
                                    break
                            if retry:
                                break
                        if retry:
                            continue
                        else:
                            forest.networks[nid][njd].rng_points.insert(-1,pt.tolist())
                ##
                #print(len(forest.networks[nid][njd].rng_points))
                #print("Check: {}".format(forest.networks[nid][njd].rng_points[0]))
                vessel,data,sub_division_map,sub_division_index,threshold = forest.networks[nid][njd].add(-1,0,isforest=True,radius_buffer=radius_buffer)
                point = data[-2,3:6] #new terminal point
                #print(point)
                closest_network = nid
                _,closest_value = close_exact(forest.networks[nid][njd].data[vessel,:].reshape(1,forest.networks[nid][njd].data.shape[1]),point)
                escape = False
                for idx in network_ids:
                    for jdx in list(range(forest.trees_per_network[idx])):
                        if idx == nid:
                            continue
                        if exact:
                            other_vessels,distances = close_exact(forest.networks[idx][jdx].data,point)
                        else:
                            other_vessels,distances = close(forest.networks[idx][jdx].data,point)
                        minimum_distance = min(distances)
                        minimum_vessel_point = (forest.networks[idx][jdx].data[other_vessels[0],0:3] + forest.networks[idx][jdx].data[other_vessels[0],3:6])/2
                        if minimum_distance < closest_value:
                            vessel_within = True
                            if not forest.convex:
                                subdivisions = 5
                                for sub in range(1,2*subdivisions):
                                    tmp = point*(sub/(2*subdivisions)) + minimum_vessel_point*(1-sub/(2*subdivisions))
                                    if not forest.boundary.within(tmp[0],tmp[1],tmp[2],2):
                                        vessel_within = False
                                        break
                            if not vessel_within:
                                continue
                            else:
                                escape = True
                                break
                    if escape:
                        break
                if escape:
                    #forest.networks[nid][njd] = deepcopy(networks_copy[nid][njd])
                    continue
                new_vessels = np.vstack((data[vessel,:],data[-2,:],data[-1,:]))
                repeat = False
                check_networks = unused_networks + [nid]
                for nikd in range(len(unused_networks)):
                    if nikd == nid:
                        check_trees = list(range(len(forest.networks[nikd])))
                        check_trees.remove(njd)
                    else:
                        check_trees = list(range(len(forest.networks[nikd])))
                    for njkd in check_trees:
                        if not forest.networks[nikd][njkd].collision_free(new_vessels,radius_buffer=radius_buffer):
                            repeat = True
                            break
                    if repeat:
                        break
                if repeat:
                    #forest.networks[nid][njd] = deepcopy(networks_copy[nid][njd])
                    continue
                else:
                    forest.networks[nid][njd].data = data
                    forest.networks[nid][njd].parameters['edge_num'] += 2
                    forest.networks[nid][njd].sub_division_map = sub_division_map
                    forest.networks[nid][njd].sub_division_index = sub_division_index
                    #print("List {}: {},  Id: {}".format(nid,active_network_tree_ids[nid],njd))
                    active_network_tree_ids[nid].remove(njd)
                    if len(active_network_tree_ids[nid]) == 0:
                        active_network_ids.remove(nid)
                    #print("Active Networks: {}".format(active_network_ids))
    forest.boundary.volume = volume
