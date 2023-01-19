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
from time import perf_counter

def compete_add(forest,network_ids=-1,radius_buffer=0.05,exact=True):
    #Find a viable new point to add within
    if network_ids == -1:
        active_network_ids = list(range(forest.number_of_networks))
    else:
        active_network_ids = [network_ids]
    #points,_ = forest.boundary.pick(size=prefetch,homogeneous=True,replacement=False)
    #points = points.tolist()
    #active_network_ids = deepcopy(network_ids)
    active_network_tree_ids = [list(range(forest.trees_per_network[net])) for net in active_network_ids]
    #scale = len(network_ids)
    volume = deepcopy(forest.boundary.volume)
    #forest.boundary.volume = forest.boundary.volume/scale
    #networks_copy = deepcopy(forest.networks)
    #unused_networks = list(set(all_network_ids) - set(active_network_ids))
    network_pbar = tqdm(total=len(active_network_ids),desc='Adding to networks',leave=False,position=1)
    while len(active_network_ids) > 0:
        nid = active_network_ids[0]
        tree_pbar = tqdm(total=len(active_network_tree_ids[nid]),desc='Adding to Network {} Trees'.format(nid),leave=False,position=2)
        tree_pbar.refresh()
        while len(active_network_tree_ids[nid]) > 0:
            njd = active_network_tree_ids[nid][0]
            repeat = True
            while repeat:
                ## Preallocate points to check for new vessel appending
                while len(forest.networks[nid][njd].rng_points) <= 1:
                    for idx in list(range(forest.number_of_networks)):
                        for jdx in list(range(forest.trees_per_network[idx])):
                            forest.networks[idx][jdx].rng_points = []
                    rng_points,_ = forest.boundary.pick(size=min(len(forest.boundary.tet_verts),1000),homogeneous=True,replacement=False)
                    rng_points = rng_points.tolist()
                    for n in range(len(rng_points)):
                        pt = np.array(rng_points.pop(0))
                        closest_network  = None
                        closest_distance = np.inf
                        for idx in list(range(forest.number_of_networks)):
                            for jdx in list(range(forest.trees_per_network[idx])):
                                if exact:
                                    distances = close_exact_v2(forest.networks[idx][jdx].data,pt)
                                else:
                                    _,distances = close(forest.networks[idx][jdx].data,pt)
                                if min(distances) < closest_distance:
                                    closest_network = idx
                                    closest_distance = min(distances)
                        forest.networks[closest_network][np.random.choice(len(forest.networks[closest_network]))].rng_points.insert(-1,pt.tolist())
                start = perf_counter()
                for n in range(len(forest.networks[nid][njd].rng_points)):
                    pt = np.array(forest.networks[nid][njd].rng_points.pop(0))
                    closest_network  = None
                    closest_distance = np.inf
                    for idx in list(range(forest.number_of_networks)):
                        for jdx in list(range(forest.trees_per_network[idx])):
                            if exact:
                                distances = close_exact_v2(forest.networks[idx][jdx].data,pt)
                            else:
                                _,distances = close(forest.networks[idx][jdx].data,pt)
                            if min(distances) < closest_distance:
                                closest_network = idx
                                closest_distance = min(distances)
                    forest.networks[closest_network][np.random.choice(len(forest.networks[closest_network]))].rng_points.insert(-1,pt.tolist())
                #print('End point reallocation: {}'.format(perf_counter()-start))
                ##
                #print(len(forest.networks[nid][njd].rng_points))
                #print("Check: {}".format(forest.networks[nid][njd].rng_points[0]))
                start = perf_counter()
                vessel,data,sub_division_map,sub_division_index,threshold = forest.networks[nid][njd].add(-1,0,isforest=True,radius_buffer=radius_buffer)
                #print('End vessel appending: {}'.format(perf_counter()-start))
                point = data[-2,3:6] #new terminal point
                #print(point)
                #closest_network = nid
                #_,closest_value = close_exact(forest.networks[nid][njd].data[vessel,:].reshape(1,forest.networks[nid][njd].data.shape[1]),point)
                #escape = False
                """
                for idx in list(range(forest.number_of_networks)):
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
                            #print('other closer')
                            vessel_within = True
                            if not forest.convex:
                                subdivisions = 5
                                for sub in range(1,2*subdivisions):
                                    tmp = point*(sub/(2*subdivisions)) + minimum_vessel_point*(1-sub/(2*subdivisions))
                                    if not forest.boundary.within(tmp[0],tmp[1],tmp[2],2):
                                        vessel_within = False
                                        #print('not within')
                                        break
                            if not vessel_within:
                                continue
                            else:
                                escape = True
                                break
                    if escape:
                        break
                """
                #if escape:
                #    #forest.networks[nid][njd] = deepcopy(networks_copy[nid][njd])
                #    continue
                start = perf_counter()
                new_vessels = np.vstack((data[vessel,:],data[-2,:],data[-1,:]))
                repeat = False
                #check_networks = unused_networks + [nid]
                check_networks = [nid]
                for nikd in check_networks:
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
                #print('End collision testing: {}'.format(perf_counter()-start))
                if repeat:
                    #print("Repeating Network {} Tree {}".format(nid,njd))
                    #forest.networks[nid][njd] = deepcopy(networks_copy[nid][njd])
                    continue
                else:
                    forest.networks[nid][njd].data = data
                    forest.networks[nid][njd].parameters['edge_num'] += 2
                    forest.networks[nid][njd].sub_division_map = sub_division_map
                    forest.networks[nid][njd].sub_division_index = sub_division_index
                    #network_index = active_network_ids.index(nid)
                    #print("List {}: {},  Id: {}".format(nid,active_network_tree_ids[nid],njd))
                    active_network_tree_ids[nid].remove(njd)
                    tree_pbar.update(1)
                    _ = tree_pbar.refresh()
                    if len(active_network_tree_ids[nid]) == 0:
                        active_network_ids.remove(nid)
                        network_pbar.update(1)
                        _ = network_pbar.refresh()
                    #print("Active Networks: {}".format(active_network_ids))
                    #print("Completed Network {} Tree {}".format(nid,njd))
    forest.boundary.volume = volume
