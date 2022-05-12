import numpy as np
import os #remove before final
import time #not neccessary?
from .check import *
from .close import *
from .local_func_v6 import *
from ..collision.sphere_proximity import *
from ..collision.collision import *
from .add_bifurcation import *
from .sample_triad import *
from .triangle import * #might not need
from .basis import *
from scipy import interpolate
from scipy.spatial import KDTree
import matplotlib.pyplot as plt #remove before final
from .get_point import *
from mpl_toolkits.mplot3d import proj3d #remove before final
from .geodesic import extract_surface,geodesic
from ..implicit.visualize.visualize import show_mesh
from .finite_difference import finite_difference


def add_branch(data,boundary,edge_number=None,gamma=None,
               nu=None,Qterm=None,Pperm=None,Pterm=None,lam=None,k=None,
               mu=None,threshold_adjuster=0.9,
               threshold_exponent=0.5,max_skip=None,target_low=None,
               target_high=None,sub_division_map=None,sub_division_index=None,
               fraction=None,isforest=False,ishomogeneous=None,isconvex=None,
               isdirected=None,isclamped=None,radius_buffer=None):
    threshold_distance = (boundary.volume)**(1/3)/(k**threshold_exponent)
    threshold_distance += threshold_distance*0.5*np.random.random(1)
    success      = False
    if ishomogeneous:
        centers = (data[:,0:3] + data[:,3:6])/2
        while not success:
            total_attempts = 0
            attempt      = 0
            while total_attempts < 40:
                point,_ = boundary.pick(homogeneous=True)
                vessel, line_distances = close_exact(data,point)
                if sum(line_distances < threshold_distance) == 0 and min(line_distances) > 4*data[vessel[0],21]:
                    escape = False
                    for i in range(max_skip):
                        if data[vessel[i],22] <= 8*Qterm:
                            vessel = vessel[i]
                            escape = True
                            break
                    if escape:
                        break
                else:
                    if attempt < 10:
                        attempt += 1
                        total_attempts += attempt
                    else:
                        print('adjusting threshold')
                        threshold_distance *= threshold_adjuster
                        attempt = 0
            proximal = data[vessel,0:3]
            distal   = data[vessel,3:6]
            terminal = point
            num = 20
            triangle = get_local_points(data,vessel,terminal,num,isclamped)
            triangle = relative_length_constraint(triangle,proximal,distal,terminal,0.25)
            if not isconvex:
                triangle = boundary_constraint(triangle,boundary,2)
            triangle = angle_constraint(triangle,terminal,distal,-0.4,True)
            if len(triangle) == 0:
                continue
            triangle = angle_constraint(triangle,terminal,distal,0.75,False)
            if len(triangle) == 0:
                continue
            triangle = angle_constraint(triangle,terminal,proximal,0,False)
            if len(triangle) == 0:
                continue
            triangle = angle_constraint(triangle,distal,proximal,0,False)
            if len(triangle) == 0:
                continue
            if data[vessel,17] >= 0:
                p_vessel = int(data[vessel,17])
                daughter1_vector = -data[p_vessel,12:15]
                daughter2_vector = (triangle - proximal)/np.linalg.norm(triangle - proximal,axis=1).reshape(-1,1)
                angle = np.array([np.dot(daughter1_vector,daughter2_vector[i]) for i in range(len(daughter2_vector))])
                triangle = triangle[angle<0]
                if len(triangle) == 0:
                    continue
            #start = time.time()
            results  = fast_local_function(data,triangle,terminal,vessel,gamma,nu,Qterm,Pperm,Pterm)
            #b_time = time.time() - start
            volume       = np.pi*(results[0]**lam)*(results[1]**mu)
            idx          = np.argmin(volume)
            bif          = results[5][idx]
            #start = time.time()
            #fd_bif,fd_idx,fd_volume,fd_trial = finite_difference(data,triangle,terminal,vessel,gamma,nu,Qterm,Pperm,Pterm)
            #fd_time = time.time() - start
            no_collision = collision_free(data,results,idx,terminal,vessel,radius_buffer)
            #no_collision_fd = collision_free_fd(fd_trial,vessel)
            #print(no_collision)
            if no_collision:
                #print('LOCAL ACCEL: {}    FINITE DIFF: {}'.format(bif,fd_bif))
                #print(np.allclose(bif,fd_bif))
                #print('adding')
                data = np.vstack((data,np.zeros((2,data.shape[1]))))
                data,sub_division_index = add_bifurcation(data,None,bif,terminal,vessel,Qterm,
                                                          results[0],results[1],results[2],results[3],
                                                          np.array(results[6]),np.array(results[7]),
                                                          np.array(results[8]),np.array(results[9]),
                                                          np.array(results[10]),np.array(results[11]),
                                                          np.array(results[12]),np.array(results[13]),
                                                          results[14],results[15],idx,sub_division_map,
                                                          sub_division_index,True)
                #print('FINITE: {}    BINDING: {}'.format(fd_time,b_time))
                #data = fd_trial
                k += 2
                results = [data,k,sub_division_map,sub_division_index]
                if not isforest:
                     return data,k,sub_division_map,sub_division_index
                else:
                     new_terminal = data[-2,:]
                     new_sister   = data[-1,:]
                     new_parent   = data[vessel,:]
                     return new_terminal,new_sister,new_parent,results
            else:
                continue
                #return data,k,sub_division_map,sub_division_index
    pure_data    = data[data[:,-1]>-1]
    segment_data = data[data[:,-1]==-1]
    vessel       = np.random.choice(list(range(pure_data.shape[0])))
    vessel_path  = segment_data[segment_data[:,29].astype(int)==vessel]
    other_vessels= segment_data[segment_data[:,29].astype(int)!=vessel]
    if pure_data.shape[0] > 1:
        other_KDTree = KDTree((other_vessels[:,0:3]+other_vessels[:,3:6])/2)
    else:
        other_KDTree = None
    mesh,pa,cp,cd= boundary.mesh(vessel_path[1:,0:3],threshold_distance,threshold_distance//fraction,dive=0,others=other_KDTree)
    #show_mesh(mesh,np.array(pa),np.array(cp))
    mesh_poly    = extract_surface(mesh)
    proximal     = vessel_path[0,0:3]
    distal       = vessel_path[-1,3:6]
    terminal     = cp[-1]
    #results      = fast_local_function(pure_data,mesh,terminal,vessel,gamma,nu,Qterm,Pperm,Pterm)
    d_max        = max(data[vessel,20]*0.1,np.sum(cd)*0.1)
    mesh_truc    = mesh[np.linalg.norm(mesh-proximal,axis=1)>d_max]
    mesh_truc    = mesh_truc[np.linalg.norm(mesh_truc-distal,axis=1)>d_max]
    #show_mesh(mesh_truc,np.array(pa),np.array(cp))
    mesh_truc    = mesh_truc[np.linalg.norm(mesh_truc-terminal,axis=1)>d_max]
    #show_mesh(mesh_truc,np.array(pa),np.array(cp))
    results      = fast_local_function(pure_data,mesh_truc,terminal,vessel,gamma,nu,Qterm,Pperm,Pterm)
    volume       = np.pi*(results[0]**lam)*(results[1]**mu)
    idx          = np.argmin(volume)
    bif          = results[5][idx]
    proximal,distal,terminal = geodesic(mesh_poly,proximal,distal,terminal,bif)
    new_parent_size   = proximal['path'].shape[0]-1
    new_sister_size   = distal['path'].shape[0]-1
    new_terminal_size = terminal['path'].shape[0]-1
    ##########
    # Ensure continuity of vessels reapply start and end nodes
    ##########
    #terminal['path'][0,:] = cp[-1]
    #distal['path'][0,:]   = vessel_path[-1,3:6]
    proximal['path'][-1,:]  = vessel_path[0,0:3]
    new_parent        = np.zeros((new_parent_size,data.shape[1]))
    new_sister        = np.zeros((new_sister_size,data.shape[1]))
    new_terminal      = np.zeros((new_terminal_size,data.shape[1]))
    new_segment_data  = other_vessels
    new_terminal[:,0:3] = terminal['path'][:-1,:]
    new_terminal[:,3:6] = terminal['path'][1:,:]
    new_terminal[:,20]  = np.linalg.norm(new_terminal[:,0:3]-new_terminal[:,3:6])
    new_terminal[:,21]  = results[0][idx]*results[2][idx]
    new_terminal[:,29]  = np.float(pure_data.shape[0])
    new_terminal[:,-1]  = -1
    new_sister[:,0:3]   = distal['path'][:-1,:]
    new_sister[:,3:6]   = distal['path'][1:,:]
    new_sister[:,20]    = np.linalg.norm(new_sister[:,0:3] - new_sister[:,3:6])
    new_sister[:,21]    = results[0][idx]*results[3][idx]
    new_sister[:,29]    = np.float(pure_data.shape[0])+1
    new_sister[:,-1]    = -1
    new_parent[:,0:3]   = proximal['path'][:-1,:]
    new_parent[:,3:6]   = proximal['path'][1:,:]
    new_parent[:,20]    = np.linalg.norm(new_parent[:,0:3] - new_parent[:,3:6])
    new_parent[:,29]    = np.float(vessel)
    new_parent[:,-1]    = -1
    if vessel != 0:
        new_parent[:,21] = results[0][idx]*results[4][idx]
    else:
        new_parent[:,21] = results[0][idx]
    for i in range(max([new_terminal.shape[0],new_sister.shape[0],new_parent.shape[0]])):
        if i < new_terminal.shape[0]:
            basis(new_terminal,i)
        if i < new_sister.shape[0]:
            basis(new_sister,i)
        if i < new_parent.shape[0]:
            basis(new_parent,i)
    new_segments = np.vstack((new_segment_data,new_terminal,new_sister,new_parent))
    pure_data = np.vstack((pure_data,np.zeros((2,data.shape[1]))))
    data,sub_division_index = add_bifurcation(pure_data,new_segments,bif,cp[-1],vessel,Qterm,
                                              results[0],results[1],results[2],results[3],
                                              np.array(results[6]),np.array(results[7]),
                                              np.array(results[8]),np.array(results[9]),
                                              np.array(results[10]),np.array(results[11]),
                                              np.array(results[12]),np.array(results[13]),
                                              results[14],results[15],idx,sub_division_map,
                                              sub_division_index)
    k += 2
    if not isforest:
        return data,k,sub_division_map,sub_division_index
    else:
        results = [data,k,sub_division_map,sub_division_index]
        return new_terminal,new_sister,new_parent,results
