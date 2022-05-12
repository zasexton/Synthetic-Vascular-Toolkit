import numpy as np
import os #remove before final
import time #not neccessary?
from .check import *
from .close import *
from .local_func_v5 import *
from ..collision.sphere_proximity import *
from ..collision.collision import *
from .add_bifurcation import *
from .sample_triad import *
from .triangle import *
from .basis import *
from scipy import interpolate
from scipy.spatial import KDTree
import matplotlib.pyplot as plt #remove before final
from .get_point import *
from mpl_toolkits.mplot3d import proj3d #remove before final
from .geodesic import extract_surface,geodesic

def add_branch(data,boundary_within=None,edge_number=None,gamma=None,
               nu=None,Qterm=None,Pperm=None,Pterm=None,lam=None,k=None,
               mu=None,boundary_threshold=None,threshold_adjuster=0.75,
               threshold_exponent=1.5,max_skip=None,target_low=None,
               target_high=None,sub_division_map=None,sub_division_index=None,
               level_func=None,func=None,func_grad=None,active_patches=None,
               patch_tree=None,patch_centers=None,fraction=None,isforest=False):
    threshold_original = boundary_threshold(k**threshold_exponent)
    threshold = np.random.uniform(threshold_original,2*threshold_original)
    success = False
    attempts = 0
    #k = 5
    pure_data = data[data[:,-1]>-1]
    segment_data = data[data[:,-1]==-1]
    int_pt_opt = []
    lengths = []
    total_length = np.sum(pure_data[:,20])
    success = False
    prob = pure_data[:,20]/total_length
    total_proximity = KDTree(segment_data[:,0:3])
    while not success:
        vessel_start = np.random.choice(list(range(pure_data.shape[0])),p=prob)
        vessel_idxs = np.argwhere(segment_data[:,29].astype(int)==vessel_start).flatten()
        other_idx = np.argwhere(segment_data[:,29].astype(int)!=vessel_start).flatten()
        proximity = KDTree(segment_data[other_idx,0:3])
        if len(vessel_idxs) > 2:
            segment_start = round(len(vessel_idxs)/2) #np.random.choice(vessel_idxs[1:-1])
            p0 = segment_data[segment_start,0:3]
        elif len(vessel_idxs) == 2:
            segment_start = vessel_idxs[0]
            p0 = segment_data[segment_start,3:6]
        elif len(vessel_idxs) == 1:
            segment_start = vessel_idxs[0]
            p0 = (segment_data[segment_start,0:3] + segment_data[segment_start,3:6])/2
        p0_norm = func_grad(*p0)
        p0_target = func(*p0)
        theta = np.linspace(0,2*np.pi,num=20,endpoint=False).reshape(-1,1)
        t1,t2,n = tangent_basis(p0_norm,p0)
        pt = p0 + t1*fraction
        int_pt_opt.append(level_func(*pt.flatten(),p0_target).x)
        dist = np.linalg.norm(int_pt_opt[-1] - p0)
        lengths.append(dist)
        print('Finding Point')
        dists = [dist]
        #fig = plt.figure()
        #ax = fig.add_subplot(111,projection='3d')
        #ax.scatter3D(segment_data[vessel_idxs[0],0],segment_data[vessel_idxs[0],1],segment_data[vessel_idxs[0],2],c='g')
        #ax.scatter3D(segment_data[vessel_idxs[-1],0],segment_data[vessel_idxs[-1],1],segment_data[vessel_idxs[-1],2],c='g')
        #ax.scatter3D(segment_data[vessel_idxs[1:-1],0],segment_data[vessel_idxs[1:-1],1],segment_data[vessel_idxs[1:-1],2],c='r')
        #plt.savefig(os.getcwd()+'\\Data\\figure_0')
        while dist < threshold:
            p1_norm = func_grad(*int_pt_opt[-1])
            t1,t2,n = tangent_basis(p1_norm,int_pt_opt[-1])
            pot_pts = int_pt_opt[-1] + t1*np.sin(theta)*fraction + t2*np.cos(theta)*fraction
            pot_rest_of_tree,_ = proximity.query(pot_pts)
            pot_main_branch = np.linalg.norm(pot_pts - p0,axis=1)
            #pot_dist = pot_rest_of_tree + pot_main_branch
            pot_dist,_ = total_proximity.query(pot_pts)
            #pot_dist = pot_main_branch
            pt_idx = np.argmax(pot_dist).flatten()[0]
            int_pt_opt.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
            #ax.scatter3D(int_pt_opt[-1][0],int_pt_opt[-1][1],int_pt_opt[-1][2],c='b')
            #plt.savefig(os.getcwd()+'\\Data\\figure_{}'.format(len(lengths)))
            #dist = min(pot_rest_of_tree[pt_idx],pot_main_branch[pt_idx])#min(pot_main_branch[pt_idx],pot_rest_of_tree[pt_idx]) #np.linalg.norm(int_pt_opt[-1] - p0)
            dist = pot_dist[pt_idx]
            dists.append(dist)
            lengths.append(np.linalg.norm(int_pt_opt[-1]-int_pt_opt[-2]))
            if len(lengths) > 400 and threshold > threshold_original*0.9:
                print('Reducing threshold')
                threshold = threshold*0.9
            elif len(lengths) >= 405:
                break
        #dists = np.array(dists)/threshold
        #timestamp = time.gmtime()
        #np.savetxt(os.getcwd()+'\\Data\\dist_data_{}{}{}{}{}{}'.format(timestamp[0],timestamp[1],timestamp[2],timestamp[3],timestamp[4],timestamp[5]),dists,delimiter=',')
        if dist < threshold and len(lengths) >= 400:
            success = False
            print('Failed: switching vessel...')
            adjust = prob[vessel_start]
            prob[vessel_start] = 0
            non_zeros = np.argwhere(prob!=0).flatten()
            if len(non_zeros) == 0:
                print('returning')
                return data,k,sub_division_map,sub_division_index
            adjust_portion = adjust/len(non_zeros)
            prob[non_zeros] = prob[non_zeros] + adjust_portion
            continue
        else:
            success = True
            break
    term = int_pt_opt[-1]
    dist, idx = total_proximity.query(term)
    vessel_start = int(segment_data[idx,29])
    vessel_idxs = np.argwhere(segment_data[:,29].astype(int)==vessel_start).flatten()
    other_idx = np.argwhere(segment_data[:,29].astype(int)!=vessel_start).flatten()
    if len(vessel_idxs) > 2:
        segment_start = round(len(vessel_idxs)/2) #np.random.choice(vessel_idxs[1:-1])
        p0 = segment_data[segment_start,0:3]
    elif len(vessel_idxs) == 2:
        segment_start = vessel_idxs[0]
        p0 = segment_data[segment_start,3:6]
    elif len(vessel_idxs) == 1:
        segment_start = vessel_idxs[0]
        p0 = (segment_data[segment_start,0:3] + segment_data[segment_start,3:6])/2
    #p0 = pure_data[vessel_start,0:3]
    """
    p0_norm = func_grad(*p0)
    p0_target = func(*p0)
    int_pt_opt = []
    t1,t2,n = tangent_basis(p0_norm,p0)
    pot_pts = p0 + t1*fraction*np.sin(theta) + t2*fraction*np.cos(theta)
    pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
    int_pt_opt.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
    dist = np.linalg.norm(int_pt_opt[-1] - p1)
    lengths.append(dist)
    print('Building Middle')
    while dist > fraction:
        p2_norm = func_grad(*int_pt_opt[-1])
        t1,t2,n = tangent_basis(p2_norm,int_pt_opt[-1])
        pot_pts = int_pt_opt[-1] + t1*np.sin(theta)*fraction + t2*np.cos(theta)*fraction
        pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
        if dist - np.linalg.norm(pot_pts[pt_idx] - p1) < 0.5*fraction:
            direct_point = int_pt_opt[-1] + ((p1 - int_pt_opt[-1])/np.linalg.norm(p1 - int_pt_opt[-1]))*fraction
            int_pt_opt.append(direct_point)
        else:
            int_pt_opt.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
        dist = np.linalg.norm(int_pt_opt[-1] - p1)
        lengths.append(np.linalg.norm(int_pt_opt[-1]-int_pt_opt[-2]))
        print(dist)
    int_pt_opt.append(p1)

    p0 = pure_data[vessel_start,0:3]
    p0_norm = func_grad(*p0)
    p0_target = func(*p0)
    int_pt_side1 = []
    t1,t2,n = tangent_basis(p0_norm,p0)
    pot_pts = p0 + t1*fraction*np.sin(theta) + t2*fraction*np.cos(theta)
    pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
    int_pt_side1.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
    dist = np.linalg.norm(int_pt_side1[-1] - p1)
    lengths.append(dist)
    print('Building Side 1')
    while dist > fraction:
        p2_norm = func_grad(*int_pt_side1[-1])
        t1,t2,n = tangent_basis(p2_norm,int_pt_side1[-1])
        pot_pts = int_pt_side1[-1] + t1*np.sin(theta)*fraction + t2*np.cos(theta)*fraction
        pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
        if dist - np.linalg.norm(pot_pts[pt_idx] - p1) < 0.5*fraction:
            direct_point = int_pt_side1[-1] + ((p1 - int_pt_side1[-1])/np.linalg.norm(p1 - int_pt_side1[-1]))*fraction
            int_pt_side1.append(direct_point)
        else:
            int_pt_side1.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
        dist = np.linalg.norm(int_pt_side1[-1] - p1)
        lengths.append(np.linalg.norm(int_pt_side1[-1]-int_pt_side1[-2]))
        print(dist)
    int_pt_side1.append(p1)
    #p1 = int_pt_opt[-1]
    p0 = pure_data[vessel_start,3:6]
    p0_norm = func_grad(*p0)
    p0_target = func(*p0)
    int_pt_side2 = []
    t1,t2,n = tangent_basis(p0_norm,p0)
    pot_pts = p0 + t1*fraction*np.sin(theta) + t2*fraction*np.cos(theta)
    pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
    int_pt_side2.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
    dist = np.linalg.norm(int_pt_side2[-1] - p1)
    lengths.append(dist)
    print('Building Side 2')
    while dist > fraction:
        p2_norm = func_grad(*int_pt_side2[-1])
        t1,t2,n = tangent_basis(p2_norm,int_pt_side2[-1])
        pot_pts = int_pt_side2[-1] + t1*np.sin(theta)*fraction + t2*np.cos(theta)*fraction
        pt_idx = np.argmin(np.linalg.norm(pot_pts - p1,axis=1)).flatten()[0]
        if dist - np.linalg.norm(pot_pts[pt_idx] - p1) < 0.5*fraction:
            direct_point = int_pt_side2[-1] + ((p1 - int_pt_side2[-1])/np.linalg.norm(p1 - int_pt_side2[-1]))*fraction
            int_pt_side2.append(direct_point)
        else:
            int_pt_side2.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
        #int_pt_side2.append(level_func(*pot_pts[pt_idx].flatten(),p0_target).x)
        dist = np.linalg.norm(int_pt_side2[-1] - p1)
        lengths.append(np.linalg.norm(int_pt_side2[-1]-int_pt_side2[-2]))
        print(dist)
    int_pt_side2.append(p1)
    """
    middle = p0
    proximal = pure_data[vessel_start,0:3]
    distal = pure_data[vessel_start,3:6]
    int_pt_opt = to_point(middle,term,func_grad,fraction)
    int_pt_side1 = to_point(proximal,term,func_grad,fraction)
    int_pt_side2 = to_point(distal,term,func_grad,fraction)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    int_pt_opt = np.array(int_pt_opt)
    int_pt_side1 = np.array(int_pt_side1)
    int_pt_side2 = np.array(int_pt_side2)
    ax.scatter3D(segment_data[1:-1,0],segment_data[1:-1,1],segment_data[1:-1,2],c='r')
    ax.scatter3D(int_pt_opt[:,0],int_pt_opt[:,1],int_pt_opt[:,2],c='g')
    ax.scatter3D(int_pt_side1[:,0],int_pt_side1[:,1],int_pt_side1[:,2],c='b')
    ax.scatter3D(int_pt_side2[:,0],int_pt_side2[:,1],int_pt_side2[:,2],c='b')
    ax.scatter3D(segment_data[0,0],segment_data[0,1],segment_data[0,2],c='r',marker='s')
    ax.scatter3D(segment_data[0,0],segment_data[0,1],segment_data[0,2],c='r',marker='d')
    #plt.show()
    int_pt_side0 = np.vstack((segment_data[vessel_idxs,0:3],segment_data[vessel_idxs[-1],3:6]))
    middle,u = interpolate.splprep([int_pt_opt[:,0],int_pt_opt[:,1],int_pt_opt[:,2]],k=min(int_pt_opt.shape[0]-1,3))
    side0,u = interpolate.splprep([int_pt_side0[:,0],int_pt_side0[:,1],int_pt_side0[:,2]],k=min(int_pt_opt.shape[0]-1,3))
    side1,u = interpolate.splprep([int_pt_side1[:,0],int_pt_side1[:,1],int_pt_side1[:,2]],k=min(int_pt_opt.shape[0]-1,3))
    side2,u = interpolate.splprep([int_pt_side2[:,0],int_pt_side2[:,1],int_pt_side2[:,2]],k=min(int_pt_opt.shape[0]-1,3))
    pt0 = segment_data[vessel_idxs[0],0:3]
    pt0113 = interpolate.splev(1/3,side0)
    pt0213 = interpolate.splev(1/3,side1)
    pt0123 = interpolate.splev(2/3,side0)
    pt012 = interpolate.splev(0.5,middle)
    pt0223 = interpolate.splev(2/3,side1)
    pt1 = segment_data[vessel_idxs[-1],3:6]
    pt1213 = interpolate.splev(1/3,side2)
    pt1223 = interpolate.splev(2/3,side2)
    pt2 = term
    nodes = np.array([pt0,pt0113,pt0213,pt0123,pt012,pt0223,pt1,pt1213,pt1223,pt2])
    tri = triangle(level_func,p0_target,pure_data[vessel_start,0:3],pure_data[vessel_start,3:6],
                   term,p0_target,p0_target,degree=3,nodes=nodes,target=p0_target)
    precomputed_pts,precomputed_targets,params = tri.generate(n=20)
    ax.scatter3D(precomputed_pts[:,0],precomputed_pts[:,1],precomputed_pts[:,2],c='y',alpha=0.5)
    n = np.linspace(0,1,num=50)
    side0_splinex,side0_spliney,side0_splinez = interpolate.splev(n,side0)
    side1_splinex,side1_spliney,side1_splinez = interpolate.splev(n,side1)
    side2_splinex,side2_spliney,side2_splinez = interpolate.splev(n,side2)
    middle_splinex,middle_spliney,middle_splinez = np.array(interpolate.splev(n,middle))
    ax.scatter3D(side0_splinex,side0_spliney,side0_splinez,c='r',marker='+')
    ax.scatter3D(side1_splinex,side1_spliney,side1_splinez,c='b',marker='+')
    ax.scatter3D(side2_splinex,side2_spliney,side2_splinez,c='b',marker='+')
    ax.scatter3D(middle_splinex,middle_spliney,middle_splinez,c='g',marker='+')
    d1 = np.sqrt(np.sum(np.square(pt0-pt1)))*0.25
    d2 = np.sqrt(np.sum(np.square(pt1-pt2)))*0.25
    d3 = np.sqrt(np.sum(np.square(pt2-pt0)))*0.25
    r = pure_data[vessel_start,21]*4
    d_max = max([d1,d2,d3,r])
    params = list(params)
    sam = sample_area(segment_data[vessel_idxs,0:3],term,func_grad,fraction)
    cull = np.linalg.norm(precomputed_pts-pt0,axis=1)>d_max
    cull_sam = np.linalg.norm(sam-pt0,axis=1)>d_max
    sam = sam[cull_sam,:]
    precomputed_pts = precomputed_pts[cull,:]
    precomputed_targets = precomputed_targets[cull,:]
    params[0] = params[0][cull]
    params[1] = params[1][cull]
    params[2] = params[2][cull]
    cull = np.linalg.norm(precomputed_pts-pt1,axis=1)>d_max
    cull_sam = np.linalg.norm(sam-pt1,axis=1)>d_max
    sam = sam[cull_sam,:]
    precomputed_pts = precomputed_pts[cull,:]
    precomputed_targets = precomputed_targets[cull,:]
    params[0] = params[0][cull]
    params[1] = params[1][cull]
    params[2] = params[2][cull]        
    cull = np.linalg.norm(precomputed_pts-pt2,axis=1)>d_max
    cull_sam = np.linalg.norm(sam-pt2,axis=1)>d_max
    sam = sam[cull_sam,:]
    precomputed_pts = precomputed_pts[cull,:]
    precomputed_targets = precomputed_targets[cull,:]
    params[0] = params[0][cull]
    params[1] = params[1][cull]
    params[2] = params[2][cull]
    ax.scatter3D(precomputed_pts[:,0],precomputed_pts[:,1],precomputed_pts[:,2],c='y',marker='+')
    local_opt = fast_local_function(a,b,pure_data,pt2.flatten(),vessel_start,gamma,nu,Qterm,Pperm,Pterm,precomputed=sam)
    test = np.zeros(data.shape[1])
    tree_volume = np.pi*(local_opt[0]**lam)*(local_opt[1]**mu)
    idx = np.argmin(tree_volume)
    s = params[0][idx]
    t = params[1][idx]
    u = params[2][idx]
    xyz = local_opt[5][idx]
    xyz_opt = xyz#np.array(level_func(*xyz,p0_target).x)
    #sam = sample_area(segment_data[vessel_idxs,0:3],term,func_grad,fraction)
    #fig = plt.figure()
    #ax = plt.add_subplot(111,projection='3d')
    ax.scatter3D(sam[:,0],sam[:,1],sam[:,2],c='r')
    print('Found local optimum')
    print('Building Vessels')
    # VESSELS GENERATED ON BEZIER PATCHES
    parent_coordinates = to_point(proximal,xyz_opt,func_grad,fraction)
    sister_coordinates = to_point(distal,xyz_opt,func_grad,fraction)
    terminal_coordinates = to_point(xyz_opt,term,func_grad,fraction)
    new_parent = np.zeros((parent_coordinates.shape[0]-1,data.shape[1]))
    new_sister = np.zeros((sister_coordinates.shape[0]-1,data.shape[1]))
    new_terminal = np.zeros((terminal_coordinates.shape[0]-1,data.shape[1]))
    terminal_segments = new_terminal.shape[0]
    sister_segments = new_sister.shape[0]
    parent_segments = new_parent.shape[0]
    """
    terminal_segments = max(round(np.sqrt(np.sum(np.square(pt2.flatten()-xyz)))/fraction),1)
    parent_segments = max(round(np.sqrt(np.sum(np.square(pt0-xyz)))/fraction),1)
    sister_segments = max(round(np.sqrt(np.sum(np.square(pt1-xyz)))/fraction),1)
    new_terminal = np.zeros((terminal_segments,data.shape[1]))
    new_sister = np.zeros((sister_segments,data.shape[1]))
    new_parent = np.zeros((parent_segments,data.shape[1]))
    terminal_s = np.linspace(s,0,num=terminal_segments+1)
    terminal_t = np.linspace(t,0,num=terminal_segments+1)
    terminal_u = np.linspace(u,1,num=terminal_segments+1)
    terminal_coordinates,terminal_targets = tri.cubic(terminal_s,terminal_t,terminal_u)
    sister_s = np.linspace(s,1,num=sister_segments+1)
    sister_t = np.linspace(t,0,num=sister_segments+1)
    sister_u = np.linspace(u,0,num=sister_segments+1)
    sister_coordinates,sister_targets = tri.cubic(sister_s,sister_t,sister_u)
    parent_s = np.linspace(s,0,num=parent_segments+1)
    parent_t = np.linspace(t,1,num=parent_segments+1)
    parent_u = np.linspace(u,0,num=parent_segments+1)
    parent_coordinates,parent_targets = tri.cubic(parent_s,parent_t,parent_u)
    """
    #term_norm = np.zeros(terminal_coordinates.shape)
    #for coor in range(terminal_coordinates.shape[0]):
        #terminal_coordinates[coor,:] = np.array(level_func(*terminal_coordinates[coor,:].flatten(),p0_target).x)
        #term_norm = func_grad(*terminal_coordinates[coor,:])
        #ax.plot3D([terminal_coordinates[coor,0],terminal_coordinates[coor,0]+term_norm[0]],
        #          [terminal_coordinates[coor,1],terminal_coordinates[coor,1]+term_norm[1]],
        #          [terminal_coordinates[coor,2],terminal_coordinates[coor,2]+term_norm[2]])
    ax.scatter3D(terminal_coordinates[:,0],terminal_coordinates[:,1],terminal_coordinates[:,2],c='black')
    ax.scatter3D(sister_coordinates[:,0],sister_coordinates[:,1],sister_coordinates[:,2],c='black')
    ax.scatter3D(parent_coordinates[:,0],parent_coordinates[:,1],parent_coordinates[:,2],c='black')
    # ADD VESSELS INTO DATA STRUCTURE
    # remove old parent vessel transient segments
    new_transient = segment_data[other_idx,:]
    new_terminal[:,0:3] = terminal_coordinates[:-1,:]
    new_terminal[:,3:6] = terminal_coordinates[1:,:]
    new_terminal[:,20] = np.linalg.norm(terminal_coordinates[1:,:]-terminal_coordinates[:-1,:],axis=1)
    new_terminal[:,21] = local_opt[0][idx]*local_opt[2][idx]
    new_terminal[:,29] = np.float(pure_data.shape[0])
    new_terminal[:,-1] = -1
    new_sister[:,0:3] = sister_coordinates[:-1,:]
    new_sister[:,3:6] = sister_coordinates[1:,:]
    new_sister[:,-1] = -1
    new_sister[:,20] = np.linalg.norm(sister_coordinates[1:,:]-sister_coordinates[:-1,:],axis=1)
    new_sister[:,21] = local_opt[0][idx]*local_opt[3][idx]
    new_sister[:,29] = np.float(pure_data.shape[0])+1
    new_parent[:,0:3] = parent_coordinates[:-1,:]
    new_parent[:,3:6] = parent_coordinates[1:,:]
    new_parent[:,20] = np.linalg.norm(parent_coordinates[1:,:]-parent_coordinates[:-1,:],axis=1)
    new_parent[:,29] = np.float(vessel_start)
    new_parent[:,-1] = -1
    if vessel_start != 0:
        new_parent[:,21] = local_opt[0][idx]*local_opt[4][idx]
    else:
        new_parent[:,21] = local_opt[0][idx]
    for i in range(max([terminal_segments,sister_segments,new_parent.shape[0]])):
        if i < terminal_segments:
            basis(new_terminal,i)
        if i < sister_segments:
            basis(new_sister,i)
        if i < new_parent.shape[0]:
            basis(new_parent,i)
    print('adding')
    new_transient = np.vstack((new_transient,new_terminal,new_sister,new_parent))
    dist, add_patches = patch_tree.query(pt2,k=5)
    pure_data = np.vstack((pure_data,np.zeros((2,data.shape[1]))))
    print(xyz)
    print(pt2)
    data,sub_division_index = add_bifurcation(pure_data,new_transient,xyz,pt2,vessel_start,Qterm,
                                              local_opt[0],local_opt[1],local_opt[2],local_opt[3],
                                              np.array(local_opt[6]),np.array(local_opt[7]),
                                              np.array(local_opt[8]),np.array(local_opt[9]),
                                              np.array(local_opt[10]),np.array(local_opt[11]),
                                              np.array(local_opt[12]),np.array(local_opt[13]),
                                              local_opt[14],local_opt[15],idx,sub_division_map,
                                              sub_division_index)
    k += 2
    if not isforest:
        return data,k,sub_division_map,sub_division_index
    else:
        results = [data,k,sub_division_map,sub_division_index]
        return new_terminal,new_sister,new_parent,results
