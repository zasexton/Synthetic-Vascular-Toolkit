import numba as nb
import numpy as np
from .add_edge import *
from .basis import *
from .calculate_length import *
from copy import deepcopy
import time

#@nb.jit(nopython=True,cache=True,nogil=True)
def add_bifurcation(tree,parent_edge,candidate_point,results,optimum,isforest=False):
    ########### UNPACK LOCAL OPTIMIZATION DATA ##############
    start = time.time()
    R0  = results[0]
    L0 = results[1]
    f_terminal = results[2]
    f_sister   = results[3]
    bif_points   = results[5]
    RR = np.array(results[6])
    RL    = np.array(results[7])
    bifs               = np.array(results[8])
    flows              = np.array(results[9])
    main_idx           = np.array(results[10])
    alt_idx            = np.array(results[11])
    main_scale         = np.array(results[12])
    alt_scale          = np.array(results[13])
    R_terminal = results[14]
    R_sister   = results[15]
    if isforest:
        sub_division_map    = deepcopy(tree.sub_division_map)
        sub_division_index  = deepcopy(tree.sub_division_index)
    else:
        sub_division_map    = tree.sub_division_map
        sub_division_index  = tree.sub_division_index
    Qterm               = tree.parameters['Qterm']
    tree.time['add_1'].append(time.time()-start)
    ###########################################################
    ############### Copy Data and Make New ####################
    start = time.time()
    if isforest:
        data = deepcopy(tree.data)
    else:
        data = tree.data
    #if sum(tree.data[tree.data[:,-1]==-1]) == 0:
    #    nosegments = True
    #else:
    #    nosegments = False
    #    segment_data = deepcopy(tree.data[tree.data[:,-1]==-1])
    nosegments = True
    data = np.vstack((data,np.zeros((2,data.shape[1])))) ## would need to be changed for adding non-linear vessels
    bifurcation_point = bif_points[optimum]
    add_edge(data[:-1,:],bifurcation_point,candidate_point,parent_edge,Qterm)
    data[-2,17] = np.float(parent_edge)
    terminal_sub_division = [-1]
    downstream_node = data[parent_edge,3:6]
    downstream_flow = data[parent_edge,22].item()
    bifurcation_node = data[-2,18].item()
    distal_node = data[parent_edge,19].item()
    left = data[parent_edge,15].item()
    right = data[parent_edge,16].item()
    lbif = data[parent_edge,23]
    rbif = data[parent_edge,24]
    downstream_rr = data[parent_edge,27]
    add_edge(data,bifurcation_point,downstream_node,parent_edge,
             downstream_flow,proximal_idx=bifurcation_node,
             distal_idx=distal_node,left_child=left,
             right_child=right)
    data[-1,17] = np.float(parent_edge)
    sister_sub_division = [-1]
    tree.time['add_2'].append(time.time()-start)
    #pidx_start = sub_division_index[parent_edge]
    #pidx_end = sub_division_index[parent_edge+1]
    start_time = time.time()
    if left > 0 or right > 0: # if this vessel is not a terminal it will have a subdivision map
        start = sub_division_index[parent_edge] #start of the
        if parent_edge+1 == len(sub_division_index):
            end = None
        else:
            end = sub_division_index[parent_edge+1]
        k = sub_division_map[start+1:end]
        data[k,28] = (data[k,28]/data[parent_edge,28])*f_sister[optimum] # replace the old bifurcation with the new scaling
        data[k,26] += 1 # incrament the depth of all subdivision vessels
        sister_sub_division.extend(k)
        data[-1,23] = lbif
        data[-1,24] = rbif
    data[-1,28] = f_sister[optimum]
    data[-1,27] = downstream_rr
    data[-2,28] = f_terminal[optimum]
    data[-1,25] = R_sister[optimum]
    data[-2,25] = R_terminal[optimum]
    tree.time['add_3'].append(time.time()-start_time)
    if int(left) > 0:
        data[int(left),17] = data[-1,-1]
    if int(right) > 0:
        data[int(right),17] = data[-1,-1]
    data[parent_edge,3:6] = bifurcation_point
    data[parent_edge,19] = bifurcation_node
    data[parent_edge,15] = data[-1,-1]
    data[parent_edge,16] = data[-2,-1]
    new_terminal_edge = int(data[-2,-1])
    new_sister_edge = int(data[-1,-1])
    basis(data,parent_edge.item())
    length(data,parent_edge.item())
    sub_division_map.extend(terminal_sub_division) #should be [-1]
    sub_division_index = np.append(sub_division_index,len(sub_division_map)-1)
    sub_division_index = np.append(sub_division_index,len(sub_division_map))
    sub_division_map.extend(sister_sub_division) #should be a copy of the parent subdivision
    #print()
    #print(sub_division_index)
    #sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten()
    #print(sub_division_index)
    start_time = time.time()

    for i in range(len(main_idx)):
        start = sub_division_index[main_idx[i]]
        if main_idx[i] == sub_division_index[-1]:
            end = None
        else:
            end = sub_division_index[main_idx[i]+1]
        if len(sub_division_map[start+1:end]) > 0 and main_idx[i] > 0:
            if alt_idx[i] > -1:
                alt_start = sub_division_index[alt_idx[i]]
                if alt_idx[i] == sub_division_index[-1]:
                    alt_end = None
                else:
                    alt_end = sub_division_index[alt_idx[i]+1]
                tmp = sub_division_map[alt_start+1:alt_end]
                data[tmp,28] = (data[tmp,28]/data[alt_idx[i],28])*alt_scale[i][optimum]
        data[main_idx[i],25] = RR[i][optimum]
        data[main_idx[i],27] = RL[i][optimum]
        data[main_idx[i],22] = flows[i]
        data[main_idx[i],23] = bifs[i*2][optimum]
        data[main_idx[i],24] = bifs[i*2+1][optimum]
        if main_idx[i] > 0:
            data[main_idx[i],28] = main_scale[i][optimum]
            if alt_idx[i] > -2:
                data[alt_idx[i],28] = alt_scale[i][optimum]
    data[0,21] = R0[optimum]
    data[0,23] = bifs[-2][optimum]
    data[0,24] = bifs[-1][optimum]
    data[0,25] = RR[-1][optimum]
    data[0,27] = RL[-1][optimum]
    data[0,22] = flows[-1]
    data[:,21] = data[:,28]*data[0,21]
    if not nosegments:
        segment_data[:,21] = data[segment_data[:,29].astype(int),28]*data[0,21]
        total_data = np.vstack((data,segment_data))
    else:
        total_data = data

    #total_data = update_map(data,main_idx,alt_idx,main_scale,alt_scale,RR,RL,flows,bifs,R0,sub_division_index,np.array(sub_division_map),optimum,new_sister_edge,new_terminal_edge)
    main_idx = np.flip(np.sort(main_idx))
    for i in range(len(main_idx)):
        sub_division_map.insert(sub_division_index[main_idx[i]]+1,int(new_sister_edge))
        sub_division_map.insert(sub_division_index[main_idx[i]]+1,int(new_terminal_edge))
        #sub_division_index[main_idx[i]] += (len(main_idx)-1-i)*2

    #total_data,main_idx,sub_division_index,sub_division_map = update_map(data,main_idx,alt_idx,main_scale,alt_scale,RR,RL,flows,bifs,R0,sub_division_index,np.array(sub_division_map),optimum,new_sister_edge,new_terminal_edge,sub_division_map)
    #adder = 0
    #for i in range(len(sub_division_index)):
    #    sub_division_index[i] += adder
    #    if i in main_idx:
    #        adder += 2
    tree.time['add_4'].append(time.time()-start_time)
    sub_division_index = update_index(sub_division_index,main_idx)
    #tree.time['add_4'].append(time.time()-start_time)
    #print()
    #print('main_idx: {}'.format(main_idx))
    #print('manual: {}'.format(sub_division_index))
    #sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten()
    #print('np arg: {}'.format(sub_division_index))
    #if time.time()-start > 100:
    #    val = 0
    #else:
    #    val = time.time()-start
    #tree.time['add_4'].append(time.time()-start_time)
    return total_data,sub_division_map,sub_division_index

@nb.jit(nopython=True,cache=True,nogil=True)
def update_index(sub_division_index,main_idx):
    adder = 0
    for i in range(len(sub_division_index)):
        sub_division_index[i] += adder
        if i in main_idx:
            adder += 2
    return sub_division_index

"""
@nb.jit(nopython=True,cache=True,nogil=True)
def update_map(data,main_idx,alt_idx,main_scale,alt_scale,RR,RL,flows,bifs,R0,sub_division_index,sub_division_map,optimum,new_sister_edge,new_terminal_edge):
    for i in range(len(main_idx)):
        start = sub_division_index[main_idx[i]]
        if main_idx[i] == sub_division_index[-1]:
            end = None
        else:
            end = sub_division_index[main_idx[i]+1]
        if len(sub_division_map[start+1:end]) > 0 and main_idx[i] > 0:
            if alt_idx[i] > -1:
                alt_start = sub_division_index[alt_idx[i]]
                if alt_idx[i] == sub_division_index[-1]:
                    alt_end = None
                else:
                    alt_end = sub_division_index[alt_idx[i]+1]
                tmp = sub_division_map[alt_start+1:alt_end]
                data[tmp,28] = (data[tmp,28]/data[alt_idx[i],28])*alt_scale[i][optimum]
        data[main_idx[i],25] = RR[i][optimum]
        data[main_idx[i],27] = RL[i][optimum]
        data[main_idx[i],22] = flows[i]
        data[main_idx[i],23] = bifs[i*2][optimum]
        data[main_idx[i],24] = bifs[i*2+1][optimum]
        if main_idx[i] > 0:
            data[main_idx[i],28] = main_scale[i][optimum]
            if alt_idx[i] > -2:
                data[alt_idx[i],28] = alt_scale[i][optimum]
    data[0,21] = R0[optimum]
    data[0,23] = bifs[-2][optimum]
    data[0,24] = bifs[-1][optimum]
    data[0,25] = RR[-1][optimum]
    data[0,27] = RL[-1][optimum]
    data[0,22] = flows[-1]
    data[:,21] = data[:,28]*data[0,21]

    return data
"""
