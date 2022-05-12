import numpy as np
import numba as nb
from functools import partial

@nb.jit(nopython=True,cache=True)
def fast_point(a,b,n=np.array([1.0,1.0,1.0],dtype=np.float64),
          e1=np.array([0.0,0.0,0.0],dtype=np.float64),
          p_0=np.array([0.0,0.0,0.0],dtype=np.float64)):
    if n[2] == -1:
        e1 = np.array([-1,0,0],dtype=np.float64)
        e2 = np.array([0,-1,0],dtype=np.float64)
    else:
        e2 = np.cross(e1,n)
    p = np.add(p_0,a.reshape((-1,1))*e1)
    p = np.add(p,b.reshape((-1,1))*e2)
    return p

a = np.array([0.5,0.6],dtype=np.float64)
b = np.array([0.7,0.8],dtype=np.float64)

#@nb.jit(nopython=True,cache=True)
def fast_local_function(data,terminal_point,edge,
                   gamma,nu,Qterm,Pperm,Pterm,precomputed=None):
    # repository for calculated and adjusted values
    reduced_resistances = []
    reduced_lengths = []
    bifurcations = []
    flows = [] # should only be changing path (not alt paths)
    main_idx = [] # index 0: data[:,15]; index 1: data[:,16]
    alt_idx = []
    new_scale = [] # compound bifurcations
    alt_scale = []
    # begin crawling through tree
    total_flow = data[edge,22]*2+Qterm*2
    triad = ((data[edge,22]+Qterm)*data[edge,0:3]+data[edge,22]*data[edge,3:6]+Qterm*terminal_point)/total_flow
    uu = data[edge,0:3] - terminal_point
    uu = uu/np.linalg.norm(uu)
    vv = data[edge,3:6] - terminal_point
    vv = vv/np.linalg.norm(vv)
    n = np.cross(vv,uu)
    if precomputed is None:
        #local_points = fast_point(a,b,n=n,e1=uu,p_0=triad)
        return
    else:
        local_points = precomputed
    #local_points_v2 = fast_point(a,b,n=n,e1=uu,p_0=triad)
    #print(local_points_v2)
    #print(local_points_v2.shape)
    #print(precomputed)
    #print(precomputed.shape)
    upstream = data[edge,0:3]
    downstream = data[edge,3:6]
    upstream_length = np.sqrt(np.sum(np.square(local_points-upstream),axis=1))
    downstream_length = np.sqrt(np.sum(np.square(local_points-downstream),axis=1))
    terminal_length = np.sqrt(np.sum(np.square(local_points-terminal_point),axis=1))
    # Begin calculating reduced resistance and bifurcation ratios
    R_terminal = (8*nu/np.pi)*terminal_length
    R_terminal_sister = (8*nu/np.pi)*downstream_length + (data[edge,25]-(8*nu/np.pi)*data[edge,20])
    f_terminal = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(gamma/4)) ** (-1/gamma)
    f_terminal_sister = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(-gamma/4)) ** (-1/gamma)
    R_0 = (8*nu/np.pi)*upstream_length + ((f_terminal**4)/R_terminal +
                                          (f_terminal_sister**4)/R_terminal_sister)**-1
    reduced_resistances.append(R_0)
    L_0 = f_terminal ** 2 * terminal_length + f_terminal_sister ** 2 * (downstream_length + data[edge,27])
    reduced_lengths.append(L_0)
    flows.append(data[edge,22]+Qterm)
    main_idx.append(edge)
    bifurcations.append(f_terminal_sister) # 0 -> 15; 1-> 16
    bifurcations.append(f_terminal)
    if edge == 0:
        # on the terminating round the new terminal is always 16 and the existing is 15
        #bifurcations.append([f_terminal_sister,f_terminal])
        L_0 = upstream_length + L_0
        R_0 = (((data[edge,22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
        alt_idx.append(-1)
        new_scale.append(np.ones(f_terminal.shape))
        alt_scale.append(np.ones(f_terminal.shape))
        return (R_0,L_0,f_terminal,f_terminal_sister,None,local_points,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17])
    if int(data[edge,15]) == previous:
        alt = int(data[edge,16])
        alt_idx.append(alt)
        position = 0
        #bifurcations.append([f_terminal,f_terminal_sister])
    else:
        alt = int(data[edge,15])
        alt_idx.append(alt)
        position = 1
        #bifurcations.append([f_terminal_sister,f_terminal])
    f_changed = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (gamma/4)) ** (-1/gamma)
    f_stagnant = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (-gamma/4)) ** (-1/gamma)
    f_terminal = f_terminal*f_changed
    f_terminal_sister = f_terminal_sister*f_changed
    f_parent = f_changed
    R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4) /R_0 + (f_stagnant ** 4) /data[alt,25]) ** -1
    L_0 = f_changed ** 2 *(upstream_length+L_0) + f_stagnant ** 2 * (data[alt,20] + data[alt,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22]+Qterm)
    main_idx.append(edge)
    new_scale.append(f_changed)
    alt_scale.append(f_stagnant)
    if position == 0:
        bifurcations.append(f_changed)
        bifurcations.append(f_stagnant)
    else:
        bifurcations.append(f_stagnant)
        bifurcations.append(f_changed)
    if edge == 0:
        L_0 = data[edge,20]+L_0
        R_0 = (((data[edge,22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
        return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,local_points,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17])
    while edge >= 0:
        if int(data[edge, 15]) == previous:
            alt = int(data[edge,16])
            alt_idx.append(alt)
            position = 0
        else:
            alt = int(data[edge,15])
            alt_idx.append(alt)
            position = 1
        f_changed = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (gamma/4)) ** (-1/gamma)
        f_stagnant = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (-gamma/4)) ** (-1/gamma)
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_parent*f_changed
        R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4 / R_0) + (f_stagnant ** 4 / data[alt, 25])) ** -1
        L_0 = f_changed ** 2 * (data[previous, 20] + L_0) + f_stagnant ** 2 *(data[alt, 20] + data[alt, 27])
        reduced_resistances.append(R_0)
        reduced_lengths.append(L_0)
        flows.append(data[edge,22]+Qterm)
        main_idx.append(edge)
        if edge >= 0:
            for j in range(len(new_scale)):
                new_scale[j] = new_scale[j]*f_changed
                alt_scale[j] = alt_scale[j]*f_changed
            new_scale.append(f_changed)
            #alt_scale = [bif*f_changed for bif in alt_scale]
            alt_scale.append(f_stagnant)
        #else:
        #    new_scale.append(np.ones(f_changed.shape))
        #    alt_scale.append(np.ones(f_changed.shape))
        if position == 0:
            bifurcations.append(f_changed)
            bifurcations.append(f_stagnant)
        else:
            bifurcations.append(f_stagnant)
            bifurcations.append(f_changed)
        previous = edge
        edge = int(data[edge,17])
    L_0 = data[0,20] + L_0
    R_0 = (((data[0, 22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
    return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,local_points,reduced_resistances,
            reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
            R_terminal_sister)
