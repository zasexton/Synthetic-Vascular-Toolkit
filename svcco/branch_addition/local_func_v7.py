import numpy as np
import numba as nb
from functools import partial
from copy import deepcopy
from scipy.optimize import minimize,approx_fprime

@nb.jit(nopython=True)
def fast_local_function(data,mesh,terminal,edge,gamma,nu,Qterm,Pperm,Pterm):
    ####################
    reduced_resistances = []
    reduced_lengths     = []
    bifurcations        = []
    flows               = [] # should only be changing path (not alt paths)
    main_idx            = [] # index 0: data[:,15]; index 1: data[:,16]
    alt_idx             = []
    new_scale           = [] # compound bifurcations
    alt_scale           = []
    ####################
    total_flow          = data[edge,22].item()+Qterm*2
    upstream            = data[edge,0:3]
    downstream          = data[edge,3:6]
    upstream_length     = np.sqrt(np.sum(np.square(mesh-upstream),axis=1))
    downstream_length   = np.sqrt(np.sum(np.square(mesh-downstream),axis=1))
    terminal_length     = np.sqrt(np.sum(np.square(mesh-terminal),axis=1))
    # Begin calculating reduced resistance and bifurcation ratios
    R_terminal          = (8*nu/np.pi)*terminal_length
    R_terminal_sister   = (8*nu/np.pi)*downstream_length + (data[edge,25]-(8*nu/np.pi)*data[edge,20])
    f_terminal          = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(gamma/4)) ** (-1/gamma)
    f_terminal_sister   = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(-gamma/4)) ** (-1/gamma)
    R_0                 = (8*nu/np.pi)*upstream_length + ((f_terminal**4)/R_terminal + (f_terminal_sister**4)/R_terminal_sister)**-1
    L_0                 = f_terminal ** 2 * terminal_length + f_terminal_sister ** 2 * (downstream_length + data[edge,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
    main_idx.append(edge.item())
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
        return (R_0,L_0,f_terminal,f_terminal_sister,None,mesh,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    if int(data[edge,15].item()) == previous:
        alt = int(data[edge,16].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 0
        #bifurcations.append([f_terminal,f_terminal_sister])
    else:
        alt = int(data[edge,15].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 1
        #bifurcations.append([f_terminal_sister,f_terminal])
    if alt == -2:
        f_changed = np.array([1.0])
        f_stagnant = np.array([0.0])
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20].item() + R_0
        L_0 = f_changed ** 2 *(upstream_length+L_0)
    else:
        f_changed = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (gamma/4)) ** (-1/gamma)
        f_stagnant = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (-gamma/4)) ** (-1/gamma)
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4) /R_0 + (f_stagnant ** 4) /data[alt,25]) ** -1
        L_0 = f_changed ** 2 *(upstream_length+L_0) + f_stagnant ** 2 * (data[alt,20] + data[alt,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
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
        return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,mesh,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    while edge >= 0:
        if int(data[edge, 15].item()) == previous:
            alt = int(data[edge,16].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 0
        else:
            alt = int(data[edge,15].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 1
        if alt == -2:
            f_changed = np.array([1.0])
            f_stagnant = np.array([0.0])
            R_0 =  (8*nu/np.pi)*data[edge,20] + R_0
            L_0 =  f_changed ** 2 * (data[previous, 20] + L_0)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
        else:
            f_changed = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (gamma/4)) ** (-1/gamma)
            f_stagnant = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (-gamma/4)) ** (-1/gamma)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4 / R_0) + (f_stagnant ** 4 / data[alt, 25])) ** -1
            L_0 = f_changed ** 2 * (data[previous, 20] + L_0) + f_stagnant ** 2 *(data[alt, 20] + data[alt, 27])
        reduced_resistances.append(R_0)
        reduced_lengths.append(L_0)
        flows.append(data[edge,22].item()+Qterm)
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
        edge = int(data[edge,17].item())
    L_0 = data[0,20] + L_0
    R_0 = (((data[0, 22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
    return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,mesh,reduced_resistances,
            reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
            R_terminal_sister)


def fast_local_function_test(data,mesh,terminal,edge,gamma,nu,Qterm,Pperm,Pterm):
    ####################
    reduced_resistances = []
    reduced_lengths     = []
    bifurcations        = []
    flows               = [] # should only be changing path (not alt paths)
    main_idx            = [] # index 0: data[:,15]; index 1: data[:,16]
    alt_idx             = []
    new_scale           = [] # compound bifurcations
    alt_scale           = []
    ####################
    #total_flow          = data[edge,22].item()+Qterm*2
    upstream            = data[edge,0:3]
    downstream          = data[edge,3:6]
    upstream_length     = np.sqrt(np.sum(np.square(mesh-upstream),axis=1))
    downstream_length   = np.sqrt(np.sum(np.square(mesh-downstream),axis=1))
    terminal_length     = np.sqrt(np.sum(np.square(mesh-terminal),axis=1))
    # Begin calculating reduced resistance and bifurcation ratios
    R_terminal          = (8*nu/np.pi)*terminal_length
    R_terminal_sister   = (8*nu/np.pi)*downstream_length + (data[edge,25]-(8*nu/np.pi)*data[edge,20])
    f_terminal          = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(gamma/4)) ** (-1/gamma)
    f_terminal_sister   = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(-gamma/4)) ** (-1/gamma)
    R_0                 = (8*nu/np.pi)*upstream_length + ((f_terminal**4)/R_terminal +
                          (f_terminal_sister**4)/R_terminal_sister)**-1
    L_0                 = f_terminal ** 2 * terminal_length + f_terminal_sister ** 2 * (downstream_length + data[edge,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
    main_idx.append(edge.item())
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
        return (R_0,L_0,f_terminal,f_terminal_sister,None,mesh,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    if int(data[edge,15].item()) == previous:
        alt = int(data[edge,16].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 0
        #bifurcations.append([f_terminal,f_terminal_sister])
    else:
        alt = int(data[edge,15].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 1
        #bifurcations.append([f_terminal_sister,f_terminal])
    if alt == -2:
        f_changed = np.array([1.0])
        f_stagnant = np.array([0.0])
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20].item() + R_0
        L_0 = f_changed ** 2 *(upstream_length+L_0)
    else:
        f_changed = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (gamma/4)) ** (-1/gamma)
        f_stagnant = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (-gamma/4)) ** (-1/gamma)
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4) /R_0 + (f_stagnant ** 4) /data[alt,25]) ** -1
        L_0 = f_changed ** 2 *(upstream_length+L_0) + f_stagnant ** 2 * (data[alt,20] + data[alt,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
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
        return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,mesh,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    while edge >= 0:
        if int(data[edge, 15].item()) == previous:
            alt = int(data[edge,16].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 0
        else:
            alt = int(data[edge,15].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 1
        if alt == -2:
            f_changed = np.array([1.0])
            f_stagnant = np.array([0.0])
            R_0 =  (8*nu/np.pi)*data[edge,20] + R_0
            L_0 =  f_changed ** 2 * (data[previous, 20] + L_0)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
        else:
            f_changed = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (gamma/4)) ** (-1/gamma)
            f_stagnant = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (-gamma/4)) ** (-1/gamma)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4 / R_0) + (f_stagnant ** 4 / data[alt, 25])) ** -1
            L_0 = f_changed ** 2 * (data[previous, 20] + L_0) + f_stagnant ** 2 *(data[alt, 20] + data[alt, 27])
        reduced_resistances.append(R_0)
        reduced_lengths.append(L_0)
        flows.append(data[edge,22].item()+Qterm)
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
        edge = int(data[edge,17].item())
    L_0 = data[0,20] + L_0
    R_0 = (((data[0, 22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
    return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,mesh,reduced_resistances,
            reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
            R_terminal_sister)

"""
def fast_local_function_bind(bif_point,data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm):
    upstream            = data[edge,0:3]
    downstream          = data[edge,3:6]
    upstream_length     = np.sqrt(np.sum(np.square(bif_point-upstream),axis=1))
    downstream_length   = np.sqrt(np.sum(np.square(bif_point-downstream),axis=1))
    terminal_length     = np.sqrt(np.sum(np.square(bif_point-terminal),axis=1))
"""
def path(data,edge):
    path_to_root = [edge]
    alt_paths    = []
    while path[-1] > 0:
        previous = edge
        edge     = int(data[edge,17].item())
        path_to_root.append(edge)
        left     = int(data[edge,15].item())
        right    = int(data[edge,16].item())
        if left == previous:
            alt_paths.append(right)
        else:
            alt_paths.append(left)
    return path_to_root,alt_paths

@nb.jit(nopython=True)
def volume(x,data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm):
    ####################
    reduced_resistances = []
    reduced_lengths     = []
    bifurcations        = []
    flows               = [] # should only be changing path (not alt paths)
    main_idx            = [] # index 0: data[:,15]; index 1: data[:,16]
    alt_idx             = []
    new_scale           = [] # compound bifurcations
    alt_scale           = []
    rescale             = np.zeros(data.shape[0])
    ####################
    #total_flow          = data[edge,22].item()+Qterm*2
    upstream            = data[edge,0:3]
    downstream          = data[edge,3:6]
    upstream_length     = np.sqrt(np.sum(np.square(x-upstream),axis=1))
    downstream_length   = np.sqrt(np.sum(np.square(x-downstream),axis=1))
    terminal_length     = np.sqrt(np.sum(np.square(x-terminal),axis=1))
    # Begin calculating reduced resistance and bifurcation ratios
    R_terminal          = (8*nu/np.pi)*terminal_length
    R_terminal_sister   = (8*nu/np.pi)*downstream_length + (data[edge,25]-(8*nu/np.pi)*data[edge,20])
    f_terminal          = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(gamma/4)) ** (-1/gamma)
    f_terminal_sister   = (1+((data[edge,22]*R_terminal_sister)/(Qterm*R_terminal))**(-gamma/4)) ** (-1/gamma)
    R_0                 = (8*nu/np.pi)*upstream_length + ((f_terminal**4)/R_terminal +
                          (f_terminal_sister**4)/R_terminal_sister)**-1
    L_0                 = f_terminal ** 2 * terminal_length + f_terminal_sister ** 2 * (downstream_length + data[edge,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
    main_idx.append(edge.item())
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
        return (R_0,L_0,f_terminal,f_terminal_sister,None,x,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    if int(data[edge,15].item()) == previous:
        alt = int(data[edge,16].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 0
        #bifurcations.append([f_terminal,f_terminal_sister])
    else:
        alt = int(data[edge,15].item())
        if alt == -1:
            alt = -2
        alt_idx.append(alt)
        position = 1
        #bifurcations.append([f_terminal_sister,f_terminal])
    if alt == -2:
        f_changed = np.array([1.0])
        f_stagnant = np.array([0.0])
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20].item() + R_0
        L_0 = f_changed ** 2 *(upstream_length+L_0)
    else:
        f_changed = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (gamma/4)) ** (-1/gamma)
        f_stagnant = (1+((data[alt,22]*data[alt,25])/((data[previous,22]+Qterm)*R_0)) ** (-gamma/4)) ** (-1/gamma)
        f_terminal = f_terminal*f_changed
        f_terminal_sister = f_terminal_sister*f_changed
        f_parent = f_changed
        R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4) /R_0 + (f_stagnant ** 4) /data[alt,25]) ** -1
        L_0 = f_changed ** 2 *(upstream_length+L_0) + f_stagnant ** 2 * (data[alt,20] + data[alt,27])
    reduced_resistances.append(R_0)
    reduced_lengths.append(L_0)
    flows.append(data[edge,22].item()+Qterm)
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
        return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,x,reduced_resistances,
                reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
                R_terminal_sister)
    previous = edge
    edge = int(data[edge,17].item())
    while edge >= 0:
        if int(data[edge, 15].item()) == previous:
            alt = int(data[edge,16].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 0
        else:
            alt = int(data[edge,15].item())
            if alt == -1:
                alt = -2
            alt_idx.append(alt)
            position = 1
        if alt == -2:
            f_changed = np.array([1.0])
            f_stagnant = np.array([0.0])
            R_0 =  (8*nu/np.pi)*data[edge,20] + R_0
            L_0 =  f_changed ** 2 * (data[previous, 20] + L_0)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
        else:
            f_changed = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (gamma/4)) ** (-1/gamma)
            f_stagnant = (1+((data[alt, 22]*data[alt,25])/((Qterm+data[previous,22])*R_0)) ** (-gamma/4)) ** (-1/gamma)
            f_terminal = f_terminal*f_changed
            f_terminal_sister = f_terminal_sister*f_changed
            f_parent = f_parent*f_changed
            R_0 = (8*nu/np.pi)*data[edge,20] + ((f_changed ** 4 / R_0) + (f_stagnant ** 4 / data[alt, 25])) ** -1
            L_0 = f_changed ** 2 * (data[previous, 20] + L_0) + f_stagnant ** 2 *(data[alt, 20] + data[alt, 27])
        reduced_resistances.append(R_0)
        reduced_lengths.append(L_0)
        flows.append(data[edge,22].item()+Qterm)
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
        edge = int(data[edge,17].item())
    L_0 = data[0,20] + L_0
    R_0 = (((data[0, 22]+Qterm)*R_0)/(Pperm-Pterm))**(1/4)
    return (R_0,L_0,f_terminal,f_terminal_sister,f_parent,x,reduced_resistances,
            reduced_lengths,bifurcations,flows,main_idx,alt_idx,new_scale,alt_scale,R_terminal,
            R_terminal_sister)


def constructor(data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm,lam,mu,maxiter,method='L-BFGS-B'):
    proximal = data[edge,0:3]
    distal=data[edge,3:6]
    d_min    = min([np.linalg.norm(proximal-distal),
                    np.linalg.norm(terminal-distal),
                    np.linalg.norm(terminal-proximal)])*0.25
    d_sides    =   [np.linalg.norm(proximal-distal),
                    np.linalg.norm(terminal-distal),
                    np.linalg.norm(terminal-proximal)]
    s0 = (d_sides[0]+d_sides[1]+d_sides[2])/2
    area = (s0*abs(s0-d_sides[0])*abs(s0-d_sides[1])*abs(s0-d_sides[2]))**0.5
    #rescale  = deepcopy(data[:,28])
    def cost(st,func=volume,terminal=terminal,proximal=data[edge,0:3],distal=data[edge,3:6],data=data,edge=edge,gamma=gamma,nu=nu,Qterm=Qterm,Pperm=Pperm,Pterm=Pterm,d_min=d_min,d_sides=d_sides,area=area,lam=lam,mu=mu):
        s = st[0]
        t = st[1]
        if s > 1:
            s = 1
        elif s < 0:
            s = 0
        if t > 1:
            t = 1
        elif t < 0:
            t = 0
        x = proximal.reshape(1,-1)*(1-t)*s + \
            distal.reshape(1,-1)*(t*s)+\
            terminal.reshape(1,-1)*(1-s)
        d = [np.linalg.norm(proximal-x),
             np.linalg.norm(distal-x),
             np.linalg.norm(terminal-x)]
        dd = min(d)
        if dd < d_min:
            return 10000
        else:
            res = volume(x,data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm)
            return np.pi*res[0]**lam*res[1]**mu
        """
        s1 = (d[0]+d[1]+d_sides[0])/2
        s2 = (d[1]+d[2]+d_sides[1])/2
        s3 = (d[2]+d[0]+d_sides[2])/2
        a1 = (s1*abs(s1-d[0])*abs(s1-d[1])*abs(s1-d_sides[0]))**0.5
        a2 = (s2*abs(s2-d[1])*abs(s2-d[2])*abs(s2-d_sides[1]))**0.5
        a3 = (s3*abs(s3-d[2])*abs(s3-d[0])*abs(s3-d_sides[2]))**0.5
        at = a1+a2+a3
        """
        #if not np.isclose(at,area):
        #    return 10000
        #else:
        #    res = volume(x,data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm)
        #    return np.pi*res[0]**lam*res[1]**mu
    x0 = np.array([0.5,0.5])
    if method == 'Newton-CG' or method == 'dogleg' or method == 'trust-ncg':
        jac = lambda x: approx_fprime(x,cost,max(d_sides)/maxiter)
        res = minimize(cost,x0,method=method,options={'maxiter':maxiter},jac=jac)
    else:
        res = minimize(cost,x0,method=method,options={'maxiter':maxiter})
    st = res.x
    s = st[0]
    t = st[1]
    if s > 1:
        s = 1
    elif s < 0:
        s = 0
    if t > 1:
        t = 1
    elif t < 0:
        t = 0
    x = proximal.reshape(1,-1)*(1-t)*s + \
        distal.reshape(1,-1)*(t*s)+\
        terminal.reshape(1,-1)*(1-s)
    return volume(x,data,terminal,edge,gamma,nu,Qterm,Pperm,Pterm)
