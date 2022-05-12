import numpy as np
from .basis import *
from .calculate_length import *
from .update import *
from .calculate_radii import *

def add_geodesic_path(data,path,lengths,parent_vessel_idx,Qterm,gamma,nu,
                      Pperm, Pterm, sub_division_index,sub_division_map):
    # Run if the added path is the root vessel
    if np.all(data[0,:]==0):
        for i in range(len(path)-1):
            data[i,0:3] = path[i]
            data[i,3:6] = path[i+1]
            basis(data,i)
            data[i,15] = i+1
            data[i,16] = -1
            data[i,17] = i - 1
            data[i,18] = i
            data[i,19] = i+1
            data[i,20] = lengths[i] #length(data,i)
            data[i,22] = Qterm
            if idx < len(path) - 2:
                data[i,23] = 1.0
            else:
                data[i,23] = 0.0
            data[i,24] = 0
            data[i,25] = (8*nu/np.pi)*np.sum(lengths[i:])
            data[i,26] = 0
            data[i,27] = np.sum(lengths[i:])
            data[i,28] = 1.0
            data[i,29] = -1
            data[i,-1] = 0
            data[i,21] = ((data[0,25]*data[0,22])/(Pperm-Pterm)) ** (1/4)
            sub_division = [-1]
            sub_division_index.append(len(sub_division_map))
            sub_division.extend(list(range(i+1,data.shape[0])))
            sub_division_map.extend(sub_division)
    else:
        ## This is going to be a bit complicated
        ## TODO needs an update strategy for updaing index 28 of vessels
        ## the subvessels are of the same depth which needs to be considered
        ## when making a strategy.

        ############################
        # Add Geodesic Terminal Vessel
        ############################
        count = int(max(data[:,19]))
        new_vessels = []
        for idx in range(len(path)-1):
            i = idx + count
            data[i,0:3] = path[idx]
            data[i,3:6] = path[idx+1]
            basis(data,i)
            if idx < len(path) - 2:
                data[i,15] = i+1
            else:
                data[i,15] = -1
            data[i,16] = -1
            if idx == 0:
                data[i,17] = data[parent_vessel_idx,-1]
            else:
                data[i,17] = i - 1
            if idx == 0:
                bif_node = i+1
                bif_vessel = i
            data[i,18] = i+1
            data[i,19] = i+2
            data[i,20] = lengths[idx] #length(data,i)
            data[i,22] = Qterm
            if idx < len(path) - 2:
                data[i,23] = 1.0
            else:
                data[i,23] = 0.0
            data[i,24] = 0
            data[i,25] = (8*nu/np.pi)*np.sum(lengths[idx:])
            data[i,26] = data[parent_vessel_idx,26]+1
            if idx < len(path) - 2:
                data[i,27] = np.sum(lengths[idx+1:])
            else:
                data[i,27] = 0.0
            data[i,28] = 1.0
            data[i,29] = -1
            data[i,-1] = i
            sub_division = [-1]
            #sub_division_index.append(sub_division_map)
            new_vessels.append(i)
            if idx < len(path) - 2:
                sub_division.extend(list(range(i+1,data.shape[0]-1)))
            sub_division_map.extend(sub_division)
        ########################
        # Add Sister vessel
        ########################
        data[-1,0:3] = path[0]
        data[-1,3:6] = data[parent_vessel_idx,3:6]
        basis(data,data.shape[0]-1)
        data[-1,15]  = data[parent_vessel_idx,15]
        data[-1,16]  = data[parent_vessel_idx,16]
        data[-1,17]  = data[parent_vessel_idx,-1]
        data[-1,18]  = bif_node
        data[-1,19]  = data[parent_vessel_idx,19]
        length(data,data.shape[0]-1)
        data[-1,22]  = data[parent_vessel_idx,22]
        data[-1,23]  = data[parent_vessel_idx,23]
        data[-1,24]  = data[parent_vessel_idx,24]
        data[-1,25]  = data[parent_vessel_idx,25] - data[parent_vessel_idx,20]*(8*nu/np.pi) + data[-1,20]*(8*nu/np.pi)
        data[-1,26]  = data[parent_vessel_idx,26]+1
        data[-1,27]  = data[parent_vessel_idx,27]
        data[-1,28]  = data[parent_vessel_idx,28]
        data[-1,29]  = -1
        data[-1,-1]  = data.shape[0]-1
        print('PARENT 28: {}'.format(data[parent_vessel_idx,28]))
        ########################
        # Replace children parent id with new id
        ########################
        left = int(data[-1,15])
        right = int(data[-1,16])
        if left != -1:
            data[left,17] = data[-1,-1]
        if right != -1:
            data[right,17] = data[-1,-1]
        new_vessels.append(data.shape[0]-1)
        ########################
        # Update parent vessel
        ########################
        data[parent_vessel_idx,3:6] = path[0]
        basis(data,parent_vessel_idx)
        data[parent_vessel_idx,15]  = bif_vessel
        data[parent_vessel_idx,16]  = data.shape[0]-1
        data[parent_vessel_idx,19]  = bif_node
        length(data,parent_vessel_idx)
        #data[parent_vessel_idx,22]  += Qterm
        sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten()
        left = int(data[parent_vessel_idx,15])
        right = int(data[parent_vessel_idx,16])
        LR = ((data[left,22]*data[left,25])/(data[right,22]*data[right,25])) **(1/4)
        lbif = (1 + LR ** (-gamma)) ** (-1/gamma)
        rbif = (1 + LR ** (gamma)) ** (-1/gamma)
        sister_sub_division = [-1]
        if data[-1,15] > 0 or data[-1,16] > 0:
            start = sub_division_index[parent_vessel_idx]
            if sub_division_index[parent_vessel_idx] == sub_division_index[-1]:
                end = None
            else:
                end = sub_division_index[parent_vessel_idx+1]
            k = sub_division_map[start+1:end]
            #data[k,28] = (data[k,28]/data[parent_vessel_idx,28])*rbif
            print(k)
            data[k,26] += 1
            sister_sub_division.extend(k)
        data[parent_vessel_idx,23] = lbif
        data[parent_vessel_idx,24] = rbif
        #data[parent_vessel_idx,25] = (8*nu/np.pi)*data[parent_vessel_idx] + ((lbif ** 4 /data[left,25]) + \
        #                                                  (rbif ** 4 /data[right,25])) ** -1
        data[parent_vessel_idx,27] = lbif**2 * (data[left,20] + data[left,27]) +\
                                     rbif**2 * (data[right,20] + data[right,27])
        #data[parent_vessel_idx,28] = 1
        sub_division_map.extend(sister_sub_division)
        parent_sub_idx = sub_division_index[parent_vessel_idx]
        sub_division_map[parent_sub_idx+1:parent_sub_idx+1] = new_vessels
        sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten()
        main_idx  = []
        alt_idx   = []
        while parent_vessel_idx >= 0:
            left = int(data[parent_vessel_idx,15])
            right = int(data[parent_vessel_idx,16])
            data[parent_vessel_idx,22] += Qterm
            main_idx.append(parent_vessel_idx)
            #if left in main_idx:
            #    alt_idx.append(right)
            #else:
            #    alt_idx.append(left)
            if left > 0 and right > 0:
                LR = ((data[left,22]*data[left,25])/(data[right,22]*data[right,25])) ** (1/4)
                lbif = (1 + LR ** (-gamma)) ** (-1/gamma)
                rbif = (1 + LR ** (gamma)) ** (-1/gamma)
            elif left > 0 and right < 0:
                lbif  = 1
                rbif  = 0
            elif right > 0 and left <  0:
                lbif = 0
                rbif = 1
            else:
                lbif = 0
                rbif = 0
            if left > 0 and right > 0:
                data[parent_vessel_idx,25] = (8*nu/np.pi) * data[parent_vessel_idx,20] +\
                                             ((lbif**4/ data[left,25]) +
                                             (rbif**4/ data[right,25])) ** -1
            elif left > 0 and right < 0:
                data[parent_vessel_idx,25] = (8*nu/np.pi) * data[parent_vessel_idx,20] +\
                                             data[left,25]
            elif right > 0 and left < 0:
                data[parent_vessel_idx,25] = (8*nu/np.pi) * data[parent_vessel_idx,20] +\
                                             data[right,25]
            data[parent_vessel_idx,23] = lbif
            data[parent_vessel_idx,24] = rbif
            data[parent_vessel_idx,27] = lbif ** 2 * (data[left,20] + data[left,27]) +\
                                         rbif ** 2 * (data[right,20] + data[right,27])
            parent_sub_index = sub_division_index[parent_vessel_idx]
            sub_division_map[parent_sub_index+1:parent_sub_index+1] = new_vessels
            #print(len(sub_division_map))
            sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten()
            parent_vessel_idx = int(data[parent_vessel_idx,17])
            left = int(data[parent_vessel_idx,15])
            right = int(data[parent_vessel_idx,16])
            if parent_vessel_idx > 0:
                if left in main_idx:
                    alt_idx.append(right)
                else:
                    alt_idx.append(left)
            else:
                alt_idx.append(-1)
        #print(main_idx)
        #print(alt_idx)
        data[0,21] = ((data[0,25]*data[0,22])/(Pperm-Pterm)) ** (1/4)
        main_idx = main_idx[::-1]
        alt_idx = alt_idx[::-1]
        #print('main idx: {}'.format(len(main_idx)))
        #print('alt  idx: {}'.format(len(alt_idx)))
        #main_idx.pop(0)
        #alt_idx.pop(0)
        new_main_scale = 1
        new_alt_scale = 1

        if int(data[main_idx[0],15]) in main_idx:
            new_main_scale *= data[main_idx[0],23]
            #if alt_idx[0] > 0:
            new_alt_scale *= data[main_idx[0],24]

        if int(data[main_idx[0],16]) in main_idx:
            new_main_scale *= data[main_idx[0],24]
            #if alt_idx[0] > 0:
            new_alt_scale = data[main_idx[0],23]
        main_idx.pop(0)
        alt_idx.pop(0)
        print('main_idx: {}'.format(main_idx))
        print('alt_idx:  {}'.format(alt_idx))
        if len(main_idx) == 0:
            print('main idx length = 0')
        for i in range(len(main_idx)):
            print('main: {}, scale: {}'.format(main_idx[i],new_main_scale))
            print('alt: {},  scale: {}'.format(alt_idx[i],new_alt_scale))
            data[main_idx[i],28] = new_main_scale
            start = sub_division_index[main_idx[i]]
            if main_idx[i] == sub_division_index[-1]:
                end = None
            else:
                end = sub_division_index[main_idx[i]+1]
            if alt_idx[i] > -1:
                alt_start = sub_division_index[alt_idx[i]]
                if sub_division_index[alt_idx[i]] == sub_division_index[-1]:
                    alt_end = None
                else:
                    alt_end = sub_division_index[alt_idx[i]+1]
                tmp = sub_division_map[alt_start+1:alt_end]
                data[tmp,28] = (data[tmp,28]/data[alt_idx[i],28])*(new_alt_scale)
                data[alt_idx[i],28] = new_alt_scale
            """
            if len(sub_division_map[start+1:end]) > 0 and main_idx[i] > 0:
                if alt_idx[i] > -1:
                    alt_start = sub_division_index[alt_idx[i]]
                    if sub_division_index[alt_idx[i]] == sub_division_index[-1]:
                        alt_end = None
                    else:
                        alt_end = sub_division_index[alt_idx[i]+1]
                    tmp = sub_division_map[alt_start+1:alt_end]
                    data[tmp,28] = (data[tmp,28]/data[alt_idx[i],28])*new_alt_scale
            """
            #if alt_idx[i] > 0:
            #    data[alt_idx[i],28] = new_alt_scale
            #    #data[alt_idx[i],21] = new_alt_scale*data[0,21]
            if int(data[main_idx[i],15]) in main_idx:
                #if alt_idx[0] > 0:
                new_alt_scale = new_main_scale*data[main_idx[i],24]
                new_main_scale *= data[main_idx[i],23]
                #else:
                #    new_alt_scale *= data[0,23]
            if int(data[main_idx[i],16]) in main_idx:
                #if alt_idx[0] > 0:
                new_alt_scale = new_main_scale*data[main_idx[i],23]
                new_main_scale *= data[main_idx[i],24]
                #else:
                #    new_alt_scale *= data[0,24]
        ########################
        # Update geodesic vessel
        ########################
        start = sub_division_index[bif_vessel]
        if sub_division_index[bif_vessel] == sub_division_index[-1]:
            end = None
        else:
            end = sub_division_index[bif_vessel+1]
        tmp = sub_division_map[start+1:end]
        #print(tmp)
        #print(new_main_scale)
        data[tmp,28] = new_main_scale
        data[bif_vessel,28] = new_main_scale
        print('SISTER ALT SCALE: {}'.format(new_alt_scale))
        #data[data.shape[0]-1,28] = data[main_idx[-1],24]
        #rbif = data[main_idx[-1],24]
        #if len(sub_division_map[start+1:end]) > 0:
        #    if (data.shape[0]-1) > -1:
        alt_start = sub_division_index[data.shape[0] - 1]
        if sub_division_index[data.shape[0]-1] == sub_division_index[-1]:
            alt_end = None
        else:
            alt_end = sub_division_index[data.shape[0]-1+1]
        tmp = sub_division_map[alt_start+1:alt_end]
        data[tmp,28] = (data[tmp,28]/data[data.shape[0]-1,28])*(new_alt_scale)#data[data.shape[0]-1,28])*new_alt_scale
        data[data.shape[0]-1,28] = new_alt_scale
        data[:,21] = data[:,28]*data[0,21]
    return data,sub_division_map,sub_division_index
