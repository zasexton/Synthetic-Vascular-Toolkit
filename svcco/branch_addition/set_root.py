import numpy as np
from .basis import *
from .calculate_length import *
from .update import *
from .calculate_radii import *

def set_root(data, boundary, Qterm, gamma, nu,
             Pperm, Pterm, fraction,
             ishomogeneous,isconvex,isdirected,
             start,direction,limit_high,limit_low,volume,
             low=-1,high=0,niter=200):
    """
    Explaination of code
    """
    p0,layer = boundary.pick()
    distance = boundary.volume**(1/3)
    steps = distance//fraction
    path = []
    if not ishomogeneous:
        path,lengths,layers = boundary.path(p0,distance,steps,dive=0)
        p1 = path[-1]
        path = np.array(path)
        data = np.zeros((len(path),data.shape[1]))
    elif start is None and direction is None:
        lengths = [0]
        p1,_ = boundary.pick(homogeneous=True)
        p1 = p1[0]
        if not isconvex:
            attempts = 0
            threshold = (volume)**(1/3)
            max_attempts = 10
            cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,threshold+threshold*0.5))
            cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,threshold))
            cell_list = list(cell_list_outter.difference(cell_list_inner))
            while np.sum(lengths) < threshold:
                if attempts > max_attempts or len(cell_list)==0:
                    attempts = 0
                    threshold = threshold*0.9
                    #print(threshold)
                    cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,threshold+threshold*0.5))
                    cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,threshold))
                    cell_list = list(cell_list_outter.difference(cell_list_inner))
                    ball_size = threshold*1.1
                    while len(cell_list) < 1:
                        cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,ball_size+ball_size*0.5))
                        cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,ball_size))
                        if threshold < ((volume)**(1/3)*(0.75**5)):
                            cell_list = list(cell_list_outter)
                            max_attmpts = min(len(cell_list),1000)
                        else:
                            cell_list = list(cell_list_outter.difference(cell_list_inner))
                        ball_size *= 1.1
                #p1,_ = boundary.pick(homogeneous=True)
                cell_id = cell_list.pop(0)
                p1,_ = boundary.pick_in_cell(cell_id)
                lengths = [np.linalg.norm(p1-p0)]
                subdivisions = 8
                p1 = p1[0]
                within = True
                for sub in range(1,subdivisions):
                    p_tmp = (p1 - p0)*(sub/subdivisions) + p0
                    if not boundary.within(p_tmp[0],p_tmp[1],p_tmp[2],2):
                        within = False
                        break
                if not within:
                    lengths = [0]
                    attempts += 1
            #threshold = (volume)**(1/3)
            #while not within:
                #boundary.cell_lookup.query_ball_point(p1)
            """
            if not within:
                print('geodesic')
                #i = boundary.tet.grid.find_closest_point(p0)
                #j = boundary.tet.grid.find_closest_point(p1)
                #path,lengths,lines = boundary.get_shortest_path(i,j)
                #path = boundary.tet_pts[path]
                #lengths[-1] = np.linalg.norm(p1-path[-1])
                #lengths[0] = np.linalg.norm(p0-path[0])
                #path = path.tolist()
                #path[0] = p0.tolist()
                #path[-1] = p1.tolist()
                #path = np.array(path)
                path,lengths,res,pf,f,rm = boundary.find_best_path(p0,p1,niter=niter)
                path = np.array(path)
                path,lengths = boundary.resample(path)
                data = np.zeros((len(path)-1,data.shape[1]))
            """
        else:
            while np.sum(lengths) < (volume)**(1/3):
                p1,_ = boundary.pick(homogeneous=True)
                lengths = [np.linalg.norm(p1-p0)]
    else:
        p0 = start
        if limit_high is not None and limit_low is not None:
            distance = limit_high - limit_low
            required_length = limit_low
        elif limit_high is not None and limit_low is None:
            distance = length_high
            required_length = 0
        else:
            distance = volume**(1/3)
            required_length = 0.5*distance
        if direction is None:
            if not boundary.within(start[0],start[1],start[2],2):
                #data = np.zeros((2,data.shape[1]))
                closest_cell = boundary.pv_polydata_surf.find_closest_cell(start)
                faces = boundary.pv_polydata_surf.faces.reshape(len(boundary.pv_polydata_surf.faces)//4,4)
                indices = faces[closest_cell,1:]
                cell_points = boundary.pv_polydata_surf.points[indices]
                #cell_points = np.sum(cell_points,axis=0)/3
                #p1,_ = boundary.pick_in_cell(closest_cell)
                #p1 = p1[0]
                p1 = np.sum(cell_points,axis=0)/3
                """
                lengths = [np.linalg.norm(p1-p0)]
                data[0, 0:3] = p0
                data[0, 3:6] = p1
                basis(data,0)
                data[0, 15] = 1
                data[0, 16] = -1
                data[0, 17] = -1
                data[0, 18] = 0
                data[0, 19] = 1
                data[0, 20] = np.sum(lengths)
                data[0, 22] = Qterm
                data[0, 26] = 0
                data[0, 28] = 1.0
                data[0, 29] = -1
                data[0, -1] = 0
                sub_division_map = [-1]
                sub_division_index = [0]
                update(data, gamma, nu)
                radii(data,Pperm,Pterm)
                """
                p0 = p1 #.flatten()
                lengths = [0]
                #p1,_ = boundary.pick(homogeneous=True)
                #lengths = [np.linalg.norm(p1-p0)]
                if not isconvex:
                    attempts = 0
                    threshold = (volume)**(1/3)
                    max_attempts = 10
                    cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,threshold+threshold*0.5))
                    cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,threshold))
                    cell_list = list(cell_list_outter.difference(cell_list_inner))
                    while np.sum(lengths) < threshold:
                        if attempts > max_attempts or len(cell_list)==0:
                            attempts = 0
                            threshold = threshold*0.9
                            #print(threshold)
                            cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,threshold+threshold*0.5))
                            cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,threshold))
                            cell_list = list(cell_list_outter.difference(cell_list_inner))
                            ball_size = threshold*1.1
                            while len(cell_list) < 1:
                                cell_list_outter = set(boundary.cell_lookup.query_ball_point(p0,ball_size+ball_size*0.5))
                                cell_list_inner  = set(boundary.cell_lookup.query_ball_point(p0,ball_size))
                                if threshold < ((volume)**(1/3)*(0.75**5)):
                                    cell_list = list(cell_list_outter)
                                    max_attmpts = min(len(cell_list),1000)
                                else:
                                    cell_list = list(cell_list_outter.difference(cell_list_inner))
                                ball_size *= 1.1
                        #p1,_ = boundary.pick(homogeneous=True)
                        cell_id = cell_list.pop(0)
                        p1,_ = boundary.pick_in_cell(cell_id)
                        lengths = [np.linalg.norm(p1-p0)]
                        subdivisions = 8
                        p1 = p1[0]
                        within = True
                        for sub in range(1,subdivisions):
                            p_tmp = (p1 - p0)*(sub/subdivisions) + p0
                            if not boundary.within(p_tmp[0],p_tmp[1],p_tmp[2],2):
                                within = False
                                break
                        if not within:
                            lengths = [0]
                            attempts += 1
                else:
                    while np.sum(lengths) < (volume)**(1/3):
                        p1,_ = boundary.pick(homogeneous=True)
                        lengths = [np.linalg.norm(p1-p0)]
                lengths = [np.linalg.norm(start-p0)]
                data[0, 0:3] = start
                data[0, 3:6] = p1
                basis(data,0)
                data[0, 15] = -1
                data[0, 16] = -1
                data[0, 17] = -1
                data[0, 18] = 0
                data[0, 19] = 1
                data[0, 20] = np.sum(lengths)
                data[0, 22] = Qterm
                data[0, 26] = 0
                data[0, 28] = 1.0
                data[0, 29] = -1
                data[0, -1] = 0
                sub_division_map = [-1]
                sub_division_index = [0]
                update(data, gamma, nu)
                radii(data,Pperm,Pterm)
                return data,sub_division_map,sub_division_index
            else:
                p1,_ = boundary.pick(homogeneous=True)
                lengths = [np.linalg.norm(p1-p0)]

        else:
            p1_tmp = start + direction*required_length + direction*distance*np.random.random(1)
            while not boundary.within(p1_tmp[0],p1_tmp[1],p1_tmp[2],2):
                p1_tmp = start + direction*distance*np.random.random(1)
            p1 = p1_tmp
            lengths = [np.linalg.norm(p1-p0)]
    if len(path) == 0:
        #print('linear root')
        data[0, 0:3] = p0
        data[0, 3:6] = p1
        basis(data,0)
        data[0, 15] = -1
        data[0, 16] = -1
        data[0, 17] = -1
        data[0, 18] = 0
        data[0, 19] = 1
        data[0, 20] = np.sum(lengths)
        data[0, 22] = Qterm
        data[0, 26] = 0
        data[0, 28] = 1.0
        data[0, 29] = -1
        data[0, -1] = 0
        sub_division_map = [-1]
        sub_division_index = [0]
        update(data, gamma, nu)
        radii(data,Pperm,Pterm)
    else:
        sub_division_map = []
        sub_division_index = []
        for i in range(data.shape[0]):
            #if i == 1:
            #    data[i,0:3] = p0
            #    data[i,3:6] = path[i]
            #elif i == data.shape[0]:
            #    data[i,0:3] = path[-1]
            #    data[i,3:6] = p1
            #else:
            #    data[i,0:3] = path[i-1]
            #    data[i,3:6] = path[i]
            data[i,0:3] = path[i]
            data[i,3:6] = path[i+1]
            basis(data,i)
            if i < data.shape[0] - 1:
                data[i,15] = i+1
            else:
                data[i,15] = -1
            data[i,16] = -1
            data[i,17] = i - 1
            data[i,18] = i
            data[i,19] = i+1
            data[i,20] = lengths[i] #length(data,i)
            data[i,22] = Qterm
            data[i,23] = 1.0
            data[i,24] = 0
            data[i,25] = (8*nu/np.pi)*np.sum(lengths[i:])
            data[i,26] = 0
            data[i,27] = np.sum(lengths[i:])
            data[i,28] = 1.0
            data[i,29] = -1
            data[i,-1] = i
            data[i,21] = ((data[0,25]*data[0,22])/(Pperm-Pterm)) ** (1/4)
            sub_division = [-1]
            #sub_division_index.append(len(sub_division_map))
            if i < data.shape[0] - 1:
                sub_division.extend(list(range(i+1,data.shape[0])))
            sub_division_map.extend(sub_division)
            #print(sub_division_map)
    sub_division_index = np.argwhere(np.array(sub_division_map)==-1).flatten().tolist()
    return data,sub_division_map,sub_division_index
