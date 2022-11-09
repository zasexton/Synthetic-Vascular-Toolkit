# File for perfusion calculation through geodesics for figure 4c
# This code calculates the normalized volume of perfusion territories
# for convex and non-convex perfusion domains

# the goal of this code is to highlight that non-convex domains require
# more extensive vascular networks to thoroughly perfuse a given space


import numba as nb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import svcco
import pyvista as pv
from tqdm import trange, tqdm
import numpy as np
import tetgen
from scipy import stats
import pymeshfix
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from tempfile import TemporaryDirectory
import os
from multiprocessing import Pool
import psutil

def get_all_paths(i,graph):
    ds, Pr = shortest_path(csgraph=graph,directed=False,indices=i,return_predecessors=True)
    return ds, Pr

def parallel_write(pkg):
    start   = pkg[0]
    stop    = pkg[1]
    tempdir = pkg[2]
    graph   = pkg[3]
    position = pkg[4]
    graph_range = list(range(start,stop))
    file_ds = np.memmap(tempdir+os.sep+'ds_{}-{}.dat'.format(start,stop),dtype=np.float64,mode='w+',shape=(len(graph_range),graph.shape[1]))
    file_Pr = np.memmap(tempdir+os.sep+'Pr_{}-{}.dat'.format(start,stop),dtype=np.int64,mode='w+',shape=(len(graph_range),graph.shape[1]))
    #tqdm(range(len(data['flow'])),desc="Building Vessel Data",position=1,leave=False)
    #graph_range = list(range(start,stop))
    checks = np.linspace(0,len(graph_range),10,dtype=int)
    for i in range(len(graph_range)):
        if i in checks:
            print("Dijkstra Solve | {} - {}:  {}%".format(start,stop,i/len(graph_range)*100))
        ds,Pr = get_all_paths(graph_range[i],graph)
        file_ds[i,:] = ds
        file_Pr[i,:] = Pr
        file_ds.flush()
        file_Pr.flush()
        del ds
        del Pr
    print("Dijkstra Solve | {} - {}:  {}%".format(start,stop,100))
    del file_ds
    del file_Pr
    return

def determine_blocks(row_number,block_number):
    base = row_number//block_number
    remain = row_number%block_number
    blocks = []
    count  = 0
    for i in range(block_number):
        tmp = [count]
        tmp.append(count+base)
        if i == block_number - 1:
            tmp[-1] += remain
        blocks.append(tmp)
        count += base
    return blocks

def build_work(blocks,tmpdir,graph):
    work = []
    for i,b in enumerate(blocks):
        b.append(tmpdir.name)
        b.append(graph)
        b.append(i)
        work.append(b)
    work = tuple(work)
    return work

def precompute_graph_predecessors(surface_object,block_num=10,parallel_num=10,mem_limit=2e9,mem_buffer=1e9,file_dir='D:'+os.sep):
    pid = os.getpid()
    process = psutil.Process(pid)
    available_mem = psutil.virtual_memory()[1]
    if available_mem-mem_buffer <= mem_limit:
        print('Process does not have enough available memory for given memory limit of {} Gb'.format(mem_limit/1e9))
        return None,None,None
    cpu_num = psutil.cpu_count()
    if parallel_num > cpu_num - 1:
        print('Process does not have enough available memory for given cpu core count {}'.format(cpu_num-1))
        print('Number of cores requested: {}'.format(parallel_num))
        return None,None,None
    disk_info = psutil.disk_usage(file_dir)
    estimated_disk_use = surface_object.graph.shape[0]**2*8
    if disk_info[2] < estimated_disk_use:
        print('File Directory does not have enough storage space with {} Gb available'.format(disk_info[2]/1e9))
        print('Estimated storage required: {}'.format(estimated_disk_use))
    tmp_dir   = TemporaryDirectory(dir=file_dir)
    surface_object.tmp_dir_dijkstra = tmp_dir
    surface_object.cell_centers = surface_object.tet.grid.cell_centers().points
    surface_object.cell_center_closest_point = []
    on_ram = (surface_object.graph.shape[1]*(surface_object.graph.shape[0]//block_num))*8*parallel_num
    while on_ram > mem_limit:
        block_num += 1
        on_ram = (surface_object.graph.shape[1]*(surface_object.graph.shape[0]//block_num))*8*parallel_num
    blocks = determine_blocks(surface_object.graph.shape[0],block_num)
    base = blocks[0][1] - blocks[0][0]
    work = build_work(blocks,tmp_dir,surface_object.graph)
    p = Pool(parallel_num)
    p.map(parallel_write,work)
    p.close()
    ds = {'ranges':[],'files':[],'mats':[]}
    Pr = {'ranges':[],'files':[],'mats':[]}
    for i in range(len(work)):
         ds['ranges'].append(work[i][0:2])
         ds['files'].append(tmp_dir.name+os.sep+'ds_{}-{}.dat'.format(work[i][0],work[i][1]))
         ds['mats'].append(np.memmap(tmp_dir.name+os.sep+'ds_{}-{}.dat'.format(work[i][0],
                                     work[i][1]),mode='r',dtype=np.float64,
                                     shape=(work[i][1]-work[i][0],surface_object.graph.shape[1])))
         Pr['ranges'].append(work[i][0:2])
         Pr['files'].append(tmp_dir.name+os.sep+'Pr_{}-{}.dat'.format(work[i][0],work[i][1]))
         Pr['mats'].append(np.memmap(tmp_dir.name+os.sep+'Pr_{}-{}.dat'.format(work[i][0],
                                     work[i][1]),mode='r',dtype=np.int64,
                                     shape=(work[i][1]-work[i][0],surface_object.graph.shape[1])))
    surface_object.ds = ds
    surface_object.Pr = Pr
    tet_vert_map = []
    for i in trange(surface_object.tet.grid.n_cells):
        surface_object.cell_center_closest_point.append(surface_object.tet.grid.find_closest_point(surface_object.cell_centers[i,:]))
        if surface_object.cell_center_closest_point[-1] < blocks[-1][0]:
            tmp = [surface_object.cell_center_closest_point[-1]//base,surface_object.cell_center_closest_point[-1]%base]
        else:
            tmp = [len(blocks)-1,surface_object.cell_center_closest_point[-1] - blocks[-1][0]]
        tet_vert_map.append(tmp)
    surface_object.tet_vert_map = tet_vert_map
    return

@nb.jit(nopython=True)
def get_path(j,ds,Pr):
    path = [j]
    lengths = []
    k = j
    while Pr[k] != -9999:
        path.append(Pr[k])
        lengths.append(ds[k])
        k = Pr[k]
    path = path[::-1]
    lengths = lengths[::-1]
    lines = []
    for jdx in range(len(path)-1):
        lines.append([path[jdx],path[jdx+1]])
    return path,lengths,lines

def perfusion_geodesic(tree,perfusion_volume):
    terminals = tree.data[tree.data[:,15]<0,:]
    terminals = terminals[terminals[:,16]<0,:]
    terminal_ids = []
    vol = perfusion_volume.tet.grid.compute_cell_sizes().cell_data['Volume']
    for idx in range(terminals.shape[0]):
        terminal_ids.append(perfusion_volume.tet.grid.find_closest_point(terminals[idx,3:6]))
    territory_id = []
    territory_volumes = np.zeros(terminals.shape[0])
    territory_min_path_distance = []
    for idx in tqdm(range(perfusion_volume.tet.grid.n_cells),desc='Calculating Perfusion Territories'):
        tmp_geodesic_lengths = []
        linear_distances     = []
        center = perfusion_volume.cell_centers[idx,:] #perfusion_volume.tet.grid.cell_centers().points[idx,:]
        cell_id = perfusion_volume.cell_center_closest_point[idx] #perfusion_volume.tet.grid.find_closest_point(center)
        for jdx in range(len(terminal_ids)):
            #tmp_ds = perfusion_volume.graph_predecessor_distan[cell_id]
            #tmp_Pr = perfusion_volume.graph_predecessor_id[cell_id]
            tmp_ds = np.array(perfusion_volume.ds['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
            tmp_Pr = np.array(perfusion_volume.Pr['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
            _,L,_ = get_path(terminal_ids[jdx],tmp_ds,tmp_Pr)
            L = sum(L)
            tmp_geodesic_lengths.append(L)
            linear_distances.append(np.linalg.norm(terminals[jdx,3:6] - center))
        tmp_geodesic_lengths = np.array(tmp_geodesic_lengths)
        min_geodesic_value = np.min(tmp_geodesic_lengths)
        min_geodesic_instances = np.where(tmp_geodesic_lengths==min_geodesic_value)
        linear_distances = np.array(linear_distances)
        if len(min_geodesic_instances) == 1:
            territory_id.append(min_geodesic_instances[0])
            territory_volumes[min_geodesic_instances[0]] += vol[idx]
            territory_min_path_distance.append(tmp_geodesic_lengths[min_geodesic_instances[0]])
        else:
            absolute_min = np.argwhere(linear_distances[min_geodesic_instances])
            absolute_min = min_geodesic_instances[absolute_min]
            territory_min_path_distance.append(tmp_geodesic_lengths[absolute_min])
            territory_id.append(absolute_min)
            territory_volumes[absolute_min] += vol[idx]
    territory_id = np.array(territory_id)
    territory_min_path_distance = np.array(territory_min_path_distance)
    perfusion_volume.tet.grid['perfusion_territory_id'] = territory_id
    return territory_id,territory_volumes/np.sum(vol),perfusion_volume.tet,vol,territory_min_path_distance

def perfusion_efficacy(tree_object,perfusion_object,flow_per_mass=0.8,density=1):
    """
    Calculate the perfusion efficiency over a perfusion domain
    parameters
    ----------
               tree_object: svcco.tree object
                            tree object generated through
                            svcco to product a vascular network

               perfusion_object: svcco.implicit.implicit
                                implicit volume generated through
                                svcco to represent the perfusion
                                domain

               flow_per_mass: float
                        target flow rate per mass unit of tissue
                        default units: 0.8 mL/(min * g)
                        value is given for myocardial tissue
                        from source:
                        https://academic.oup.com/bjaed/article/5/2/61/422091
    return
    ------
               mass_at_risk: float
                            mass, in grams, of tissue that falls below the
                            perscribed target flow per mass value
               volume_at_risk: float
                            volume, in mL, of tissue that falls below the
                            perscribed target flow per mass value
    """
    volume_at_risk         = 0
    mass_at_risk           = 0
    total_volume           = perfusion_object.volume
    terminal_flow          = tree_object.parameters['Qterm']*60 # mL/min
    territory_id,volumes,_,vols,paths = perfusion_geodesic(tree_object,perfusion_object)
    volumes                = total_volume*volumes
    mass                   = volumes*density
    territories            = list(range(int(max(territory_id.flatten()))+1))
    risk                   = np.zeros(territory_id.flatten().shape[0])
    for terr in territories:
        m_ids    = np.argwhere(territory_id.flatten()==terr).flatten()
        m_values = vols[m_ids]*density
        p_values = paths.flatten()[m_ids]
        m_ids_sort = np.argsort(p_values).flatten()
        #target = sum(m_values)*flow_per_mass
        target = mass[terr]*flow_per_mass
        perfused_mass = terminal_flow/flow_per_mass
        tmp_mass = 0
        for idx in m_ids_sort:
            m_vol = m_values[idx]
            if m_vol+tmp_mass < perfused_mass:
                tmp_mass += m_vol
            else:
                risk[m_ids[idx]] = 1
        m = mass[terr]
        if target > terminal_flow:
            mass_at_risk += mass[terr] - perfused_mass
    volume_at_risk         = mass_at_risk/density
    perfusion_object.tet.grid['risk'] = risk
    return mass_at_risk,volume_at_risk

q = 4
resolution = 25


"""
heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
"""
