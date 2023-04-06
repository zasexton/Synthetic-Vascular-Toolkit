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

q = 4
resolution = 120

def get_cube():
    cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
    cube = svcco.surface()
    cube.set_data(cu.points,cu.point_normals)
    cube.solve()
    cube.build(q=4,resolution=25)
    print('cube constructed')
    return cube

def get_heart():
    heart = svcco.surface()
    heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
    heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
    heart.set_data(heart_points,heart_normals)
    heart.solve()
    heart.build(q=4,resolution=60,k=2,buffer=5)
    print('heart constructed')
    return heart

def get_disk():
    disk = pv.Disc(inner=2.8,outer=4,r_res=20,c_res=100)
    cyl  = disk.extrude([0,0,2],capping=True).triangulate()
    cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
    cyl  = cyl.subdivide(2)
    cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
    cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
    cylinder = svcco.surface()
    cylinder.set_data(cyl.points,cyl.point_normals)
    cylinder.solve()
    cylinder.build(q=q,resolution=resolution)
    print('cylinder constructed')
    return cylinder

def get_gyrus():
    left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
    gyrus_no_scale = pv.read(left_gyrus)
    heart = get_heart()
    sf = (heart.volume/gyrus_no_scale.volume)**(1/3)
    gyrus_scaled = gyrus_no_scale.scale([sf,sf,sf])
    left_gyrus_scaled = "left_gyrus_scaled.vtp"
    gyrus_scaled.save(left_gyrus_scaled)
    gyrus = svcco.surface()
    gyrus.load(left_gyrus_scaled)
    gyrus.solve()
    gyrus.build(q=q,resolution=resolution,buffer=5)
    print('gyrus constructed')
    return gyrus

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

def precompute_graph_predecessors(surface_object,block_num=10,parallel_num=10):
    file_dir  = 'D:'+os.sep
    tmp_dir   = TemporaryDirectory(dir=file_dir)
    #print(tmp_dir.name)
    surface_object.tmp_dir_dijkstra = tmp_dir
    #surface_object.graph_predecessor_distance = np.memmap(tmp_dir.name + os.sep + 'ds.dat',dtype='int32',mode='w+',shape=surface_object.graph.shape)
    #surface_object.graph_predecessor_id       = np.memmap(tmp_dir.name + os.sep + 'Pr.dat',dtype='int32',mode='w+',shape=surface_object.graph.shape)
    surface_object.cell_centers = surface_object.tet.grid.cell_centers().points
    surface_object.cell_center_closest_point = []
    #for i in trange(surface_object.tet.grid.n_points):
    #    ds,Pr = get_all_paths(i,surface_object.graph)
    #    surface_object.graph_predecessor_distance[i,:] = ds
    #    surface_object.graph_predecessor_id[i,:]       = Pr
    #    del ds
    #    del Pr
    blocks = determine_blocks(surface_object.graph.shape[0],block_num)
    base = blocks[0][1] - blocks[0][0]
    work = build_work(blocks,tmp_dir,surface_object.graph)
    p = Pool(parallel_num)
    p.map(parallel_write,work)
    ds = {'ranges':[],'files':[],'mats':[]}
    Pr = {'ranges':[],'files':[],'mats':[]}
    for i in range(len(work)):
         ds['ranges'].append(work[i][0:2])
         ds['files'].append(tmp_dir.name+os.sep+'ds_{}-{}.dat'.format(work[i][0],work[i][1]))
         ds['mats'].append(np.memmap(tmp_dir.name+os.sep+'ds_{}-{}.dat'.format(work[i][0],work[i][1]),mode='r',dtype=np.float64,shape=(work[i][1]-work[i][0],surface_object.graph.shape[1])))
         Pr['ranges'].append(work[i][0:2])
         Pr['files'].append(tmp_dir.name+os.sep+'Pr_{}-{}.dat'.format(work[i][0],work[i][1]))
         Pr['mats'].append(np.memmap(tmp_dir.name+os.sep+'Pr_{}-{}.dat'.format(work[i][0],work[i][1]),mode='r',dtype=np.int64,shape=(work[i][1]-work[i][0],surface_object.graph.shape[1])))
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
    p.close()
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
    for idx in tqdm(range(perfusion_volume.tet.grid.n_cells),desc='Calculating Perfusion Territories'):
        tmp_geodesic_lengths = []
        linear_distances     = []
        center = perfusion_volume.cell_centers[idx,:] #perfusion_volume.tet.grid.cell_centers().points[idx,:]
        cell_id = perfusion_volume.cell_center_closest_point[idx] #perfusion_volume.tet.grid.find_closest_point(center)
        tmp_ds = np.array(perfusion_volume.ds['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
        tmp_Pr = np.array(perfusion_volume.Pr['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
        for jdx in range(len(terminal_ids)):
            #tmp_ds = perfusion_volume.graph_predecessor_distan[cell_id]
            #tmp_Pr = perfusion_volume.graph_predecessor_id[cell_id]
            #tmp_ds = np.array(perfusion_volume.ds['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
            #tmp_Pr = np.array(perfusion_volume.Pr['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:])
            #tmp_ds = perfusion_volume.ds['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:]
            #tmp_Pr = perfusion_volume.Pr['mats'][perfusion_volume.tet_vert_map[idx][0]][perfusion_volume.tet_vert_map[idx][1],:]
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
        else:
            absolute_min = np.argwhere(linear_distances[min_geodesic_instances])
            absolute_min = min_geodesic_instances[absolute_min]
            territory_id.append(absolute_min)
            territory_volumes[absolute_min] += vol[idx]
        del tmp_ds
        del tmp_Pr
    territory_id = np.array(territory_id)
    perfusion_volume.tet.grid['perfusion_territory_id'] = territory_id
    return territory_id,territory_volumes/np.sum(vol),perfusion_volume.tet

def test(surf_object,size=10,restarts=10,name='default'):
    t = svcco.tree()
    t.set_boundary(surf_object)
    perfusion_folder = os.getcwd()+os.sep+name+os.sep
    delaunay = surf_object.pv_polydata.delaunay_3d()
    convexity = surf_object.volume/delaunay.volume
    if convexity > 0.95:
        t.convex = True
    t.set_root()
    add_amount = size//restarts
    VOLS  = []
    COUNT = []
    _,vols,_ = perfusion_geodesic(t,surf_object)
    for i in range(restarts):
        t.n_add(add_amount)
        if i == 0:
            _,vols,tet = perfusion_geodesic(t,surf_object)
            np.save(perfusion_folder + 'perfusion_volumes_num_terminals_{}.npy'.format(i*add_amount+1),vols)
        else:
            _,vols,tet = perfusion_geodesic(t,surf_object)
            np.save(perfusion_folder + 'perfusion_volumes_num_terminals_{}.npy'.format(i*add_amount+1),vols)
        VOLS.append(vols)
        if i == 0:
            COUNT.append(add_amount)
        else:
            COUNT.append(COUNT[-1]+add_amount)
    return VOLS,COUNT

def results(test_list,size=10,restarts=10,bins=50,repeat=1):
    fig,ax = plt.subplots(nrows=2,ncols=2)
    #ax = ax.flatten()
    #cw = plt.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=0, vmax=size+1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    cmap.set_array([])
    DATA = {'volumes':[],'tree_size':[]}
    for i in range(len(test_list)):
        vols,sizes = test(test_list[i],size=size,restarts=restarts)
        DATA['volumes'].append(vols)
        DATA['tree_size'].append(sizes)
        DATA['repeat'] = repeat
        idx = i//2
        jdx = i%2
        for j in range(len(vols)):
            #color_scale = (sizes[j]/size)*255
            x_min = min(vols[-1])
            x_max = max(vols[-1])
            x_range = np.linspace(x_min,x_max)
            clean_vols = vols[j][vols[j]>0]
            freq,edges = np.histogram(clean_vols,range=(0,1),bins=bins)
            centers = 0.5*(edges[1:]+edges[:-1])
            width   = edges[1]-edges[0]
            #if j == 0:
            #    ax[idx][jdx].bar(centers,freq/len(vols[j]),width=width,color='blue')
            #else:
            ax[idx][jdx].bar(centers,freq/len(clean_vols),width=width,alpha=0.4,color=cmap.to_rgba(len(vols[j])))
            #ax[idx][jdx].hist(vols[j],bins=50,density=True,alpha=0.25,color=cw(color_scale),range = (x_min,x_max))
            #kde = stats.gaussian_kde(vols[j])
            #ax[idx][jdx].plot(x_range,kde(x_range)/10,color=cw(color_scale),alpha=0.75)
    #sm = plt.cm.ScalarMappable(cmap='coolwarm')
    #sm.set_array(range(0,size))
    if repeat > 1:
        for j in range(len(test_list)):
            for i in range(len(DATA['volumes'][j])):
                DATA['volumes'][j][i] = DATA['volumes'][j][i].tolist()
        for j in range(repeat-1):
            for i in range(len(test_list)):
                vols,sizes = test(test_list[i],size=size,restarts=restarts)
                for k in range(len(vols)):
                    DATA['volumes'][i][k].extend(vols[k])
    fig.colorbar(cmap,ax=ax.ravel().tolist(),label="Number of Terminals")
    fig.savefig('perfusion-{}_num_vessels-{}_restarts-{}_num_bins-{}.svg'.format(len(test_list),size,restarts,bins),format='svg')
    return fig,ax,DATA

"""
q = 4
resolution = 20 #120

cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve(quiet=False)
cube.build(q=q,resolution=resolution,verbose=True)
print('preprocessing cells')

blocks = determine_blocks(cube.graph.shape[0],10)
tmpdir = TemporaryDirectory(dir='D:'+os.sep)
work = build_work(blocks,tmpdir,cube.graph)



precompute_graph_predecessors(cube)
print('cube constructed')

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve(quiet=False)
heart.build(q=4,resolution=120,k=2,buffer=5,verbose=True)
print('preprocessing cells')
precompute_graph_predecessors(heart)
print('heart constructed')

heart = get_heart()
precompute_graph_predecessors(heart,block_num=100,parallel_num=16)
results = test(heart,size=50,restarts=50,name='heart')

disk = pv.Disc(inner=2.8,outer=4,r_res=20,c_res=100)
cyl  = disk.extrude([0,0,2],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
cylinder = svcco.surface()
cylinder.set_data(cyl.points,cyl.point_normals)
cylinder.solve(quiet=False)
cylinder.build(q=q,resolution=resolution,verbose=True)
print('preprocessing cells')
precompute_graph_predecessors(cylinder)
print('cylinder constructed')

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
gyrus_no_scale = pv.read(left_gyrus)
sf = (heart.volume/gyrus_no_scale.volume)**(1/3)
gyrus_scaled = gyrus_no_scale.scale([sf,sf,sf])
left_gyrus_scaled = "left_gyrus_scaled.vtp"
gyrus_scaled.save(left_gyrus_scaled)
gyrus = svcco.surface()
gyrus.load(left_gyrus_scaled)
gyrus.solve(quiet=False)
gyrus.build(q=q,resolution=resolution,buffer=5,verbose=True)
print('preprocessing cells')
precompute_graph_predecessors(gyrus)
print('gyrus constructed')

f,a,data = results([cube,cylinder,heart,gyrus],size=50,restarts=50,repeat=10)
file = open('perfusion_fast_data.pkl','wb+')
pickle.dump(data,file)
file.close()
"""
