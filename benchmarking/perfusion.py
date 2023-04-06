import svcco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
from tqdm import tqdm, trange
import tetgen
from scipy import stats
import pymeshfix
import pickle
import os
from scipy.spatial import cKDTree

q = 4
resolution = 120

def get_cube():
    cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
    cube = svcco.surface()
    cube.set_data(cu.points,cu.point_normals)
    cube.solve()
    cube.build(q=q,resolution=60)
    print('cube constructed')
    return cube

def get_heart():
    heart = svcco.surface()
    heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
    heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
    heart.set_data(heart_points,heart_normals)
    heart.solve()
    heart.build(q=q,resolution=60,k=2,buffer=5)
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
    cylinder.build(q=q,resolution=60)
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
    gyrus.build(q=q,resolution=60,buffer=5)
    print('gyrus constructed')
    return gyrus

###########################################
# Test function
###########################################

def perfusion_volumes(tree,subdivisions=0,tet=None):
    terminals = tree.data[tree.data[:,15]<0,:]
    terminals = terminals[terminals[:,16]<0,:]
    territory_id = []
    territory_volumes = np.zeros(len(terminals))
    if tet is None:
        if subdivisions > 0:
            surf = tree.boundary.tet.grid.extract_surface()
            surf = surf.triangulate()
            surf = surf.subdivide(subdivisions)
            surf = svcco.utils.remeshing.remesh.remesh_surface(surf)
            surf = svcco.utils.remeshing.remesh.remesh_surface(surf)
            meshfix = pymeshfix.MeshFix(surf)
            meshfix.repair(verbose=True)
            meshfix.repair(verbose=True)
            surf = meshfix.mesh
            #surf = surf.subdivide(subdivisions)
            #surf = svcco.utils.remeshing.remesh.remesh_surface(surf)
            #surf = svcco.utils.remeshing.remesh.remesh_surface(surf)
            #meshfix = pymeshfix.MeshFix(surf)
            #meshfix.repair(verbose=True)
            #meshfix.repair(verbose=True)
            #surf = meshfix.mesh
            tet  = tetgen.TetGen(surf)
            tet.tetrahedralize()
        else:
            tet = tree.boundary.tet
    vol = tet.grid.compute_cell_sizes().cell_data['Volume']
    for idx, cell_center in tqdm(enumerate(tet.grid.cell_centers().points),desc='Calculating Perfusion Territories'):
        #for idx, cell_center in enumerate(tree.boundary.tet.grid.cell_centers().points):
        closest_terminal = np.argmin(np.linalg.norm(cell_center.reshape(1,-1)-terminals[:,3:6],axis=1))
        territory_id.append(closest_terminal)
        territory_volumes[closest_terminal] += vol[idx]
    territory_id = np.array(territory_id)
    tet.grid['perfusion_territory_id'] = territory_id
    return territory_id,territory_volumes/np.sum(territory_volumes),tet


def preprocess(surf_object):
    surf_object.cell_centers = surf_object.tet.grid.cell_centers().points
    surf_object.cell_center_closest_point = []
    KDTREE = cKDTree(surf_object.tet.grid.points)
    surf_object.TET_KDTREE = KDTREE
    return

def subpoints(pt1,pt2,num):
    t = np.linspace(0,1,num).reshape(-1,1)
    return pt2*t+(1-t)*pt1

def perfusion_simple(tree,perfusion_volume):
    terminals = tree.data[tree.data[:,15]<0,:]
    terminals = terminals[terminals[:,16]<0,:]
    element_volumes = perfusion_volume.tet.grid.compute_cell_sizes().cell_data['Volume']
    terminal_territory_volumes = np.zeros(terminals.shape[0]+1)
    n_cells = perfusion_volume.tet.grid.n_cells
    territory_ids = []
    for idx in trange(n_cells):
        pt1 = perfusion_volume.cell_centers[idx,:]
        distances = np.linalg.norm(terminals[:,3:6] - pt1.reshape(1,-1),axis=1).flatten()
        ordered_terminals = np.argsort(distances).flatten()
        territory_id = None
        perfused = False
        for jdx in ordered_terminals:
            pt2 = terminals[jdx,3:6]
            line_pts = subpoints(pt1,pt2,20)
            next_line = False
            for kdx in range(line_pts.shape[0]):
                if not perfusion_volume.within(line_pts[kdx,0],line_pts[kdx,1],line_pts[kdx,2],2):
                    next_line = True
                    break
            if next_line:
                perfused = False
                continue
            else:
                perfused = True
                territory_id = jdx
                break
        if perfused:
            territory_ids.append(territory_id)
            terminal_territory_volumes[territory_id] += element_volumes[idx]
        else:
            territory_ids.append(-1)
            terminal_territory_volumes[-1] += element_volumes[idx]
    territory_ids = np.array(territory_ids)
    return territory_ids,terminal_territory_volumes/np.sum(element_volumes),perfusion_volume.tet

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
        for jdx in range(len(terminal_ids)):
            #print('getting shortest')
            _,L,_ = perfusion_volume.get_shortest_path(idx,jdx)
            L = sum(L)
            tmp_geodesic_lengths.append(L)
            linear_distances.append(np.linalg.norm(terminals[jdx,3:6] - perfusion_volume.tet.grid.cell_centers().points[idx,:]))
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
    territory_id = np.array(territory_id)
    return territory_id,territory_volumes/np.sum(vol),perfusion_volume.tet


def test(surf_object,size=1000,restarts=50,name='default'):
    t = svcco.tree()
    t.set_boundary(surf_object)
    perfusion_folder = os.getcwd()+os.sep+name+os.sep
    delaunay = surf_object.pv_polydata.delaunay_3d()
    convexity = surf_object.volume/delaunay.volume
    if convexity > 0.95:
        t.convex = True
    t.set_root()
    add_amount = size//restarts
    TERS  = []
    VOLS  = []
    COUNT = []
    _,vols,_ = perfusion_simple(t,surf_object)
    for i in range(restarts):
        t.n_add(add_amount)
        if i == 0:
            ters,vols,tet = perfusion_simple(t,surf_object)
            np.save(perfusion_folder + 'perfusion_volumes_num_terminals_{}.npy'.format(i*add_amount+1),vols)
            np.save(perfusion_folder + 'territories_{}.npy'.format(i*add_amount+1),ters)
            tet.grid.save(perfusion_folder + 'perfusion_mesh.vtu')
        else:
            ters,vols,tet = perfusion_simple(t,surf_object)
            np.save(perfusion_folder + 'perfusion_volumes_num_terminals_{}.npy'.format(i*add_amount+1),vols)
            np.save(perfusion_folder + 'territories_{}.npy'.format(i*add_amount+1),ters)
        VOLS.append(vols)
        TERS.append(ters)
        if i == 0:
            COUNT.append(add_amount)
        else:
            COUNT.append(COUNT[-1]+add_amount)
    return VOLS,COUNT

def test_heart(size=1000,restarts=50):
    t = svcco.tree()
    t.set_boundary(heart)
    t.set_root()
    add_amount = size//restarts
    VOLS  = []
    COUNT = []
    _,vols,_ = perfusion_volumes(t)
    #VOLS.append(vols)
    #COUNT.append(1)
    for i in range(restarts):
        t.n_add(add_amount)
        if i == 0:
            _,vols,tet = perfusion_volumes(t)
        else:
            _,vols,tet = perfusion_volumes(t,tet=tet)
        VOLS.append(vols)
        if i == 0:
            COUNT.append(add_amount)
        else:
            COUNT.append(COUNT[-1]+add_amount)
    return VOLS,COUNT

def test_cylinder(size=1000,restarts=50):
    t = svcco.tree()
    t.set_boundary(c)
    t.convex = True
    t.set_root()
    add_amount = size//restarts
    VOLS  = []
    COUNT = []
    _,vols,_ = perfusion_volumes(t)
    #VOLS.append(vols)
    #COUNT.append(1)
    for i in range(restarts):
        t.n_add(add_amount)
        if i == 0:
            _,vols,tet = perfusion_volumes(t)
        else:
            _,vols,tet = perfusion_volumes(t,tet=tet)
        VOLS.append(vols)
        if i == 0:
            COUNT.append(add_amount)
        else:
            COUNT.append(COUNT[-1]+add_amount)
    return VOLS,COUNT

def test_gyrus(size=1000,restarts=50):
    t = svcco.tree()
    t.set_boundary(gyrus)
    t.set_root()
    add_amount = size//restarts
    VOLS  = []
    COUNT = []
    _,vols,_ = perfusion_volumes(t)
    #VOLS.append(vols)
    #COUNT.append(1)
    for i in range(restarts):
        t.n_add(add_amount)
        if i == 0:
            _,vols,tet = perfusion_volumes(t)
        else:
            _,vols,tet = perfusion_volumes(t,tet=tet)
        VOLS.append(vols)
        if i == 0:
            COUNT.append(add_amount)
        else:
            COUNT.append(COUNT[-1]+add_amount)
    return VOLS,COUNT

def results(test_list,size=100,restarts=100,bins=50,repeat=1):
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



def get_hist_data(data,hist_range=(0,1),bins=50):
    data['frequency'] = []
    data['centers']   = []
    data['widths']    = []
    for idx in range(len(data['volumes'])):
        temp_freq  = []
        temp_cent  = []
        temp_width = []
        for jdx in range(len(data['volumes'][idx])):
            vols = data['volumes'][idx][jdx]
            clean_vols = vols[vols>0]
            freq,edges = np.histogram(clean_vols,range=hist_range,bins=bins)
            centers = 0.5*(edges[1:]+edges[:-1])
            temp_freq.append(freq/len(clean_vols))
            temp_cent.append(centers)
            temp_width.append(edges[1]-edges[0])
        data['frequency'].append(temp_freq)
        data['centers'].append(temp_cent)
        data['widths'].append(temp_width)
    return data

#f,a,data = results([cube,cylinder,heart,gyrus],repeat=10)
#file = open('perfusion_data.pkl','wb+')
#pickle.dump(data,file)
#file.close()

"""
cube = get_cube()
preprocess(cube)
results = test(cube,size=50,restarts=50,name='cube')
print('done')

heart = get_heart()
preprocess(heart)
results = test(heart,size=50,restarts=50,name='heart')
print('done')

disk = get_disk()
preprocess(disk)
results = test(disk,size=50,restarts=50,name='disk')
print('done')

gyrus = get_gyrus()
preprocess(gyrus)
results = test(gyrus,size=50,restarts=50,name='gyrus')
print('done')

"""
