import svcco
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from tqdm import tqdm
import tetgen
from scipy import stats
import pymeshfix

cube = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(3)
sphere = pv.Sphere().triangulate().subdivide(1)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build(q=2,resolution=30)
s.pv_polydata_surf.save('cube.vtp')

t = svcco.tree()
t.set_boundary(s)
t.convex = True
t.set_root()
t.n_add(300)

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=6,resolution=85,k=200,buffer=5)
heart.pv_polydata_surf.save('heart.vtp')

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = cyl.compute_normals(auto_orient_normals=True)
c = svcco.surface()
c.set_data(cyl.points,cyl.point_normals)
c.solve()
c.build(q=8,k=200,resolution=100)
c.pv_polydata_surf.save('cylinder.vtp')

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"

gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
gyrus.build(q=4,resolution=40,buffer=5)

###########################################
# Test function
###########################################

def perfusion_volumes(tree,subdivisions=1,tet=None):
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

def test_cube(size=1000,restarts=50):
    t = svcco.tree()
    t.set_boundary(s)
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

def results(test_list,size=1000,restarts=20,bins=50):
    fig,ax = plt.subplots(nrows=2,ncols=2)
    #ax = ax.flatten()
    cw = plt.get_cmap('coolwarm')
    for i in range(len(test_list)):
        vols,sizes = test_list[i](size=size,restarts=restarts)
        idx = i//2
        jdx = i%2
        for j in range(len(vols)):
            color_scale = (sizes[j]/size)*255
            x_min = min(vols[-1])
            x_max = max(vols[-1])
            x_range = np.linspace(x_min,x_max)
            clean_vols = vols[j][vols[j]>0]
            freq,edges = np.histogram(clean_vols,bins=bins)
            centers = 0.5*(edges[1:]+edges[:-1])
            width   = edges[1]-edges[0]
            #if j == 0:
            #    ax[idx][jdx].bar(centers,freq/len(vols[j]),width=width,color='blue')
            #else:
            ax[idx][jdx].bar(centers,freq/len(vols[j]),width=width,alpha=0.4,color=cw(color_scale))
            #ax[idx][jdx].hist(vols[j],bins=50,density=True,alpha=0.25,color=cw(color_scale),range = (x_min,x_max))
            #kde = stats.gaussian_kde(vols[j])
            #ax[idx][jdx].plot(x_range,kde(x_range)/10,color=cw(color_scale),alpha=0.75)
    sm = plt.cm.ScalarMappable(cmap='coolwarm')
    sm.set_array(range(0,size))
    fig.colorbar(sm,ax=ax.ravel().tolist())
    plt.show()
