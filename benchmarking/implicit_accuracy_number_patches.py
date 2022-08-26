# Code for accuracy of the implicit solution
import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import tqdm
###########################################
# Code for Building Surface
###########################################

cu = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build(q=4,resolution=120)
print('cube constructed')

heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=4,resolution=120,k=2,buffer=5)
print('heart constructed')

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl,hausd=0.005)
cyl  = cyl.compute_normals(auto_orient_normals=True,feature_angle=90)
cylinder = svcco.surface()
cylinder.set_data(cyl.points,cyl.point_normals)
cylinder.solve()
cylinder.build(q=4,k=2,resolution=120)
print('cylinder constructed')

left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"
gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
gyrus.build(q=4,k=2,resolution=120,buffer=5)
print('gyrus constructed')

###########################################
# Code for evaluation accuracy
###########################################

def place_bin(value,bin_num,bin_range):
    ind = None
    BINS = np.linspace(bin_range[0],bin_range[1],bin_num+1)
    for i in range(len(BINS)-1):
        if value > BINS[i] and value < BINS[i+1]:
            ind = i
            break
    return ind

def get_values(s,value_min,value_max,reps,samples):
    count = 0
    values  = []
    patches = []
    pbar = tqdm(total=reps)
    while count < reps:
        xb = s.x_range[1] - s.x_range[0]
        yb = s.y_range[1] - s.y_range[0]
        zb = s.z_range[1] - s.z_range[0]
        xp = (s.x_range[0] + np.random.random(1)*xb).item()
        yp = (s.y_range[0] + np.random.random(1)*yb).item()
        zp = (s.z_range[0] + np.random.random(1)*zb).item()
        tmp = s.DD[0]([xp,yp,zp,len(s.patches)])[0]
        v = []
        p = []
        if tmp > value_max or tmp < value_min:
            continue
        if np.isnan(tmp):
            continue
        sample = np.linspace(2,len(s.patches),samples,dtype=int)
        for i in sample:
            v.append(s.DD[0]([xp,yp,zp,i]).tolist())
            p.append(i)
        v = np.array(v)
        p = np.array(p)
        v = abs(v - v[-1])
        v = v/max(v)
        p = p/max(p)
        values.append(v.tolist())
        patches.append(p.tolist())
        count += 1
        pbar.update(1)
    pbar.close()
    values = np.array(values)
    patches = np.array(patches)
    mean = np.mean(values,axis=0)
    std  = np.std(values,axis=0)
    p_mean = np.mean(patches,axis=0)
    #fig, ax = plt.subplots()
    #ax.scatter(p_mean,mean)
    #ax.set_xlabel('Percent of Total Patches')
    #ax.set_ylabel('Relative Error')
    #plt.show()
    return p_mean,mean,std

def test(s,bins=10,bin_range=(-1,0),reps=3,samples=100):
    pts = s.pv_polydata.points
    pts = [pt.tolist() for pt in pts]
    #pts = [pt.append(len(s.patches)) for pt in pts]
    values = []
    for pt in pts:
        data = pt
        data.append(len(s.patches))
        values.append(s.DD[0](data)[0])
    values = np.array(values).flatten()
    #s.depth_interior_min = min(values)
    s.depth_interior_max = max(values)
    bin_r = abs(bin_range[1] - bin_range[0])
    bin_range = [abs(bin_range[0])*s.depth_interior_min,abs(bin_range[0])*s.depth_interior_min+bin_r*abs(s.depth_interior_min - s.depth_interior_max)]
    BINS = np.linspace(bin_range[0],bin_range[1],bins+1)
    VALUES_MEAN = []
    VALUES_STD  = []
    PATCHES     = []
    for i in range(bins):
        p_mean,mean,std = get_values(s,BINS[i],BINS[i+1],reps,samples)
        VALUES_MEAN.append(mean)
        VALUES_STD.append(std)
        PATCHES.append(p_mean)
    #fig,ax = plt.subplots()
    #for i in range(bins):
    #    ax.scatter(PATCHES[i],VALUES_MEAN[i],label="{}-{}".format(BINS[i],BINS[i+1]))
    #ax.set_xlabel('Percent of Total Patches')
    #ax.set_ylabel('Relative Error')
    #plt.legend()
    #plt.show()
    return PATCHES,VALUES_MEAN,VALUES_STD

def convexity(surf_list):
    """
    Convexity measure obtained from paper below:

    CON  = V/V(CH)

    Shi, X., Li, R., Sheng, Y. (2020). A New Volume-Based Convexity Measure for
    3D Shapes. In: , et al. Advances in Computer Graphics. CGI 2020. Lecture
    Notes in Computer Science(), vol 12221. Springer, Cham.
    https://doi.org/10.1007/978-3-030-61864-3_6
    """
    con = []
    for surf in surf_list:
        delaunay = surf.pv_polydata.delaunay_3d()
        surf.convexity = surf.volume/delaunay.volume
        con.append(surf.convexity)
    return con

def gen_figure(surf_list,bins=1,bin_range=(-1,0),reps=10,samples=100):
    con = convexity(surf_list)
    c = np.array(con)
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    cmap.set_array([])
    surf_data = []
    fig,ax = plt.subplots()
    for surf in surf_list:
        p,vm,vs = test(surf,bins=bins,bin_range=bin_range,reps=reps,samples=samples)
        for i in range(bins):
            ax.scatter(p[i],vm[i],color=cmap.to_rgba(surf.convexity),marker=(2+i,2,0))
        surf_data.append((p,vm,vs))
    ax.set_xlabel('Percent of Total Patches')
    ax.set_ylabel('Relative Error')
    fig.colorbar(cmap,label=r"Convexity")
    name = "Patch_accuracy_convexity_bins{}_range{}_{}_reps{}.svg".format(bins,bin_range[0],bin_range[1],reps)
    fig.savefig(name,format="svg")
    plt.show()
    return surf_data
#gen_figure([cube,cylinder,heart,gyrus],bins=4)
"""
def test(s,bins=10,bin_range=(-1,0),reps=3):
    mesh = s.tet.grid
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    BINS = np.linspace(bin_range[0],bin_range[1],bins+1)
    VALUES = [[None]*reps]*bins
    done = False
    while not done:
        xb = mesh.bounds[1] - mesh.bounds[0]
        yb = mesh.bounds[3] - mesh.bounds[2]
        zb = mesh.bounds[5] - mesh.bounds[4]
        xp = (mesh.bounds[0] + np.random.random(1)*xb).item()
        yp = (mesh.bounds[2] + np.random.random(1)*yb).item()
        zp = (mesh.bounds[4] + np.random.random(1)*zb).item()
        tmp = s.DD[0]([xp,yp,zp,len(s.patches)])[0]
        tmp_ind = place_bin(tmp,bins,bin_range)
        if tmp_ind is None:
            continue
        if not None in VALUES[tmp_ind]:
            continue
        value  = []
        patch  = []
        for i in range(2,len(s.patches)):
            value.append(s.DD[0]([xp,yp,zp,i]))
            patch.append(i)
        value = np.array(value)
        name = value[-1].item()
        patch = np.array(patch)
        value = abs(value - value[-1])
        value = value/max(value)
        next_ind = VALUES[tmp_ind].index(None)
        VALUES[tmp_ind][next_ind] = value.tolist()
        flat_values = list(chain(*VALUES))
        if None not in flat_values:
            done = True
            break
        #ax.plot(patch,value,label=str(name))
    for i in range(bins):
        VALUE_bin = np.array(VALUES[i])
        VALUE_MEAN = np.mean(VALUE_bin,axis=0)
        VALUE_STD  = np.std(VALUE_bin,axis=0)
        ax.scatter(patch/patch[-1],VALUE_MEAN)
    #ax.fill_between(patch,VALUE_MEAN-VALUE_STD,VALUE_MEAN+VALUE_STD,alpha=0.5)
    plt.xlabel('Percent of Total Patches')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.show()
    return VALUES
"""

#V = test(s)
