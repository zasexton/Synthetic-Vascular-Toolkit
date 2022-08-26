import svcco
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from tqdm import tqdm
import seaborn as sns

cube = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(5)
sphere = pv.Sphere().triangulate().subdivide(1)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()

"""
heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()

disk = pv.Disc(r_res=20,c_res=100)
cyl  = disk.extrude([0,0,1],capping=True).triangulate()
cyl  = svcco.utils.remeshing.remesh.remesh_surface(cyl)
cyl  = cyl.subdivide(2)
cyl  = cyl.compute_normals(auto_orient_normals=True)
c = svcco.surface()
c.set_data(cyl.points,cyl.point_normals)
c.solve()


left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"

gyrus = svcco.surface()
gyrus.load(left_gyrus)
gyrus.solve()
"""
def test(surface_object,patch_size_range=(10,20),range_step=10):
    points = surface_object.points
    normals = surface_object.normals
    lower_size = []
    all_cond = []
    all_sizes = []
    start = patch_size_range[0]
    while start < patch_size_range[1]:
        surface_object.set_data(points,normals,local_min=start,local_max=2*start)
        surface_object.solve(quiet=False)
        condition_numbers = []
        patch_sizes = []
        for patch in surface_object.patches:
            condition_numbers.append(np.linalg.cond(patch.A_inv))
            patch_sizes.append(patch.points.shape[0])
        all_cond.append(np.array(condition_numbers))
        all_sizes.append(np.array(patch_sizes))
        lower_size.append(int(start))
        start += range_step
    lower_size = np.array(lower_size)
    #all_cond = np.array(all_cond)
    figure,ax = plt.subplots()
    ax.boxplot(all_cond,labels=lower_size)
    ax.set_yscale('log')
    plt.savefig('condition_number_plot.svg', format="svg", bbox_inches='tight')
    figure,ax = plt.subplots()
    ax.boxplot(all_sizes,labels=lower_size)
    plt.savefig('condition_patch_size_plot.svg', format="svg", bbox_inches='tight')
