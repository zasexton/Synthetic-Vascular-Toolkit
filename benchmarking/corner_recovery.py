# Corner recovery
import svcco
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build()

def test(mpu_object,h=0,q_range=(1,8),resolution=20,k=2,buf=1.25,level=0,workers=1,plane_axis=2,plane_value=0.5,unit_scale='mm',name="untitled"):
    if plane_axis == 0:
        plane_value = (mpu_object.x_range[1]-mpu_object.x_range[0])*plane_value + mpu_object.x_range[0]
    elif plane_axis == 1:
        plane_value = (mpu_object.y_range[1]-mpu_object.y_range[0])*plane_value + mpu_object.y_range[0]
    elif plane_axis == 2:
        plane_value = (mpu_object.z_range[1]-mpu_object.z_range[0])*plane_value + mpu_object.z_range[0]
    colormap = plt.cm.RdBu
    #main_axes = list(range(len(DIMS)))
    #main_axes.remove(plane_axis)
    c = np.arange(q_range[0], q_range[1] + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    fig,ax = plt.subplots()
    for q in range(q_range[0],q_range[1]+1):
        mpu_object.build(q=q,h=h)
        DIMS,results_exact = svcco.implicit.visualize.visualize.mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                                                             k=len(mpu_object.patches),plane_axis=plane_axis,
                                                                             plane_value=plane_value,buf=buf)
        main_axes = list(range(len(DIMS)))
        main_axes.remove(plane_axis)
        CS = ax.contour(DIMS[main_axes[0]],DIMS[main_axes[1]],results_exact,levels=[level],linewidths=(2,))
        C = CS.collections[0]
        C.set_color(cmap.to_rgba(q+1))
        #fmt = {CS.levels[0]:r"$\Gamma_{q=" +str(i) + r"}$"}
        #ax.clabel(CS,fmt=fmt,fontsize=10)
    fig.colorbar(cmap, ticks=c,label=r"Patch Degree, $q$")
    name = "corner_recovery_q{}-{}.svg".format(q_range[0],q_range[1])
    fig.savefig(name,format="svg")
    plt.show()
