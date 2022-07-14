import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import as_completed
from skimage import measure
from time import time
from tqdm import tqdm
from pickle import loads
#import pandas as pd
import vtk
import plotly.graph_objects as go

from ..tests.bumpy_sphere import bumpy_sphere

def mpu_meshgrid(mpu_object,buf=1.25,res=10,workers=1,k=None,plane_axis=None,plane_value=None,marching=False,gradient=False,record=False,global_f=False):
    dims = []
    if not marching:
        for d in range(mpu_object.ddim):
            if d is plane_axis:
                continue
            dims.append(np.linspace(mpu_object.dim_range[d*2]-buf,
                                    mpu_object.dim_range[d*2+1]+buf,res))
    else:
        res = res*1j
        x,y,z = np.ogrid[mpu_object.dim_range[0]-buf:mpu_object.dim_range[1]+buf:res,
                         mpu_object.dim_range[2]-buf:mpu_object.dim_range[3]+buf:res,
                         mpu_object.dim_range[4]-buf:mpu_object.dim_range[5]+buf:res]
        results = mpu_object.function_marching(x,y,z)
        return (x,y,z),results
    DIMS = np.meshgrid(*dims)
    DIMSf = []
    for d in range(len(DIMS)):
        DIMSf.append(DIMS[d].flatten())
    value = []
    if plane_value is not None:
        plane_values = [plane_value]*len(DIMSf[0])
        DIMSf.insert(plane_axis,plane_values)
        plane = np.array(plane_values).reshape(DIMS[0].shape)
        DIMS.insert(plane_axis,plane)
    if workers > 1:
        chunksize = len(DIMSf[0])//workers
        number_of_chunks = len(DIMSf[0])//chunksize
        executor = PPE(max_workers=workers)
        if global_f:
            pass
        elif k is not None:
            k = [k]*len(DIMSf[0])
            DIMSf.append(k)
        else:
            k = [len(mpu_object.patches)]*len(DIMSf[0])
            DIMSf.append(k)
        if not gradient:
            result_generator  = executor.map(loads(mpu_object.pickled_DD[0]),zip(*DIMSf),chunksize=chunksize)
        else:
            result_generator  = executor.map(loads(mpu_object.pickled_DD[1]),zip(*DIMSf),chunksize=chunksize)
        executor.shutdown(wait=True)
        if not gradient:
            results = []
            for result in result_generator:
                results.append(result)
            results = np.array(results)
            results = results.reshape(DIMS[0].shape)
        else:
            u = []
            v = []
            w = []
            for result in result_generator:
                u.append(result[0])
                v.append(result[1])
                w.append(result[2])
            u = np.array(u)
            u = u.reshape(DIMS[0].shape)
            v = np.array(v)
            v = u.reshape(DIMS[0].shape)
            w = np.array(w)
            w = u.reshape(DIMS[0].shape)
            results = [u,v,w]
    else:
        time_data = []
        results = []
        if global_f:
            pass
        elif k is not None:
            k = [k]*len(DIMSf[0])
            DIMSf.append(k)
        else:
            k = [len(mpu_object.patches)]*len(DIMSf[0])
            DIMSf.append(k)
        if not gradient:
            for i in zip(*DIMSf):
                start = time()
                results.append(mpu_object.DD[0](i))
                stop = time() - start
                time_data.append(stop)
            results = np.array(results)
            results = results.reshape(DIMS[0].shape)
            if record:
                results = time_data
        else:
            u = []
            v = []
            w = []
            for i in zip(*DIMSf):
                start = time()
                results = mpu_object.DD[1](i)
                stop = time() - start
                time_data.append(stop)
                u.append(results[0])
                v.append(results[1])
                w.append(results[2])
            u = np.array(u)
            u = u.reshape(DIMS[0].shape)
            v = np.array(v)
            v = v.reshape(DIMS[0].shape)
            w = np.array(w)
            w = w.reshape(DIMS[0].shape)
            results = [u,v,w]
            if record:
                results = time_data
    return DIMS,results

def plot_volume(mpu_object,resolution=20,workers=1,cmin=-1,cmax=0,surface_count=14,k=None,global_f=False):
    DIMS,results = mpu_meshgrid(mpu_object,res=resolution,workers=workers,k=k,global_f=global_f)
    fig = go.Figure(data=(go.Isosurface(
                    x=DIMS[0].flatten(),
                    y=DIMS[1].flatten(),
                    z=DIMS[2].flatten(),
                    value=results.flatten(),
                    opacity=0.1,
                    isomin=cmin,
                    isomax=cmax,
                    surface_count=surface_count),go.Scatter3d(x = mpu_object.points[:,0],y=mpu_object.points[:,1],z=mpu_object.points[:,2])))
    fig.show()

def plot_volume_individual(func,x_range,y_range,z_range,buf=1.25,res=20,surface_count=15,cmin=-1,cmax=0):
    res = res*1j
    x,y,z = np.ogrid[x_range[0]-buf:x_range[1]+buf:res,
                     y_range[0]-buf:y_range[1]+buf:res,
                     z_range[0]-buf:z_range[1]+buf:res]
    results = func(x,y,z)
    fig = go.Figure(data=(go.Isosurface(
                    x=x.flatten(),
                    y=y.flatten(),
                    z=z.flatten(),
                    value=results.flatten(),
                    opacity=0.1,
                    isomin=cmin,
                    isomax=cmax,
                    surface_count=surface_count)))
    fig.show()

def plot_slice_error(mpu_object,resolution=20,k=None,workers=1,plane_axis=2,plane_value=0,unit_scale='mm'):
    DIMS,results_exact = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                      k=len(mpu_object.patches),plane_axis=plane_axis,
                                      plane_value=plane_value)
    DIMS,results_approximate = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                      k=k,plane_axis=plane_axis,plane_value=plane_value)
    DIMSv,voxels = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                k=len(mpu_object.patches),marching=True)
    verts, faces, normals, values = measure.marching_cubes(voxels,level=0)
    verts *= np.array([np.diff(ar.flat)[0] for ar in [DIMSv[0],DIMSv[1],DIMSv[2]]])
    verts += np.array([DIMSv[0].min(),DIMSv[1].min(),DIMSv[2].min()])

    error = results_exact - results_approximate
    error_min = error.min()
    error_max = error.max()
    error_divide = max(abs(error_min),abs(error_max))
    error = error/(error_divide)
    colormap = plt.cm.RdBu
    main_axes = list(range(len(DIMS)))
    main_axes.remove(plane_axis)
    axes_labels = ['x ({})'.format(unit_scale),'y ({})'.format(unit_scale),'z ({})'.format(unit_scale)]
    axes_labels_all = ['x ({})'.format(unit_scale),'y ({})'.format(unit_scale),'z ({})'.format(unit_scale)]
    axes_labels.remove(axes_labels[plane_axis])
    fig = plt.figure(figsize=(12,6))
    spec = gridspec.GridSpec(ncols=4, nrows=1,
                             width_ratios=[1, 1, 1, 1.5])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax3 = fig.add_subplot(spec[2])
    ax4 = fig.add_subplot(spec[3],projection='3d')

    cax1 = ax1.contourf(DIMS[main_axes[0]],DIMS[main_axes[1]],results_exact,norm=colors.TwoSlopeNorm(0))
    ax1.contour(DIMS[main_axes[0]],DIMS[main_axes[1]],results_exact,levels=[0],linewidths=(2,))
    #ax1.set_title('Exact Interpolation \n {} Patches'.format(len(mpu_object.patches)))
    ax1.set_xlabel(axes_labels[0])
    ax1.set_ylabel(axes_labels[1])

    cax2 = ax2.contourf(DIMS[main_axes[0]],DIMS[main_axes[1]],results_approximate,norm=colors.TwoSlopeNorm(0))
    ax2.contour(DIMS[main_axes[0]],DIMS[main_axes[1]],results_approximate,levels=[0],linewidths=(2,))
    #ax2.set_title('Approximate Interpolation \n {} Patches'.format(k))
    ax2.set_xlabel(axes_labels[0])

    cax3 = ax3.contourf(DIMS[main_axes[0]],DIMS[main_axes[1]],error,cmap=colormap,norm=colors.TwoSlopeNorm(0))

    #ax3.set_title('Relative Truncation Error')
    ax3.set_xlabel(axes_labels[0])

    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax4.add_collection3d(mesh)
    ax4.plot_surface(DIMS[0],DIMS[1],DIMS[2],alpha=0.2,color='r')
    ax4.set_xlim(DIMSv[0].min(),DIMSv[0].max())
    ax4.set_ylim(DIMSv[1].min(),DIMSv[1].max())
    ax4.set_zlim(DIMSv[2].min(),DIMSv[2].max())
    ax4.set_xlabel(axes_labels_all[0])
    ax4.set_ylabel(axes_labels_all[1])
    ax4.set_zlabel(axes_labels_all[2])
    plt.draw()
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0.1, p1[2]-p0[0], 0.05])
    ax_cbar.set_in_layout(True)
    cbar1 = plt.colorbar(cax1, cax=ax_cbar, orientation='horizontal', pad=0.25)
    labels = cbar1.ax.get_xticklabels()
    cbar1.ax.set_xticklabels(labels, rotation=40)


    ax_cbar1 = fig.add_axes([p2[0], 0.1, p2[2]-p2[0], 0.05])
    ax_cbar1.set_in_layout(True)
    cbar2 = plt.colorbar(cax3, cax=ax_cbar1, orientation='horizontal', pad=0.25)
    labels = cbar2.ax.get_xticklabels()
    cbar2.ax.set_xticklabels(labels, rotation=40)
    plt.subplots_adjust(bottom=0.25)

    plt.show()

def plot_gradient(mpu_object,workers=1,k=None,resolution=20,plane_axis=2,plane_value=0,
                  contour_overlay=True,unit_scale='mm'):
    DIMS,gradient = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                 plane_axis=plane_axis,plane_value=plane_value,
                                 gradient=True)
    DIMS,gradient_approximate = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                             plane_axis=plane_axis,plane_value=plane_value,
                                             k=k,gradient=True)
    if contour_overlay:
        DIMS_c,result = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                     plane_axis=plane_axis,plane_value=plane_value)
    main_axes = list(range(len(DIMS)))
    main_axes.remove(main_axes[plane_axis]) #use del?
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)

    if contour_overlay:
        cax = ax.contourf(DIMS[main_axes[0]],DIMS[main_axes[1]],result,norm=colors.TwoSlopeNorm(0))
        p0 = ax.get_position().get_points().flatten()
        ax_cbar = fig.add_axes([p0[0], 0.1, p0[2]-p0[0], 0.05])
        ax_cbar.set_in_layout(True)
        cbar1 = plt.colorbar(cax, cax=ax_cbar, orientation='horizontal', pad=0.25)
        labels = cbar1.ax.get_xticklabels()
        cbar1.ax.set_xticklabels(labels, rotation=40)

    ax.quiver(DIMS[main_axes[0]],DIMS[main_axes[1]],
              gradient[main_axes[0]],gradient[main_axes[1]])

    ax.quiver(DIMS[main_axes[0]],DIMS[main_axes[1]],
              gradient_approximate[main_axes[0]],gradient_approximate[main_axes[1]],color='r')
    axes_labels = ['x ({})'.format(unit_scale),'y ({})'.format(unit_scale),'z ({})'.format(unit_scale)]
    axes_labels.remove(axes_labels[plane_axis])
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def plot_gradient_error(mpu_object,workers=1,k=None,resolution=20,plane_axis=2,plane_value=0,
                  contour_overlay=True,unit_scale='mm'):
    DIMS,gradient = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                 plane_axis=plane_axis,plane_value=plane_value,
                                 gradient=True)
    DIMS,approximate = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                    plane_axis=plane_axis,plane_value=plane_value,
                                    k=k,gradient=True)
    DIMS_c,result = mpu_meshgrid(mpu_object,res=resolution,workers=workers,
                                 plane_axis=plane_axis,plane_value=plane_value)
    gradient = [gradient[0].flatten(),gradient[1].flatten(),gradient[2].flatten()]
    approximate = [approximate[0].flatten(),approximate[1].flatten(),approximate[2].flatten()]
    alignment_error = []
    magnitude_error = []
    for i in range(len(gradient[0])):
        grad_mag = (gradient[0][i]**2+gradient[1][i]**2+gradient[2][i]**2)**(1/2)
        approx_mag = (approximate[0][i]**2+approximate[1][i]**2+approximate[2][i]**2)**(1/2)
        alignment_error.append((gradient[0][i]/grad_mag)*(approximate[0][i]/approx_mag)+(gradient[1][i]/grad_mag)*(approximate[1][i]/approx_mag)+
                               (gradient[2][i]/grad_mag)*(approximate[2][i]/approx_mag))
        magnitude_error.append(grad_mag-approx_mag)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(result.flatten(),alignment_error)
    ax1.set_xlabel('Contour Layer')
    ax1.set_ylabel('Alignment Difference')
    ax2 = fig.add_subplot(122)
    ax2.scatter(result.flatten(),magnitude_error)
    ax2.set_xlabel('Contour Layer')
    ax2.set_ylabel('Magnitude Difference')
    plt.show()

def plot_error(mpu_object,workers=1,k=None,resolution=20,plane_axis=2,plane_value=0):
    DIMS,result = mpu_meshgrid(mpu_object,workers=workers,res=resolution,plane_axis=plane_axis,
                               plane_value=plane_value)
    DIMS,approx = mpu_meshgrid(mpu_object,workers=workers,k=k,res=resolution,plane_axis=plane_axis,
                               plane_value=plane_value)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    main_axes = list(range(len(DIMS)))
    main_axes.remove(main_axes[plane_axis])
    contours = ax.contour(DIMS[main_axes[0]],DIMS[main_axes[1]],result)
    levels = contours.levels[1:-1]
    level_areas = []
    level_error = []
    error_per_area = []
    resultf = result.flatten()
    approxf = approx.flatten()
    l0 = -1
    a0 = 0
    passed_levels = []
    for idx, level in enumerate(levels):
        #contour = contours.collections[idx]
        #if len(contour.get_paths()) == 0:
        #    continue
        #else:
        #    passed_levels.append(level)
        #vs = contour.get_paths()[0].vertices
        #a = 0
        #d0_0,d1_0 = vs[0]
        #for [d0_1,d1_1] in vs[1:]:
        #    dd0 = d0_1 - d0_0
        #    dd1 = d1_1 - d1_0
        #    a += 0.5*(d1_0*dd0 - d0_0*dd1)
        #    d0_0 = d0_1
        #    d1_0 = d1_1
        #level_areas.append(a-a0)
        #a0 = a
        lower_filter = set(np.argwhere(l0<resultf).flatten())
        upper_filter = set(np.argwhere(resultf<level).flatten())
        bounded_filter = list(lower_filter.intersection(upper_filter))
        level_error.append(np.mean(resultf[bounded_filter] - approxf[bounded_filter]))
        l0 = level
        #error_per_area.append(level_error[-1]/level_areas[-1])
    plt.clf()
    plt.plot(levels,level_error)
    plt.show()

def plot_time(mpu_object,resolution=20,k_start=2,k_end=100,gradient=False,plane_axis=None,plane_value=None):
    all_time = []
    mean_time = []
    k_values  = []
    sd_time_values = []
    all_error = []
    mean_error = []
    sd_error = []
    all_k = []
    #d_type = []
    DIMS,exact = mpu_meshgrid(mpu_object,k=None,res=resolution,gradient=gradient,
                                  plane_axis=plane_axis,plane_value=plane_value)
    for i in tqdm(range(k_start,k_end+1),desc='Evaluating Time Grids  '):
        k_values.append(i)
        DIMS,time_data = mpu_meshgrid(mpu_object,k=i,res=resolution,gradient=gradient,
                                      plane_axis=plane_axis,plane_value=plane_value,
                                      record=True)
        DIMS,approx = mpu_meshgrid(mpu_object,k=i,res=resolution,gradient=gradient,
                                      plane_axis=plane_axis,plane_value=plane_value)
        error = abs(exact.flatten() - approx.flatten())
        k_tmp = [i]*len(error)
        #d_type.extend(['error']*len(error))
        #d_type.extend(['time']*len(time_data))
        mean_error.append(np.mean(error))
        sd_error.append(np.std(error))
        all_time.extend(time_data)
        mean_time.append(np.mean(time_data))
        sd_time_values.append(np.std(time_data))
        all_error.extend(error)
        all_k.extend(k_tmp)

    data = pd.DataFrame()
    data['times'] = all_time
    data['k'] = all_k
    data['error'] = all_error
    sn.lineplot(data=data,x='k',y='times')
    ax2 = plt.twinx()
    sn.lineplot(data=data,x='k',y='error',ax=ax2,color='red')
    plt.show()

def time_PU_global(mpu_object):
    start = time()
    mpu_object.solve()
    pu_time = time()-start
    start = time()
    mpu_object.solve(PU=False)
    global_time = time()-start
    return pu_time,global_time

#def pu_v_global(mpu_object,start=10,stop=40):

def show_path(points,points2):
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    ax.scatter3D(points[0,0],points[0,1],points[0,2],color='g')
    ax.scatter3D(points[1:,0],points[1:,1],points[1:,2],color='r')
    ax.scatter3D(points2[:,0],points2[:,1],points2[:,2],color='b')
    plt.show()

def show_mesh(points,points2,points3):
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    ax.scatter3D(points[:,0],points[:,1],points[:,2],color='r')
    ax.scatter3D(points2[:,0],points2[:,1],points2[:,2],color='b')
    ax.scatter3D(points3[:,0],points3[:,1],points3[:,2],color='g')
    plt.show()

def newton(mpu_object):
    f1 = mpu_object.function([1,1,1,None])
    f2 = mpu_object.function([1.1,1,1,None])
    g = np.array(mpu_object.gradient([1,1,1,None]))
    print('F1: {}'.format(f1))
    print('F2: {}'.format(f2))
    print('Ap: {}'.format(f1+np.dot(np.array([0.1,0,0]),g)))

def show_patches(mpu_object):
    patch_sets = [set(tuple([tuple(pp) for pp in p.points.tolist()])) for p in mpu_object.patches]
    unique_points = []
    overlap_points = []
    for patch_idx,patch_i in enumerate(patch_sets):
        patch_i_diff = patch_i.difference({})
        patch_i_overlap = set([])
        for patch_jdx,patch_j in enumerate(patch_sets):
            if patch_idx == patch_jdx:
                continue
            else:
                patch_i_diff = patch_i_diff.difference(patch_j)
                patch_i_overlap.update(patch_i.intersection(patch_j))
            print(len(unique_points))
        unique_points.extend(list(patch_i_diff))
        overlap_points.extend(list(patch_i_overlap))
    unique_points = np.array(unique_points)
    overlap_points = np.array(overlap_points)
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    ax.scatter3D(unique_points[:,0],unique_points[:,1],unique_points[:,2],color='r')
    ax.scatter3D(overlap_points[:,0],overlap_points[:,1],overlap_points[:,2],color='b')
    plt.show()
#get area of contourf for L1 error estimate as a function of "tissue depth" which will basically argue
#that for our area of interest we can ignore the problematic areas of unstable far-field growth

#for gradient error take the angle and magnitude differences and also show them as a function of tissue
# layer position
