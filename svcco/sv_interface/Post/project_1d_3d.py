#!/usr/bin/env python

import pdb
import sys
import os
import vtk
import shutil

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from lift_laplace import StiffnessMatrix
from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo, ClosestPoints, cell_connectivity, region_grow, collect_arrays
from simulation_io import collect_results, get_caps_db
from compare_1d import plot_1d_3d_interior as plt1d3d


def plot_projection(db, geometries):
    for geo in geometries:
        # plot options
        opt = {'legend_col': False,
               'legend_row': False,
               'sharex': True,
               'sharey': 'row',
               'dpi': 200,
               'w': 15,
               'h': 6}

        res = defaultdict(lambda: defaultdict(dict))
        time = {}

        time_inflow, _ = db.get_inflow_smooth(geo)

        f_asym = '/home/pfaller/work/osmsc/data_generated/asymptotic/' + geo + '.vtp'
        f_proj = '/home/pfaller/work/osmsc/data_generated/initial_pressure/' + geo + '.vtp'
        # f_res_1d = db.get_1d_flow_path(geo)
        f_res_1d = '/home/pfaller/work/osmsc/studies/1spb_length/1d_flow_last/' + geo + '.npy'
        f_rerun = '/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '.vtp'
        f_oned = db.get_1d_geo(geo)

        # collect_results('3d_rerun', res, time, f_asym, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))
        collect_results('3d_rerun', res, time, f_rerun, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))
        collect_results('3d', res, time, f_proj, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))
        collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])

        plot_1d_3d_interior(db, opt, geo, res, time)

        # new plot
        res0 = defaultdict(lambda: defaultdict(dict))
        time0 = {}

        time_inflow, _ = db.get_inflow_smooth(geo)
        f_rerun = '/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '_first100.vtp'
        collect_results('3d_rerun', res0, time0, f_rerun, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))

        res1 = defaultdict(lambda: defaultdict(dict))
        time1 = {}

        time_inflow, _ = db.get_inflow_smooth(geo)
        f_rerun = '/home/pfaller/work/osmsc/studies/ini_zero/3d_flow/' + geo + '_first1000.vtp'
        collect_results('3d_rerun', res1, time1, f_rerun, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))

        res2 = defaultdict(lambda: defaultdict(dict))
        time2 = {}

        time_inflow, _ = db.get_inflow_smooth(geo)
        f_rerun = '/home/pfaller/work/osmsc/studies/ini_steady/3d_flow/' + geo + '_first100.vtp'
        collect_results('3d_rerun', res2, time2, f_rerun, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo))

        plot_1d_3d_interior_time(db, opt, geo, res0, time0, res1, res2)

def plot_1d_3d_interior(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get models
    models = [k[:-4] for k in time.keys() if '_all' in k]

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    cap_br = list(caps.values())
    cap_names = list(caps.keys())

    names = db.get_cap_names(geo)
    names['RCCA'] = 'Right common carotid'

    if len(res) > 50:
        dpi = opt['dpi'] // 4
        sharey = False
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fig, ax = plt.subplots(len(post.fields), len(res), figsize=(opt['w'], opt['h']), dpi=dpi, sharex='col', sharey=sharey)

    for i, f in enumerate(['pressure', 'flow']):
        for j, br in enumerate(res.keys()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                if br in cap_br:
                    name = names[cap_names[cap_br.index(br)]]
                    # if not name.isupper():
                    #     name = name.capitalize()
                else:
                    name = 'branch ' + str(br)
                ax[i, j].set_title(name)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Vessel path [-]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in models:
                path = res[br][m + '_path']
                # if f == 'flow' and m == '1d':
                #     pdb.set_trace()
                if m == '3d_rerun':
                    ax[i, j].plot(path / path[-1], res[br][f][m + '_int_last'][:, -1] * post.convert[f], post.styles[m])
                else:
                    ax[i, j].plot(path / path[-1], res[br][f][m + '_int'][:, -1] * post.convert[f], post.styles[m])
                lg.append(m)

            # ax[i, j].legend(lg)
            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_xticks([0, 1])
            if f == 'flow':
                ax[i, j].set_ylim(0, 2.5)

    # add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'interior'), bbox_inches='tight')
    plt.close(fig)

def plot_1d_3d_interior_time(db, opt, geo, res, time, res1, res2):
    # get post-processing constants
    post = Post()

    # get models
    models = [k[:-4] for k in time.keys() if '_all' in k]

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    cap_br = list(caps.values())
    cap_names = list(caps.keys())

    names = db.get_cap_names(geo)
    names['RCCA'] = 'Right common carotid'

    if len(res) > 50:
        dpi = opt['dpi'] // 4
        sharey = False
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fig, ax = plt.subplots(1, len(res), figsize=(15, 3), dpi=dpi, sharey=sharey)

    f = 'flow'
    m = '3d_rerun'
    n_max = 100
    dt = db.get_3d_timestep(geo)
    for j, br in enumerate(res.keys()):
        ax[j].grid(True)
        if br in cap_br:
            name = names[cap_names[cap_br.index(br)]]
            # if not name.isupper():
            #     name = name.capitalize()
        else:
            name = 'branch ' + str(br)
        ax[j].set_title(name)
        ax[j].set_xlabel('Time [s]')
        ax[j].xaxis.set_tick_params(which='both', labelbottom=True)
        # ax[j].xaxis.set_ticks(np.linspace(0, n_max * dt, 4))
        ax[j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if opt['legend_col'] or j == 0:
            ax[j].set_ylabel('Max. flow deviation from mean flow [-]')
            ax[j].yaxis.set_tick_params(which='both', labelleft=True)

        path = res[br][m + '_path']
        results = [res1, res]#, res2
        colors = ['k', plt.get_cmap("tab10")(1)]#, 'r'
        for col, res_sol in zip(colors, results):
            diff = []
            for n in range(n_max):
                sol = res_sol[br][f][m + '_int'][:, n]

                # exclude caps
                if br == 0:
                    sol = sol[1:]
                else:
                    sol = sol[:-1]

                # relative difference
                diff += [np.max(np.abs(sol / np.mean(sol) - 1))]
                # ax[j].plot(path[1:] / path[-1], sol, post.styles[m])
            ax[j].semilogy(np.arange(n_max) * dt, diff, post.styles[m], color=col)
        ax[j].set_xlim([-0.001, n_max * dt])
        # ax[j].set_ylim([1e-5, 1e-1])
    fig.savefig(db.get_post_path(geo, 'interior_time'), bbox_inches='tight')
    plt.close(fig)


def project_1d_3d_lift(f_1d, f_vol, f_out, field):
    # read volume mesh
    vol = read_geo(f_vol).GetOutput()
    points_vol = v2n(vol.GetPoints().GetData())
    cells = cell_connectivity(vol)

    # read 1d results
    oned = read_geo(f_1d).GetOutput()
    points_1d = v2n(oned.GetPoints().GetData())

    # get volume points closest to centerline
    cp_vol = ClosestPoints(vol)
    ids_vol = np.unique(cp_vol.search(points_1d))

    # get centerline points closest to selected volume points
    cp_1d = ClosestPoints(oned)
    ids_cent = cp_1d.search(points_vol[ids_vol])

    # visualize imprint of centerline in volume mesh
    imprint = np.zeros(vol.GetNumberOfPoints())
    imprint[ids_vol] = 1
    arr = n2v(imprint)
    arr.SetName('imprint')
    vol.GetPointData().AddArray(arr)

    # get 1d field field
    field_1d = v2n(oned.GetPointData().GetArray(field))

    # create laplace FEM stiffness matrix
    laplace = StiffnessMatrix(cells['tetra'], points_vol)

    # solve laplace equation (map desired field from 1d to 3d)
    field_3d = laplace.HarmonicLift(ids_vol, field_1d[ids_cent])

    # create output array
    arr = n2v(field_3d)
    arr.SetName(field)
    vol.GetPointData().AddArray(arr)

    # write to file
    write_geo(f_out, vol)


def get_1d_3d_map(f_1d, f_vol):
    # read geoemtries
    vol = read_geo(f_vol).GetOutput()
    oned = read_geo(f_1d).GetOutput()

    # get points
    points_vol = v2n(vol.GetPoints().GetData())
    points_1d = v2n(oned.GetPoints().GetData())

    # get volume points closest to centerline
    cp_vol = ClosestPoints(vol)
    seed_points = np.unique(cp_vol.search(points_1d))

    # map centerline points to selected volume points
    cp_1d = ClosestPoints(oned)
    seed_ids = np.array(cp_1d.search(points_vol[seed_points]))

    # call region growing algorithm
    ids, dist, rad = region_grow(vol, seed_points, seed_ids, n_max=999)

    # check 1d to 3d map
    assert np.max(ids) <= oned.GetNumberOfPoints() - 1, '1d-3d map non-conforming'

    return ids, dist, rad


def add_array(geo, name, array):
    arr = n2v(array)
    arr.SetName(name)
    geo.GetPointData().AddArray(arr)


def project_1d_3d_grow(f_1d, f_vol, f_wall, f_out):
    # read geometries
    vol = read_geo(f_vol).GetOutput()
    cent = read_geo(f_1d).GetOutput()
    wall = read_geo(f_wall).GetOutput()

    # get 1d -> 3d map
    map_ids, map_iter, map_rad = get_1d_3d_map(f_1d, f_vol)

    # get arrays
    arrays_cent = collect_arrays(cent.GetPointData())

    # map all centerline arrays to volume geometry
    for name, array in arrays_cent.items():
        add_array(vol, name, array[map_ids])

    # add mapping to volume mesh
    for name, array in zip(['MapIds', 'MapIters'], [map_ids, map_iter]):
        add_array(vol, name, array)

    # inverse map
    map_ids_inv = {}
    for i in np.unique(map_ids):
        map_ids_inv[i] = np.where(map_ids == i)

    # create radial coordinate [0, 1]
    rad = np.zeros(vol.GetNumberOfPoints())
    for i, ids in map_ids_inv.items():
        rad_max = np.max(map_rad[ids])
        if rad_max == 0:
            rad_max = np.max(map_rad)
        rad[ids] = map_rad[ids] / rad_max
    add_array(vol, 'rad', rad)

    # set points at wall to hard 1
    wall_ids = collect_arrays(wall.GetPointData())['GlobalNodeID'].astype(int) - 1
    rad[wall_ids] = 1

    # mean velocity
    names = ['flow', 'velocity']
    for n in names:
        for a in arrays_cent.keys():
            if n in a:
                u_mean = arrays_cent[a] / arrays_cent['CenterlineSectionArea']

                # parabolic velocity
                u_quad = 2 * u_mean[map_ids] * (1 - rad**2)

                # scale parabolic flow profile to preserve mean flow
                for i, ids in map_ids_inv.items():
                    u_mean_is = np.mean(u_quad[map_ids_inv[i]])
                    u_quad[ids] *= u_mean[i] / u_mean_is

                # parabolic velocity vector field
                velocity = np.outer(u_quad, np.ones(3)) * arrays_cent['CenterlineSectionNormal'][map_ids]

                # add to volume mesh
                if n == 'velocity':
                    aname = a
                elif n == 'flow':
                    aname = 'velocity'
                add_array(vol, aname, velocity)

    # write to file
    write_geo(f_out, vol)


def get_error(f_3d, f_1d, f_out):
    geo_3d = read_geo(f_3d).GetOutput()
    geo_1d = read_geo(f_1d).GetOutput()
    arrays_3d = collect_arrays(geo_3d.GetPointData())
    arrays_1d = collect_arrays(geo_1d.GetPointData())

    for m in arrays_1d.keys():
        if 'pressure' in m:
            norm = np.mean(arrays_3d[m])
            err = np.abs(arrays_3d[m] - arrays_1d[m]) / norm
            add_array(geo_1d, 'error_' + m, err)
        if 'velocity' in m:
            norm = np.mean(np.linalg.norm(arrays_3d[m], axis=1))
            err = np.linalg.norm(arrays_3d[m] - arrays_1d[m], axis=1) / norm
            add_array(geo_1d, 'error_' + m, err)
            pdb.set_trace()
    write_geo(f_out, geo_1d)


def main(db, geometries):
    for geo in geometries:
        print(geo)
        f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
        f_0d = db.get_0d_flow_path_vtp(geo)
        f_1d = db.get_1d_flow_path_vtp(geo)
        f_wall = db.get_surfaces(geo, 'wall')
        f_out = db.get_initial_conditions_pressure(geo) #'test.vtu'#
        pdb.set_trace()

        if os.path.exists(f_out):
            print('  projection exists, skipping')
            continue

        if os.path.exists(f_0d):
            print('  using 0d')
            f_red = f_0d
        elif os.path.exists(f_1d):
            print('  using 1d')
            f_red = f_1d
        else:
            print('  no 0d/1d solution found')
            continue

        # if os.path.exists(f_out):
        #     print('  map exists')
        #     continue

        project_1d_3d_grow(f_red, f_vol, f_wall, f_out)


def main_paper():
    db = Database('deformable')
    geo = '0069_0001'
    f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
    f_red = db.get_0d_flow_path_vtp(geo)
    f_wall = db.get_surfaces(geo, 'wall')

    for m in ['0d', '1d']:
        if m == '0d':
            f_red = db.get_0d_flow_path_vtp(geo)
        elif m == '1d':
            f_red = db.get_1d_flow_path_vtp(geo)
        f_out = f_red.replace('.vtp', '.vtu')
        project_1d_3d_grow(f_red, f_vol, f_wall, f_out)


def convert_time(db, geometries):
    for geo in geometries:
        f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
        f_res = db.get_volume(geo)
        f_red = db.get_3d_flow(geo)
        f_wall = db.get_surfaces(geo, 'wall')
        d_out = os.path.join('/home/pfaller/work/osmsc/extrapolation/', geo)
        f_out = os.path.join(d_out, geo + '_mapped.vtu')
        f_err = os.path.join(d_out, geo + '_error.vtu')

        os.makedirs(d_out, exist_ok=True)

        project_1d_3d_grow(f_red, f_vol, f_wall, f_out)

        shutil.copy(f_res, os.path.join(d_out, geo + '.vtu'))
        shutil.copy(f_red, d_out)

        get_error(f_res, f_out, f_err)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    # d, g, _ = input_args(descr)
    # main(d, g)
    main_paper()
    # convert_time(d, g)
    # plot_projection(d, g)
    # f_vol = '/home/pfaller/downloads/0069_0001_post-interv (1)/0069_0001_post-interv/Meshes/0069_0001.vtu'
    # f_1d = d.get_1d_flow_path_vtp('0069_0001')
    # f_wall = '/home/pfaller/downloads/0069_0001_post-interv (1)/0069_0001_post-interv/Meshes/wall.vtp'
    # f_out = '/home/pfaller/0069_0001_interv_ic.vtu'
    # project_1d_3d_grow(f_1d, f_vol, f_wall, f_out)
