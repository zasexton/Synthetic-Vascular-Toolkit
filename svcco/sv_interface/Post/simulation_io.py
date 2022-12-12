#!/usr/bin/env python
# coding=utf-8

import argparse
import glob
import os
import csv
import re
import sys
import scipy
import pdb
import vtk
from scipy.interpolate import interp1d

import numpy as np
from collections import defaultdict, OrderedDict

from .common import get_dict
from .get_database import input_args
from .vtk_functions import read_geo, write_geo, collect_arrays, get_all_arrays, ClosestPoints
from .get_bc_integrals import get_res_names
from .vtk_to_xdmf import write_xdmf

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n


def map_meshes(nd_id_src, nd_id_trg):
    """
    Map source mesh to target mesh
    """
    index = np.argsort(nd_id_src)
    search = np.searchsorted(nd_id_src[index], nd_id_trg)
    return index[search]


def read_hist(db, geo):
    # read from text files
    d_hist = '/home/pfaller/work/osmsc/studies/ini_1d_quad/hist/0186_0002'
    res_bc = defaultdict(dict)
    for f_name in os.listdir(d_hist):
        if 'Hist' in f_name:
            if f_name[0] == 'P':
                f = 'pressure'
            elif f_name[0] == 'Q':
                f = 'velocity'
            if 'COR' in f_name:
                t = 'coronary'
            elif 'RCR' in f_name:
                t = 'rcr'
            res_bc[f][t] = np.loadtxt(os.path.join(d_hist, f_name), skiprows=1)

    # read centerline
    cent = read_geo(db.get_centerline_path(geo)).GetOutput()
    n_point = cent.GetNumberOfPoints()
    arrays_cent, _ = get_all_arrays(cent)
    branches = arrays_cent['BranchId']

    # get time step
    nt_out = db.get_3d_increment(geo)

    # get outlets
    caps = get_caps_db(db, geo)
    del caps['inflow']
    bct = db.get_bcs(geo)['bc_type']

    # assign results to centerline
    arrays = defaultdict(lambda: defaultdict(lambda: np.zeros(n_point)))
    counter = defaultdict(int)
    for c, br in caps.items():
        # get bc type
        t = bct[c]

        # last point in branch is outlet
        i_c = np.where(branches == br)[0][-1]

        # loop result fields
        for f in res_bc.keys():
            # extract results for this outlet
            res = res_bc[f][t]
            if len(res.shape) == 2:
                res = res_bc[f][t].T[counter[t]]

            # loop all time steps and assign outlet value
            for i in np.arange(0, len(res), nt_out):
                arrays[f][f + '_' + str(i).zfill(5)][i_c] = res[i]
        counter[t] += 1

    # add to centerline
    for arr in arrays.values():
        for n, a in arr.items():
            out_array = n2v(a)
            out_array.SetName(n)
            cent.GetPointData().AddArray(out_array)

    # add empty area
    out_array = n2v(np.zeros(n_point))
    out_array.SetName('area')
    cent.GetPointData().AddArray(out_array)

    # write to file
    f_out = os.path.join(d_hist, geo + '.vtp')
    write_geo(f_out, cent)


def read_results_0d(fpath):
    """
    Read 0d simulation results from dictionary
    """
    return get_dict(fpath)


def read_results_1d(res_dir, params_file=None):
    """
    Read results from oneDSolver and store in dictionary
    Args:
        res_dir: directory containing 1D results
        params_file: optional, path to dictionary of oneDSolver input parameters

    Returns:
    Dictionary sorted as [result field][segment id][time step]
    """
    # requested output fields
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']

    # read 1D simulation results
    results_1d = {}
    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(res_dir, '*branch*seg*_' + field + '.dat'))

        # loop segments
        results_1d[field] = defaultdict(dict)
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')

                # loop nodes
                results_1d_f = []
                for line in reader:
                    results_1d_f.append([float(l) for l in line if l][1:])

            # store results and GroupId
            seg = int(re.findall(r'\d+', f_res)[-1])
            branch = int(re.findall(r'\d+', f_res)[-2])
            results_1d[field][branch][seg] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d


def write_results(f_out, cent, arrays, only_last=True):
    """
    Write results to vtp file
    """
    # export last time step (= initial conditions)
    if only_last:
        for f, a in arrays[0]['point'].items():
            out_array = n2v(a)
            out_array.SetName(f)
            cent.GetPointData().AddArray(out_array)
    # export all time steps
    else:
        for t in arrays.keys():
            for f in arrays[t]['point'].keys():
                out_array = n2v(arrays[t]['point'][f])
                out_array.SetName(f + '_' + t)
                cent.GetPointData().AddArray(out_array)

    # write to file
    write_geo(f_out, cent)


def map_rom_to_centerline(rom, geo_cent, res, time, only_last=True):
    """
    Map 0d or 1d results to centerline
    """
    # assemble output dict
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()

    # get centerline arrays
    arrays_cent, _ = get_all_arrays(geo_cent)

    # centerline points
    points = v2n(geo_cent.GetPoints().GetData())

    # pick results
    if only_last:
        name = rom + '_int_last'
        t_vec = time[rom]
    else:
        name = rom + '_int'
        t_vec = time[rom + '_all']

    # loop all result fields
    for f in res[0].keys():
        if 'path' in f:
            continue
        array_f = np.zeros((arrays_cent['Path'].shape[0], len(t_vec)))
        n_outlet = np.zeros(arrays_cent['Path'].shape[0])
        for br in res.keys():
            # get centerline path
            path_cent = arrays_cent['Path'][arrays_cent['BranchId'] == br]
            path_cent /= path_cent[-1]

            # get 0d path
            path_0d = res[br][rom + '_path']
            path_0d /= path_0d[-1]

            # linearly interpolate results along centerline
            f_cent = interp1d(path_0d, res[br][f][name].T)(path_cent).T

            # store in global array
            array_f[arrays_cent['BranchId'] == br] = f_cent

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(arrays_cent['BranchId'] == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(arrays_cent['CenterlineId'][ip])[0][0]

            # id of upstream junction
            jc = arrays_cent['BifurcationId'][ip - 1]

            # centerline within junction
            jc_cent = np.where(np.logical_and(arrays_cent['BifurcationId'] == jc,
                                              arrays_cent['CenterlineId'][:, cid]))[0]

            # length of centerline within junction
            jc_path = np.append(0, np.cumsum(np.linalg.norm(np.diff(points[jc_cent], axis=0), axis=1)))
            jc_path /= jc_path[-1]

            # results at upstream branch
            res_br_u = res[arrays_cent['BranchId'][jc_cent[0] - 1]][f][name]

            # results at beginning and end of centerline within junction
            f0 = res_br_u[-1]
            f1 = res[br][f][name][0]

            # map 1d results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T, fill_value='extrapolate')(jc_path).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1

            # normalize by number of outlets
        array_f[n_outlet > 0] = (array_f[n_outlet > 0].T / n_outlet[n_outlet > 0]).T

        # assemble time steps
        if only_last:
            arrays[0]['point'][f] = array_f[:, -1]
        else:
            for i, t in enumerate(t_vec):
                arrays[str(t)]['point'][f] = array_f[:, i]

    return arrays


def load_results_3d(f_res_3d):
    """
    Read 3d results embedded in centerline and sort according to branch at time step
    """
    # read 1d geometry
    reader = read_geo(f_res_3d).GetOutput()
    res = collect_arrays(reader.GetPointData())

    # names of output arrays
    res_names = get_res_names(reader, ['pressure', 'velocity'])

    # get time steps
    has_time = np.all(['_' in k for k in res_names])

    if has_time:
        times = np.unique([float(k.split('_')[1]) for k in res_names])
    else:
        times = np.zeros(1)

    # get branch ids
    branches = np.unique(res['BranchId']).tolist()
    if -1 in branches:
        branches.remove(-1)

    # add time
    out = {'time': times}

    # initilize output arrays [time step, branch]
    for f in res_names:
        if has_time:
            name = f.split('_')[0]
        else:
            name = f
        out[name] = {}
        for br in branches:
            ids = res['BranchId'] == br
            out[name][br] = np.zeros((times.shape[0], np.sum(ids)))

    # read branch-wise results from geometry
    for f in res_names:
        if has_time:
            name, time = f.split('_')
        else:
            name, time = f, 0.0
        for br in branches:
            ids = res['BranchId'] == br
            out[name][br][float(time) == times] = res[f][ids]

    # add area (identical for all time steps)
    out['area'] = {}
    for br in branches:
        ids = res['BranchId'] == br
        out['area'][br] = np.tile(res['area'][ids], (times.shape[0], 1))

    # rename velocity to flow
    try:
        out['flow'] = out['velocity']
    except:
        raise RuntimeError('No results in file ' + f_res_3d)
    del out['velocity']

    return out


def get_time(model, res, time, dt_3d=0, nt_3d=0, ns_3d=0, t_in=0):
    if '3d_rerun' in model:
        time[model + '_all'] = res['time'] * dt_3d
    elif '3d' in model:
        time[model] = np.array([0] + res['time'].tolist())
        time[model + '_all'] = time[model]
    elif '1d' in model:
        dt = 1e-3
        time[model + '_all'] = np.arange(0, res['pressure'][0][0].shape[1] + 1)[1:] * dt
        time[model + '_all'] = np.append(0, time[model + '_all'])
    elif '0d' in model:
        time[model + '_all'] = res['time']
    else:
        raise RuntimeError('Unknown model ' + model)

    # time steps for last cycle
    if not model == '3d':
        # how many full cycles where completed?
        n_cycle = max(1, int(time[model + '_all'][-1] // t_in))
        time[model + '_n_cycle'] = n_cycle

        # first and last time step in cycle
        t_end = t_in
        t_first = t_end * (n_cycle - 1)
        t_last = t_end * n_cycle

        # tolerance (<< time step * numstep) to prevent errors due to time step round-off
        eps = 1.0e-3

        # select last cycle and shift time to start from zero
        try:
            time[model + '_last_cycle_i'] = np.logical_and(time[model + '_all'] >= t_first - eps, time[model + '_all'] <= t_last + eps)
            time[model] = time[model + '_all'][time[model + '_last_cycle_i']] - t_first
        except:
            pdb.set_trace()
        cycle_range = []
        for i in np.arange(1, n_cycle + 1):
            t_first = t_end * (i - 1)
            t_last = t_end * i
            bound0 = time[model + '_all'] >= t_first - eps
            bound1 = time[model + '_all'] <= t_last + eps
            time[model + '_i_cycle_' + str(i)] = np.logical_and(bound0, bound1)
            time[model + '_cycle_' + str(i)] = time[model + '_all'][time[model + '_i_cycle_' + str(i)]] - t_first
            cycle_range += [np.where(time[model + '_i_cycle_' + str(i)])[0]]
        time[model + '_cycles'] = np.array(cycle_range, dtype=object)
    # elif '3d_rerun' in model:
    #     time_steps = res['time'].astype(int)
    #     pdb.set_trace()



def check_consistency(r_oned, res_1d, res_3d):
    n_br_res_1d = len(res_1d['area'].keys())
    n_br_res_3d = len(res_3d['area'].keys())
    n_br_geo_1d = np.unique(v2n(r_oned.GetOutput().GetPointData().GetArray('BranchId'))).shape[0]

    if n_br_res_1d != n_br_res_3d:
        return '1d and 3d results incosistent'

    if r_oned.GetNumberOfCells() + n_br_geo_1d != r_oned.GetNumberOfPoints():
        return '1d model connectivity inconsistent'

    return None


def get_branches(arrays):
    """
    Get list of branch IDs from point arrays
    """
    branches = np.unique(arrays['BranchId']).astype(int).tolist()
    if -1 in branches:
        branches.remove(-1)
    return branches


def get_caps_db(db, geo, f_surf=None):
    """
    Get caps for OSMSC models
    """
    return get_caps(db.get_centerline_outlet_path(geo), db.get_centerline_path(geo), f_surf)


def get_caps(f_outlet, f_centerline, f_surf=None):
    """
    Map outlet names to centerline branch id
    Args:
        f_outlet: ordered list of outlet names (created during centerline extraction)
        f_centerline: centerline geometry (.vtp)

    Returns:
        dictionary {cap name: BranchId}
    """
    caps = OrderedDict()
    caps['inflow'] = 0

    # read ordered outlet names from file
    outlet_names = []
    if not os.path.exists(f_outlet):
        return None
    with open(f_outlet) as file:
        for line in file:
            outlet_names += line.splitlines()

    # read centerline
    cent = read_geo(f_centerline).GetOutput()
    if not cent.GetPointData().HasArray('BranchId'):
        raise RuntimeError('centerline branch extraction failed')
    branch_id = v2n(cent.GetPointData().GetArray('BranchId'))

    # find outlets and store outlet name and BranchId
    ids = vtk.vtkIdList()
    i_outlet = 0

    # closest surface points
    if f_surf:
        # transfer surface ids
        surf = read_geo(f_surf).GetOutput()
        cell_to_point = vtk.vtkCellDataToPointData()
        cell_to_point.SetInputData(surf)
        cell_to_point.Update()
        face_id = v2n(cell_to_point.GetOutput().GetPointData().GetArray('BC_FaceID'))
        cp = ClosestPoints(f_surf)
        br_to_bcface = OrderedDict()
        br_to_bcface[0] = face_id[cp.search([list(cent.GetPoint(0))])[0]]

    # loop all centerline points
    for i in range(1, cent.GetNumberOfPoints()):
        cent.GetPointCells(i, ids)

        # check if cap
        if ids.GetNumberOfIds() == 1:
            # this works since the points are numbered according to the order of outlets
            caps[outlet_names[i_outlet]] = branch_id[i]

            # find closest surface point
            if f_surf:
                i_point = cp.search([list(cent.GetPoint(i))])[0]
                br_to_bcface[branch_id[i]] = face_id[i_point]

            i_outlet += 1

    if f_surf:
        return caps, br_to_bcface
    else:
        return caps


def res_1d_to_path(path, res):
    path_1d = []
    int_1d = []
    for seg, res_1d_seg in sorted(res.items()):
        # 1d results are duplicate at FE-nodes at corners of segments
        if seg == 0:
            # start with first FE-node
            i_start = 0
        else:
            # skip first FE-node (equal to last FE-node of previous segment)
            i_start = 1

        # generate path for segment FEs, assuming equidistant spacing
        p0 = path[seg]
        p1 = path[seg + 1]
        path_1d += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
        int_1d += res_1d_seg[i_start:].tolist()

    return np.array(path_1d), np.array(int_1d)


def collect_results(model, res, time, f_res, centerline=None, dt_3d=0, nt_3d=0, ns_3d=0, t_in=0, caps=None):
    # read results
    # todo: store 1d results in vtp as well
    if '0d' in model:
        res_in = read_results_0d(f_res)
        f_geo = centerline
        if res_in['time'][0] > 0:
            print('truncating results')
            i_start = np.argmin(np.abs(res_in['time'] - t_in))

            # truncate time
            for f in res_in.keys():
                if f == 'time':
                    res_in[f] = res_in[f][i_start:] - res_in[f][i_start]
                else:
                    for br in res_in[f].keys():
                        for n in res_in[f][br].keys():
                            res_in[f][br][n] = res_in[f][br][n][i_start:]
    elif '1d' in model:
        res_in = get_dict(f_res)
        f_geo = centerline
    elif '3d' in model:
        res_in = load_results_3d(f_res)
        f_geo = f_res
    else:
        raise ValueError('Model ' + model + ' not recognized')

    # read geometry
    geo = read_geo(f_geo)

    # extract point and cell arrays from geometry
    arrays, _ = get_all_arrays(geo.GetOutput())

    # get branches
    branches = get_branches(arrays)

    # simulation time steps
    get_time(model, res_in, time, dt_3d=dt_3d, nt_3d=nt_3d, ns_3d=ns_3d, t_in=t_in)

    # loop outlets
    for br in branches:
        # 1d-path along branch (real length units)
        branch_path = arrays['Path'][arrays['BranchId'] == br]

        # loop result fields
        for f in ['flow', 'pressure', 'area']:
            if '0d' in model:
                if f == 'area':
                    res[br][f]['0d_int'] = np.zeros(res_in['flow'][br].shape)
                else:
                    res[br][f]['0d_int'] = res_in[f][br]
                    res[br]['0d_path'] = res_in['distance'][br]
            elif '1d' in model:
                res[br]['1d_path'], res[br][f]['1d_int'] = res_1d_to_path(branch_path, res_in[f][br])
                if res[br][f]['1d_int'].shape[1] + 1 == time['1d_all'].shape[0]:
                    res[br][f]['1d_int'] = np.hstack((np.zeros((res[br][f]['1d_int'].shape[0], 1)), res[br][f]['1d_int']))
            elif '3d' in model:
                res[br][model + '_path'] = branch_path
                res[br][f][model + '_int'] = res_in[f][br].T

            # copy last time step at t=0
            if model == '3d':
                res[br][f][model + '_int'] = np.tile(res[br][f][model + '_int'], (1, 2))[:,
                                             res[br][f][model + '_int'].shape[1] - 1:]

            if br == 0:
                # inlet
                i_cap = 0
            else:
                # outlet
                i_cap = -1

            # extract cap results
            res[br][f][model + '_cap'] = res[br][f][model + '_int'][i_cap, :]

    # get last cycle
    for br in res.keys():
        for f in res[br].keys():
            if 'path' not in f:
                res[br][f][model + '_all'] = res[br][f][model + '_cap']

                if model + '_last_cycle_i' in time and len(time[model + '_last_cycle_i']) > 1:
                    res[br][f][model + '_int_last'] = res[br][f][model + '_int'][:, time[model + '_last_cycle_i']]
                    res[br][f][model + '_cap_last'] = res[br][f][model + '_cap'][time[model + '_last_cycle_i']]
                elif model == '3d':
                    res[br][f][model + '_int_last'] = res[br][f][model + '_int']
                    res[br][f][model + '_cap_last'] = res[br][f][model + '_cap']


def collect_results_spatial(model, res, time, f_res, dt_3d=0, t_in=0):
    geo = read_geo(f_res).GetOutput()

    # fields to export
    fields = ['pressure', 'velocity']

    # get all result array names
    res_names = get_res_names(geo, fields)

    # extract all point arrays
    arrays, _ = get_all_arrays(geo)

    # sort results according to GlobalNodeID
    mask = map_meshes(arrays['GlobalNodeID'], np.arange(1, geo.GetNumberOfPoints() + 1))

    # get time steps
    times = np.unique([float(k.split('_')[1]) for k in res_names])

    # simulation time steps
    get_time(model, {'time': times}, time, dt_3d, t_in)

    # initialize results
    res[model]['pressure'] = np.zeros((times.shape[0], geo.GetNumberOfPoints()))
    res[model]['velocity'] = np.zeros((times.shape[0], geo.GetNumberOfPoints(), 3))

    # extract results
    for f in res_names:
        n, t = f.split('_')
        res[model][n][float(t) == times] = arrays[f][mask]

    # extract periodic cycle
    # if model + '_last_cycle_i' in time:
    #     for n in fields:
    #         res[model][n] = res[model][n][time[model + '_last_cycle_i']]


def collect_results_db_0d(db, geo):
    f_res_0d = db.get_0d_flow_path(geo)
    f_oned = db.get_1d_geo(geo)

    if not os.path.exists(f_res_0d):
        return None, None

    time_inflow, _ = db.get_inflow_smooth(geo)

    res = defaultdict(lambda: defaultdict(dict))
    time = {}
    collect_results('0d', res, time, f_res_0d, centerline=f_oned, t_in=time_inflow[-1])

    return res, time


def collect_results_db_3d(db, geo, m):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    if m == '3d':
        # get paths
        f_res_3d_osmsc = db.get_3d_flow(geo)
        if not os.path.exists(f_res_3d_osmsc):
            return None, None

        # collect osmsc results
        collect_results('3d', res, time, f_res_3d_osmsc)
    elif m == '3d_rerun':
        f_res_3d_rerun = db.get_3d_flow_rerun(geo)
        if not os.path.exists(f_res_3d_rerun):
            return None, None

        time_inflow, _ = db.get_inflow_smooth(geo)
        if time_inflow is None:
            return None, None

        # collect rerun results
        collect_results('3d_rerun', res, time, f_res_3d_rerun, dt_3d=db.get_3d_timestep(geo),
                        nt_3d=db.get_3d_increment(geo), ns_3d=db.get_3d_numstep(geo), t_in=time_inflow[-1])

    return res, time


def collect_results_db(db, geo, models, deformable=False):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_0d = db.get_0d_flow_path(geo)
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow(geo)
    f_oned = db.get_1d_geo(geo)
    f_cent = db.get_centerline_path(geo)

    # get paths for 3d models
    if deformable:
        f_res_3d_rerun = ['/home/pfaller/work/osmsc/studies/deformable/3d_flow/' + geo + '.vtp']
    else:
        f_res_3d_rerun = ['/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '.vtp',
                          '/home/pfaller/work/osmsc/studies/ini_zero/3d_flow/' + geo + '.vtp']

    time_inflow, _ = db.get_inflow_smooth(geo)

    # collect results
    if '3d_rerun' in models:
        for f_rerun in f_res_3d_rerun:
            if os.path.exists(f_rerun):
                collect_results('3d_rerun', res, time, f_rerun, t_in=time_inflow[-1], dt_3d=db.get_3d_timestep(geo), ns_3d=db.get_3d_numstep(geo))
                break
    if '3d' in models and os.path.exists(f_res_3d):
        collect_results('3d', res, time, f_res_3d)
    if '1d' in models and os.path.exists(f_res_1d):
        collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])
    if '0d' in models and os.path.exists(f_res_0d):
        collect_results('0d', res, time, f_res_0d, centerline=f_cent, t_in=time_inflow[-1])

    return res, time


def collect_results_db_3d_3d(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_3d_osmsc = db.get_3d_flow(geo)
    if not os.path.exists(f_res_3d_osmsc):
        return None, None

    # collect osmsc results
    collect_results('3d', res, time, f_res_3d_osmsc)

    f_res_3d_rerun = db.get_3d_flow_rerun(geo)
    if not os.path.exists(f_res_3d_rerun):
        return res, time

    time_inflow, _ = db.get_inflow_smooth(geo)
    if time_inflow is None:
        return res, time

    # collect rerun results
    collect_results('3d_rerun', res, time, f_res_3d_rerun, dt_3d=db.get_3d_timestep(geo),
                    nt_3d=db.get_3d_increment(geo), ns_3d=db.get_3d_numstep(geo), t_in=time_inflow[-1])

    return res, time


def collect_results_db_3d_3d_spatial(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_3d_osmsc = db.get_volume(geo)
    f_res_3d_rerun = db.get_res_3d_vol_rerun(geo)

    if not os.path.exists(f_res_3d_osmsc) or not os.path.exists(f_res_3d_rerun):
        return None, None

    time_inflow, _ = db.get_inflow_smooth(geo)

    if time_inflow is None:
        return None, None

    # collect results
    collect_results_spatial('3d', res, time, f_res_3d_osmsc)
    collect_results_spatial('3d_rerun', res, time, f_res_3d_rerun, dt_3d=db.get_3d_timestep(geo), t_in=time_inflow[-1])

    return res, time


def export_last(db, geo):
    f_res_1d = db.get_1d_flow_path(geo)

    time_inflow, _ = db.get_inflow_smooth(geo)

    # read results
    res = get_dict(f_res_1d)

    # get time information
    time = {}
    get_time('1d', res, time, t_in=time_inflow[-1])

    res_out = {'time': time['1d']}
    for f in res.keys():
        if f == 'params':
            continue
        res_out[f] = {}
        for br in res[f].keys():
            res_out[f][br] = {}
            for seg in res[f][br].keys():
                res_out[f][br][seg] = res[f][br][seg][:, time['1d_last_cycle_i']]
    np.save(db.gen_file('1d_flow_last', geo), res_out)


def export_rom_vtp_db(db, geo, model, only_last=True):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_0d = db.get_0d_flow_path(geo)
    f_res_1d = db.get_1d_flow_path(geo)
    f_oned = db.get_1d_geo(geo)
    f_cent = db.get_centerline_path(geo)
    cent = read_geo(f_cent).GetOutput()

    # collect results
    time_inflow, _ = db.get_inflow_smooth(geo)
    if '1d' == model and os.path.exists(f_res_1d):
        collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])
        f_out = db.get_1d_flow_path_vtp(geo, only_last=only_last)
    elif '0d' == model and os.path.exists(f_res_0d):
        collect_results('0d', res, time, f_res_0d, centerline=f_cent, t_in=time_inflow[-1])
        f_out = db.get_0d_flow_path_vtp(geo, only_last=only_last)
    else:
        return

    arrays = map_rom_to_centerline(model, cent, res, time, only_last=only_last)
    pdb.set_trace()
    write_results(f_out, cent, arrays, only_last=only_last)


def main(db, geometries):
    for geo in geometries:
        print('Processing ' + geo)

        # read_hist(db, geo)
        if not os.path.exists(db.get_0d_flow_path(geo)):
            continue

        for m in ['0d', '1d']:
            export_rom_vtp_db(db, geo, m, only_last=True)
        # export_last(db, geo)


def main_cover(db, geometries):
    for geo in geometries:
        print('Processing ' + geo)

        for m in ['0d', '1d']:
            export_rom_vtp_db(db, geo, m, only_last=True)
        # export_last(db, geo)


if __name__ == '__main__':
    descr = 'Retrieve simulation results'
    d, g, _ = input_args(descr)
    main(d, g)
