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

from common import get_dict
from get_database import Database, Post, input_args
from vtk_functions import read_geo, write_geo, collect_arrays, get_all_arrays, ClosestPoints
from get_bc_integrals import get_res_names
from vtk_to_xdmf import write_xdmf
from simulation_io import get_caps_db, collect_results_db, collect_results_db_3d_3d, \
    collect_results_db_3d_3d_spatial

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

# interpolate 1d to 3d in space and time (allow extrapolation due to round-off errors at bounds)
interp = lambda x_1d, y_1d, x_3d: interp1d(x_1d, y_1d.T, fill_value='extrapolate')(x_3d)

f = 'pressure'
out_path = '/home/pfaller/work/osmsc/cover/'

def main(db, geometries):
    # get post-processing constants
    post = Post()

    for geo in geometries:
        # generate plots
        print(geo)

        # read results
        deform = 'deformable' in db.study
        res, time = collect_results_db(db, geo, post.models, deformable=deform)

        # name of reference solution
        m_ref = '3d_rerun'
        roms = ['0d', '1d']

        # times
        t_ref = time[m_ref]

        res_all = defaultdict(list)

        err = []
        for m_rom in roms:
            # reference time
            t_rom = time[m_rom]

            for br in list(res.keys()):
                # retrieve 3d results
                res_ref = res[br][f][m_ref + '_int_last']

                # map paths to interval [0, 1]
                path_rom = res[br][m_rom + '_path'] / res[br][m_rom + '_path'][-1]
                path_ref = res[br][m_ref + '_path'] / res[br][m_ref + '_path'][-1]

                # interpolate in space and time
                res_rom = interp(path_rom, res[br][f][m_rom + '_int_last'], path_ref)
                res_rom = interp(t_rom, res_rom, t_ref)

                if m_rom == '0d':
                    # add 3d results only once in loop
                    res_all[m_ref] += [res_ref]

                    # error (use 0d since generally worse than 1d)
                    err += [np.linalg.norm(res_ref - res_rom, axis=0)]

                # add rom results
                res_all[m_rom] += [res_rom]

        # error over all time steps
        err_t = np.linalg.norm(np.array(err), axis=0)

        # find time step with smallest error
        i_min = np.argmin(err_t)
        t_min = t_ref[i_min]

        # all results at time step with smallest error
        res_min = defaultdict(list)
        for m in roms:
            for br in list(res.keys()):
                res_min[m] += [res_all[m][br][:, i_min]]

        # generate "straight" centerline
        cent = straighten(read_geo(db.get_centerline_path(geo)).GetOutput(), res)

        # export for plotting
        i_3d = np.where(time['3d_rerun_last_cycle_i'])[0][i_min]
        for m in roms:
            write_rom(db, geo, cent, res_min, m)
        write_3d(db, geo, i_3d)

def write_3d(db, geo, i_3d):
    # read 3d results
    res_3d = read_geo('/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '.vtu').GetOutput()

    # read 3d mesh
    geo_3d = read_geo(db.get_volume_mesh(geo)).GetOutput()

    # get results at 3d time step
    name = f + '_' + str(db.get_3d_increment(geo) * i_3d).zfill(5)

    # get results
    array = n2v(v2n(res_3d.GetPointData().GetArray(name)) * Post().convert[f])
    array.SetName(f)

    # export
    geo_3d.GetPointData().AddArray(array)
    write_geo(out_path + '/3d/' + geo + '.vtu', geo_3d)

def write_rom(db, geo, cent, res, m_rom):
    # assign results
    branches = v2n(cent.GetPointData().GetArray('BranchId'))
    res_cent = np.zeros(cent.GetNumberOfPoints())
    for br, res_br in enumerate(res[m_rom]):
        res_cent[branches == br] = res_br * Post().convert[f]

    # set results
    array = n2v(res_cent)
    array.SetName(f)
    cent.GetPointData().AddArray(array)

    # export
    write_geo(out_path + '/' + m_rom + '/' + geo + '.vtp', cent)

def straighten(cent, res, m_rom='0d'):
    # read centerline arrays
    branches = v2n(cent.GetPointData().GetArray('BranchId'))
    area = v2n(cent.GetPointData().GetArray('CenterlineSectionArea'))
    path = v2n(cent.GetPointData().GetArray('Path'))
    points = v2n(cent.GetPoints().GetData())

    # new "linearized" arrays
    area_lin = np.zeros(cent.GetNumberOfPoints())
    points_lin = np.zeros((cent.GetNumberOfPoints(), 3))
    for br in range(np.max(v2n(cent.GetPointData().GetArray('BranchId'))) + 1):
        # all branch nodes
        i_br = branches == br
        path_br = path[i_br]
        area_br = area[i_br]
        points_br = points[i_br]

        # get branch segments
        path_rom = res[br][m_rom + '_path']

        # loop all branch segments
        for i in range(len(path_rom) - 1):
            # first and last point in segment
            p0 = np.argmin(np.abs(path_br - path_rom[i]))
            p1 = np.argmin(np.abs(path_br - path_rom[i + 1]))

            # segment points
            i_seg = np.where(i_br)[0][p0:p1+1]

            # segments for interpolation
            dist_seg = np.array([0.0, 1.0])
            path_seg = path_br[p0:p1+1].copy()
            path_seg -= path_seg[0]
            path_seg /= path_seg[-1]

            # linearly interpolate areas
            area_seg = np.array([area_br[p0], area_br[p1]])
            area_lin[i_seg] = interp(dist_seg, area_seg, path_seg)

            # linearly interpolate points
            points_seg = np.vstack((points_br[p0], points_br[p1]))
            points_lin[i_seg] = interp(dist_seg, points_seg, path_seg).T

    # add radius
    array = n2v(np.sqrt(area_lin / np.pi))
    array.SetName('segment_radius')
    cent.GetPointData().AddArray(array)

    # replace points
    points = vtk.vtkPoints()
    points.Initialize()
    for p in points_lin:
        points.InsertNextPoint(p)
    cent.SetPoints(points)

    return cent



if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
