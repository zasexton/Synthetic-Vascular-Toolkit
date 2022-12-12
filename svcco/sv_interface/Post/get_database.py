#!/usr/bin/env python

import os
import sys
import shutil
import glob
import subprocess
import csv
import re
import argparse
import pdb

import numpy as np
from collections import OrderedDict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from .get_bcs import get_bcs, get_params
from .vtk_functions import read_geo
from .common import get_dict
from .common import coronary_sv_to_oned


def input_args(description):
    """
    Handles input arguments to scripts
    Args:
        description: script description (hgelp string)

    Returns:
        database: Database object for study
        geometries: list of geometries to evaluate
    """
    # parse input arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('study', help='study name')
    parser.add_argument('-g', '--geo', help='individual geometry or subset name')
    parser.add_argument('-m', '--mode', help='select mode')
    param = parser.parse_args()

    # get model database
    database = Database(param.study)

    # choose geometries to evaluate
    if param.geo in database.get_geometries():
        geometries = [param.geo]
    elif param.geo is None:
        geometries = database.get_geometries()
    elif param.geo == '-1':
        geometries = reversed(database.get_geometries())
    elif param.geo[-1] == ':':
        geo_all = database.get_geometries()
        geo_first = geo_all.index(param.geo[:-1])
        geometries = geo_all[geo_first:]
    elif param.geo[-3:] == ':-1':
        geo_all = database.get_geometries()
        geo_all.reverse()
        geo_first = geo_all.index(param.geo[:-3])
        geometries = geo_all[geo_first:]
    else:
        geometries = database.get_geometries_select(param.geo)

    return database, geometries, param


class Database:
    def __init__(self, study=''):
        # study name, if any
        self.study = study

        # path to database
        self.db_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'database')

        # folder for simulation files
        self.fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

        # folder where generated data is saved
        self.fpath_gen = '/home/pfaller/work/osmsc/data_generated'

        # folder for paths and segmentations
        self.fpath_seg_path = '/home/pfaller/work/osmsc/data_additional/models/'

        # folder for extras
        self.fpath_extras = '/home/pfaller/work/osmsc/data_additional/sim_files_extra/'

        # folder for simulation studies
        self.fpath_studies = '/home/pfaller/work/osmsc/studies'

        # folder containing model images
        self.fpath_png = os.path.join(self.db_path, 'png')

        # file containing simulation parameters
        self.db_params = os.path.join(self.db_path, 'parameters.npy')

        # folder for simulation studies
        self.fpath_study = os.path.join(self.fpath_studies, self.study)

        # folder where simulation is run
        self.fpath_solve = os.path.join(self.fpath_study, 'simulation')

        # fields to extract
        self.res_fields = ['velocity', 'pressure']

        # svproject object
        self.svproj = SVProject()

    def is_excluded(self, geo):
        # excluded geometries by Nathan
        imaging = ['0001', '0020', '0044']
        animals = ['0066', '0067', '0068', '0069', '0070', '0071', '0072', '0073', '0074']
        single_vessel = ['0158', '0164', '0165']

        exclude_nathan = imaging + animals + single_vessel

        # # excluded models by martin (say rest but are exercise)
        # exclude_martin = ['0063_2001', '0064_2001', '0065_2001', '0075_2001', '0076_2001', '0080_1001', '0081_1001',
        #                   '0082_1001', '0083_2002', '0084_1001', '0086_1001', '0107_1001', '0111_1001']

        if geo[:4] in exclude_nathan:  # or geo in exclude_martin:
            return True
        else:
            return False

    def exclude_geometries(self, geometries):
        return [g for g in geometries if not self.is_excluded(g)]

    def get_geometries(self):
        return sorted(get_dict(self.db_params).keys())

    def get_geometries_select(self, name):
        # only geometries where a 0d AND a 3d_rerun solution exisis
        if name == 'paper':
            geometries = []
            res_0d = get_dict(self.get_log_file_0d())
            res_1d = get_dict(self.get_log_file_1d())
            for geo in sorted(list(res_0d.keys())):
                fpath1 = '/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '.vtp'
                fpath2 = '/home/pfaller/work/osmsc/studies/ini_zero/3d_flow/' + geo + '.vtp'
                # check if 3d results exist
                if os.path.exists(fpath1) or os.path.exists(fpath2):
                    # check if 0d AND 1d results exist
                    if res_0d[geo] == 'success' and res_1d[geo] == 'success':
                        geometries += [geo]
        elif name == 'published':
            geometries = np.loadtxt('geometries_paper.txt', dtype='str')

        elif name == 'coarctation':
            geometries = ['0066_0001', '0067_0001', '0068_0001', '0069_0001', '0070_0001', '0071_0001', '0072_0001',
                          '0073_0001', '0074_0001', '0106_0001', '0107_0001', '0111_0001', '0090_0001', '0091_0001',
                          '0092_0001', '0093_0001', '0094_0001', '0095_0001', '0101_0001', '0102_0001', '0103_0001',
                          '0104_0001', '0105_0001']
        elif name == 'fix_surf_id':
            geometries = ['0140_2001', '0144_1001', '0147_1001', '0160_6001', '0161_0001', '0162_3001', '0163_0001']
        elif name == 'fix_surf_discr':
            geometries = ['0069_0001', '0164_0001']
        elif name == 'fix_surf_displacement':
            geometries = ['0065_0001', '0065_1001', '0065_2001', '0065_3001', '0065_4001', '0078_0001', '0079_0001',
                          '0091_0001', '0091_2001', '0092_0001', '0108_0001', '0154_0001', '0154_1001', '0165_0001',
                          '0166_0001', '0183_1002', '0187_0002']
        elif name == 'fix_inlet_node':
            geometries = ['0080_0001', '0082_0001', '0083_2002', '0084_1001', '0088_1001', '0112_1001', '0134_0002']
        elif name == 'fix_surf_orientation':
            geometries = ['0069_0001']
        elif name == 'fix_coarse_inflow':
            geometries = ['0078_0001', '0079_0001', '0080_0001', '0108_0001', '0166_0001', '0167_0001', '0172_0001',
                          '0184_0001']
        elif name == 'bifurcation_outlet':
            geometries = ['0080_0001', '0082_0001', '0083_2002', '0084_1001', '0088_1001', '0112_1001', '0134_0002']
        elif name == 'bifurcation_inlet':
            geometries = ['0065_1001', '0076_1001', '0081_0001', '0081_1001', '0086_0001', '0086_1001', '0089_1001',
                          '0148_1001', '0155_0001', '0162_3001']
        elif name == 'for_aekaansh':
            geometries = ['0172_0001', '0173_1001', '0183_1002', '0187_0002']
        elif name == 'rerun':
            geometries = ['0002_0001', '0003_0001', '0006_0001', '0063_0001', '0063_1001', '0064_0001', '0064_1001',
                          '0065_0001', '0065_1001', '0075_0001', '0076_0001', '0076_1001', '0077_0001', '0077_1001',
                          '0078_0001', '0080_0001', '0081_0001', '0082_0001', '0086_0001', '0090_0001', '0091_0001',
                          '0092_0001', '0093_0001', '0094_0001', '0095_0001', '0096_0001', '0097_0001', '0098_0001',
                          '0099_0001', '0101_0001', '0102_0001', '0103_0001', '0104_0001', '0105_0001', '0106_0001',
                          '0107_0001', '0108_0001', '0111_0001', '0118_1000', '0125_0001', '0126_0001', '0129_0000',
                          '0130_0000', '0131_0000', '0134_0002', '0138_1001', '0139_1001', '0140_2001', '0141_1001',
                          '0144_1001', '0145_1001', '0146_1001', '0148_1001', '0149_1001', '0151_0001', '0154_0001',
                          '0155_0001', '0156_0001', '0160_6001', '0161_0001', '0162_3001', '0163_0001', '0166_0001',
                          '0167_0001', '0172_0001', '0173_1001', '0174_0000', '0176_0000', '0183_1002', '0184_0001',
                          '0185_0001', '0186_0002', '0187_0002', '0188_0001', '0189_0001']
        elif name == 'rerun_all':
            geometries = ['0001_0001', '0002_0001', '0003_0001', '0006_0001', '0063_0001', '0063_1001', '0064_0001',
                          '0064_1001', '0065_0001', '0065_1001', '0066_0001', '0067_0001', '0068_0001', '0069_0001',
                          '0070_0001', '0071_0001', '0072_0001', '0073_0001', '0074_0001', '0075_0001', '0075_1001',
                          '0076_0001', '0076_1001', '0077_0001', '0077_1001', '0078_0001', '0079_0001', '0080_0001',
                          '0081_0001', '0082_0001', '0083_2002', '0086_0001', '0088_1001', '0089_1001', '0090_0001',
                          '0091_0001', '0092_0001', '0093_0001', '0094_0001', '0095_0001', '0096_0001', '0097_0001',
                          '0098_0001', '0099_0001', '0101_0001', '0102_0001', '0103_0001', '0104_0001', '0105_0001',
                          '0106_0001', '0107_0001', '0108_0001', '0110_0001', '0111_0001', '0112_1001', '0118_1000',
                          '0125_0001', '0126_0001', '0129_0000', '0130_0000', '0131_0000', '0134_0002', '0138_1001',
                          '0139_1001', '0140_2001', '0141_1001', '0142_1001', '0144_1001', '0145_1001', '0146_1001',
                          '0147_1001', '0148_1001', '0149_1001', '0150_0001', '0151_0001', '0154_0001', '0155_0001',
                          '0156_0001', '0157_0000', '0160_6001', '0161_0001', '0162_3001', '0163_0001', '0165_0001',
                          '0166_0001', '0167_0001', '0172_0001', '0173_1001', '0174_0000', '0175_0000', '0176_0000',
                          '0183_1002', '0184_0001', '0185_0001', '0186_0002', '0187_0002', '0188_0001', '0189_0001']
        elif name == 'inflow_oscillation':
            geometries = ['0176_0000', '0162_3001', '0161_0001', '0154_0001', '0148_1001', '0144_1001', '0138_1001',
                          '0106_0001', '0077_1001', '0063_1001']
        elif name == 'inflow_nan':
            geometries = ['0174_0000', '0163_0001', '0160_6001', '0156_0001', '0155_0001', '0151_0001', '0149_0001',
                          '0146_1001', '0141_1001', '0139_1001', '0131_0000', '0129_0000', '0111_0001', '0065_1001',
                          '0064_1001', '0006_0001']
        elif name == 'fixed_dt':
            geometries = ['0068_0001', '0092_0001', '0099_0001']
        elif name == 'rerun_coronary':
            geometries = ['0173_1001', '0183_1002', '0186_0002', '0187_0002', '0189_0001']
        elif name == 'resistance':
            geometries = []
            for geo in self.get_geometries():
                bc_def = self.get_bcs(geo)
                if bc_def is None:
                    continue
                if 'resistance' in bc_def['bc_type'].values():
                    geometries += [geo]
        elif 'units' in name:
            geometries = []
            for geo in self.get_geometries():
                bc_def = self.get_bcs(geo)
                if bc_def is None:
                    continue
                _, part, unit = name.split('_')
                if part == 's' and bc_def['params']['sim_units'] == unit:
                    geometries += [geo]
                elif part == 'm' and bc_def['params']['model_units'] == unit:
                    geometries += [geo]
            print(geometries)
            for geo in geometries:
                bc_def = self.get_bcs(geo)

                print(bc_def['params']['sim_units'])
            sys.exit(1)
        elif name in ['aorta', 'aortofemoral', 'pulmonary', 'cerebrovascular', 'coronary']:
            geometries = []
            for geo in self.get_geometries():
                bc_def = self.get_bcs(geo)
                if bc_def is not None and bc_def['params']['deliverable_category'].lower() == name:
                    geometries += [geo]

        else:
            raise Exception('Unknown selection ' + name)
        return geometries

    def get_bcs_local(self, geo):
        # folder for tcl files with boundary conditions
        fpath_bc = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/repos_ready_cpm_scripts'

        # try two different offsets of tcl name vs geo name
        # todo: find out why there are several variants
        for o in [-1, 0]:
            tcl, tcl_bc = get_tcl_paths(fpath_bc, geo, o)
            if os.path.exists(tcl) and os.path.exists(tcl_bc):
                return get_bcs(tcl, tcl_bc)
        return None

    def get_bcs(self, geo):
        return get_dict(self.db_params)[geo]

    def has_loop(self, geo):
        # todo: find automatic way to check for loop
        loop = ['0001_0001', '0106_0001', '0188_0001']
        return geo in loop

    def get_png(self, geo):
        pretty = os.path.join(self.fpath_png, '../png_pretty', geo + '.png')
        sim = os.path.join(self.fpath_png, 'OSMSC' + geo + '_sim.png')
        vol = os.path.join(self.fpath_png, 'OSMSC' + geo + '_vol.png')
        if os.path.exists(pretty):
            return pretty
        else:
            if os.path.exists(sim):
                return sim
            else:
                return vol

    def get_img(self, geo):
        return exists(os.path.join(self.fpath_sim, geo, 'image_data', 'vti', 'OSMSC' + geo[:4] + '-cm.vti'))

    def get_json(self, geo):
        return os.path.join(self.fpath_gen, 'json', geo + '.json')

    def get_surface_dir(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces', geo)

    def get_sv_meshes(self, geo):
        fdir = os.path.join(self.fpath_gen, 'sv_meshes', geo)
        fdir_caps = os.path.join(fdir, 'caps')
        os.makedirs(fdir, exist_ok=True)
        os.makedirs(fdir_caps, exist_ok=True)
        return fdir

    def get_sv_surface(self, geo):
        return exists(os.path.join(self.get_sv_meshes(geo), geo + '.vtp'))

    def get_sv_surface_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_sv', geo + '.vtp')

    def get_bc_flow_path(self, geo):
        return os.path.join(self.db_path, 'bc_flow', geo + '.npy')

    def get_3d_flow(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow', geo + '.vtp')

    def get_3d_flow_rerun(self, geo):
        return self.gen_file('3d_flow', geo, 'vtp')

    def get_3d_flow_rerun_bc(self, geo):
        return self.gen_file('3d_flow', geo + '_bc')

    def get_sv_flow_path(self, geo, model):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['flow'], 'inflow_' + model + '.flow')

    def get_sv_flow_path_rel(self, geo, model):
        sim_dir = os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo)
        return os.path.relpath(self.get_sv_flow_path(geo, model), sim_dir)

    def get_centerline_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', geo + '.vtp')

    def get_centerline_vmtk_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_vmtk', geo + '.vtp')

    def get_centerline_outlet_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', 'outlets_' + geo)

    def get_surfaces_grouped_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped', geo + '.vtp')

    def get_surfaces_cut_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_cut', geo + '.vtu')

    def get_surfaces_grouped_path_oned(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped_oned', geo + '.vtp')

    def get_centerline_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_sections', geo + '.vtp')

    def get_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'sections', geo + '.vtp')

    def get_bifurcation_path(self, geo):
        return os.path.join(self.fpath_gen, 'bifurcation_pressure', geo + '.vtp')

    def get_initial_conditions(self, geo):
        return os.path.join(self.get_sv_meshes(geo), 'initial.vtu')

    def get_initial_conditions_pressure(self, geo):
        # return os.path.join(self.get_sv_meshes(geo), 'initial_pressure.vtu')
        return os.path.join(self.fpath_gen, 'initial_pressure', geo + '.vtu')

    def get_asymptotic(self, geo):
        return os.path.join(self.fpath_gen, 'asymptotic', geo + '.vtu')

    def get_initial_conditions_steady(self, geo):
        return os.path.join(self.fpath_gen, 'steady', geo + '.vtu')

    def get_initial_conditions_steady0(self, geo):
        return os.path.join(self.fpath_gen, 'steady0', geo + '.vtu')

    def get_initial_conditions_irene(self, geo):
        return os.path.join(self.fpath_gen, 'irene', geo + '.vtu')

    def get_sv_initial_conditions(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo, 'mesh-complete',
                            'initial.vtu')

    def get_bc_comparison_path(self, geo, m):
        return self.gen_file('0d_flow_from_3d', geo, 'png')
        # return os.path.join(self.fpath_gen, 'bcs_' + m, geo + '.png')

    def get_bc_0D_path(self, geo, m):
        return self.gen_file('0d_flow_from_3d', geo)
        # return os.path.join(self.fpath_gen, 'bcs_' + m, geo + '.npy')

    def gen_dir(self, name):
        fdir = os.path.join(self.fpath_study, name)
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def gen_file(self, name, geo, ext='npy'):
        fdir = self.gen_dir(name)
        return os.path.join(fdir, geo + '.' + ext)

    def get_0d_flow_path(self, geo):
        return self.gen_file('0d_flow', geo)

    def get_0d_flow_path_vtp(self, geo, only_last=True):
        if only_last:
            return self.gen_file('0d_flow', geo + '_last', 'vtp')
        else:
            return self.gen_file('0d_flow', geo, 'vtp')

    def get_1d_flow_path(self, geo):
        return self.gen_file('1d_flow', geo)

    def get_1d_flow_path_xdmf(self, geo):
        return self.gen_file('1d_flow', geo, 'xdmf')

    def get_1d_flow_path_vtp(self, geo, only_last=True):
        if only_last:
            return self.gen_file('1d_flow', geo + '_last', 'vtp')
        else:
            return self.gen_file('1d_flow', geo, 'vtp')

    def get_post_path(self, geo, name):
        return self.gen_file('1d_3d_comparison', geo + '_' + name, 'png')

    def get_groupid_path(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'outletface_groupid.dat')

    def get_statistics_dir(self):
        return self.gen_dir('statistics')

    def get_solve_dir(self, geo):
        fsolve = os.path.join(self.fpath_solve, geo)
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_0d(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '0d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_1d(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '1d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_3d(self, geo):
        # fsolve = os.path.join(self.get_solve_dir(geo), '3d')
        fsolve = os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo)
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_3d_perigee(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '3d_perigee')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_svproj_dir(self, geo):
        fdir = os.path.join(self.fpath_gen, 'svprojects', geo)
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def get_svproj_file(self, geo):
        fdir = self.get_svproj_dir(geo)
        return os.path.join(fdir, '.svproj')

    def get_svpre_file(self, geo, solver):
        name = geo
        if solver == 'perigee':
            name += '_perigee'
        return os.path.join(self.get_solve_dir_3d(geo), name + '.svpre')

    def get_solver_file(self, geo):
        return os.path.join(self.get_solve_dir_3d(geo), 'solver.inp')

    def get_svproj_mdl_file(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['models'], geo + '.mdl')

    def get_svproj_sjb_file(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo + '.sjb')

    def add_dict(self, dict_file, geo, add):
        dict_db = get_dict(dict_file)
        dict_db[geo] = add
        np.save(dict_file, dict_db)

    def get_log_file_1d(self):
        return os.path.join(self.fpath_solve, 'log_1d.npy')

    def get_log_file_0d(self):
        return os.path.join(self.fpath_solve, 'log_0d.npy')

    def get_bc_err_file(self, m):
        return os.path.join(self.fpath_gen, 'bc_err_' + m + '.npy')

    def add_log_file_1d(self, geo, log):
        self.add_dict(self.get_log_file_1d(), geo, log)

    def add_log_file_0d(self, geo, log):
        self.add_dict(self.get_log_file_0d(), geo, log)

    def add_bc_err(self, geo, m, log):
        self.add_dict(self.get_bc_err_file(m), geo, log)

    def get_1d_3d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '1d_3d_comparison.npy')

    def get_0d_1d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '0d_1d_comparison.npy')

    def add_0d_1d_comparison(self, geo, err):
        self.add_dict(self.get_0d_1d_comparison(), geo, err)

    def add_1d_3d_comparison(self, geo, err):
        self.add_dict(self.get_1d_3d_comparison(), geo, err)

    def get_0d_3d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '0d_3d_comparison.npy')

    def add_0d_3d_comparison(self, geo, err):
        self.add_dict(self.get_0d_3d_comparison(), geo, err)

    def get_3d_3d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '3d_3d_comparison.npy')

    def get_convergence_path(self):
        return os.path.join(self.gen_file('statistics', '', ''), 'convergence.npy')

    def add_3d_3d_comparison(self, geo, err):
        self.add_dict(self.get_3d_3d_comparison(), geo, err)

    def add_convergence(self, geo, err):
        self.add_dict(self.get_convergence_path(), geo, err)

    def get_1d_geo(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), geo + '.vtp')

    def get_1d_params(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'parameters.npy')

    def get_seg_path(self, geo):
        return os.path.join(self.fpath_seg_path, geo)

    def get_cap_names(self, geo):
        caps = self.get_surface_names(geo, 'caps')

        bc_def = self.get_bcs(geo)
        names = {}
        if bc_def is not None:
            for c, n in bc_def['spname'].items():
                if isinstance(n, list):
                    names[c] = ' '.join(n).lower().capitalize()

        for c in caps:
            if c not in names:
                names[c] = c.replace('_', ' ').lower().capitalize()

        return names

    # todo: adapt to units?
    def get_path_file(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '-cm.paths')

    # todo: adapt to units?
    def get_seg_dir(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '_groups-cm')

    def get_inflow_osmsc(self, geo):
        fpath = os.path.join(self.fpath_extras, geo, 'extras', 'bctdat-in-cm.flow.txt')
        if not os.path.exists(fpath):
            return None, None

        # read from file
        flow = np.loadtxt(fpath)
        time, inflow = flow[:, 0], flow[:, 1]

        # get simulation parameters
        bc_def = self.get_bcs(geo)
        if bc_def is None:
            return None, None

        # fix flow in last time step
        inflow[-1] = inflow[0]

        return time, inflow

    def get_inflow(self, geo):
        # read inflow conditions
        if not os.path.exists(self.get_bc_flow_path(geo)):
            return None, None
        flow = np.load(self.get_bc_flow_path(geo), allow_pickle=True).item()

        # read 3d boundary conditions
        bc_def = self.get_bcs(geo)
        if bc_def is None:
            return None, None

        # extract inflow data
        time = flow['time']
        inflow = flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]

        return time, inflow

    def get_inflow_smooth_path(self, geo):
        return os.path.join(self.db_path, 'inflow', geo + '.txt')

    def get_inflow_smooth(self, geo):
        f = self.get_inflow_smooth_path(geo)
        if os.path.exists(f):
            m = np.loadtxt(f)
            return m[:, 0], m[:, 1]
        else:
            return None, None

    def get_surfaces_upload(self, geo):
        surfaces = glob.glob(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', '*.vtp'))
        surfaces.append(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp'))
        return surfaces

    def add_cap_ordered(self, caps, keys_ordered, keys_left, c):
        for k in sorted(caps):
            if c in k.lower() and k in keys_left:
                keys_ordered.append(k)
                keys_left.remove(k)

    def get_surfaces(self, geo, surf='all'):
        fdir = self.get_surface_dir(geo)
        surfaces_all = glob.glob(os.path.join(fdir, '*.vtp'))
        if surf == 'all':
            surfaces = surfaces_all
        elif surf == 'outlets' or surf == 'caps':
            exclude = ['all_exterior', 'wall', 'stent']
            if surf == 'outlets':
                exclude += ['inflow']
            surfaces = [x for x in surfaces_all if not any(e in x for e in exclude)]
        elif surf in self.get_surface_names(geo):
            surfaces = os.path.join(fdir, surf + '.vtp')
        else:
            print('Unknown surface option ' + surf)
            surfaces = []
        return surfaces

    def get_surface_names(self, geo, surf='all'):
        surfaces = self.get_surfaces(geo, surf)
        surfaces = [surfaces] if isinstance(surfaces, str) else surfaces
        surfaces = [os.path.splitext(os.path.basename(s))[0] for s in surfaces]
        surfaces.sort()

        if surf == 'caps':
            # nicely ordered cap names for output
            surfaces = sorted(surfaces)
            caps = surfaces.copy()
            keys_left = surfaces.copy()
            keys_ordered = []
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'inflow')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'aorta')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'p_')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'd_')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'left')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'right')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'l_')
            self.add_cap_ordered(caps, keys_ordered, keys_left, 'r_')
            keys_ordered += keys_left
            surfaces = keys_ordered

        return surfaces

    def get_surface_ids(self, geo, surf='all'):
        surfaces = self.get_surface_names(geo, surf)
        bc_def = self.get_bcs(geo)
        ids = []
        for s in surfaces:
            ids += [int(float(bc_def['spid'][s]))]
        ids.sort()
        return np.array(ids)

    def get_volume(self, geo):
        return os.path.join(self.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    def get_volume_mesh(self, geo):
        return os.path.join(self.get_sv_meshes(geo), geo + '.vtu')

    def get_res_3d_vol_rerun(self, geo):
        return os.path.join(self.fpath_study, '3d_flow', geo + '.vtu')

    def get_res_3d_surf_rerun(self, geo):
        return os.path.join(self.fpath_study, 'simulation', geo + '.vtp')

    def get_outlet_names(self, geo):
        bc_def = self.get_bcs(geo)
        if bc_def is None:
            return None
        names = [k for k, v in sorted(bc_def['preid'].items(), key=lambda kv: kv[1])]

        names_out = []
        for n in names:
            if 'wall' not in n and 'inflow' not in n and 'stent' not in n:
                names_out += [n]
        return names_out

    def count_inlets(self, geo):
        n_inlet = 0
        for s in self.get_surface_names(geo):
            if 'inflow' in s:
                n_inlet += 1
        return n_inlet

    def read_results(self, fpath):
        if os.path.exists(fpath):
            res = np.load(fpath, allow_pickle=True).item()
        else:
            print('no results in ' + fpath)
            return None

        if 'pressure' not in res or len(res['pressure']) == 0:
            print('results empty in ' + fpath)
            return None

        return res

    def get_3d_numstep(self, geo):
        # get model parameters
        bc_def = self.get_bcs(geo)

        return int(float(bc_def['params']['sim_steps_per_cycle']))

    def get_3d_timestep(self, geo):
        # read inflow conditions
        time, inflow = self.get_inflow(geo)

        # number of time steps
        numstep = self.get_3d_numstep(geo)

        # time step
        return time[-1] / numstep

    def get_3d_increment(self, geo):
        time, _ = self.get_inflow(geo)
        if time is None:
            return None

        # number of time steps
        numstep = self.get_3d_numstep(geo)

        # output increment
        nt_out = numstep / len(time)

        # check if increment can be represented by a uniform series of integers
        assert nt_out % 1 == 0.0, 'output not equally spaced'

        return int(nt_out)

    def get_time_constants(self, geo):
        params = self.get_bcs(geo)
        time, _ = self.get_inflow(geo)

        # skip geometries without BCs
        if params is None:
            return

        # collect all time constants
        tau_bc = {}
        for cp, bc in params['bc'].items():
            if 'Rd' in bc:
                tau = bc['Rd'] * bc['C']
            elif 'Pim' in bc:
                p = {}
                cor = coronary_sv_to_oned(bc)
                p['R1'], p['R2'], p['R3'], p['C1'], p['C2'] = (cor['Ra1'], cor['Ra2'], cor['Rv1'], cor['Ca'], cor['Cc'])
                tau1 = p['C1'] / (1 / (p['R2'] + p['R1']) + 1 / p['R3'])
                tau2 = p['C2'] / (1 / (p['R2'] + p['R3']) + 1 / p['R1'])
                tau = tau1 + tau2
            else:
                tau = 0

            #  tau in cardiac cycles
            tau_bc[cp] = tau / time[-1]

        return tau_bc


class SimVascular:
    """
    simvascular object to handle external calls
    """

    def __init__(self):
        self.svpre = '/home/pfaller/work/repos/svSolver/build/svSolver-build/bin/svpre'
        self.svsolver = '/home/pfaller/work/repos/svSolver/build/svSolver-build/bin/svsolver'
        self.svpost = '/home/pfaller/work/repos/svSolver/build/svSolver-build/bin/svpost'
        self.zerodsolver = '/home/pfaller/work/repos/svZeroDSolver_cpp/build/svzerodsolver'
        self.onedsolver = '/home/pfaller/work/repos/svOneDSolver_fork/build_skyline/bin/OneDSolver'
        self.sv = '/home/pfaller/work/repos/SimVascular_fork/build/SimVascular-build/sv'
        self.sv_legacy_io = '/home/pfaller/work/repos/SimVascularLegacyIO/build/SimVascular-build/sv'
        # self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/sv'
        self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/bin/simvascular'
        self.perigee = '/home/pfaller/work/repos/PERIGEE/tools/sv_file_converter/build'

    def run_pre(self, pre_folder, pre_file):
        subprocess.run([self.svpre, pre_file], cwd=pre_folder)

    def run_solver(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.svsolver, run_file], cwd=run_folder)

    def run_post(self, run_folder, args):
        subprocess.run([self.svpost] + args, cwd=run_folder, stdout=open(os.devnull, "w"))
        # run_command(run_folder, [self.svpost, args])

    def run_solver_0d(self, run_file, out_file='tmp.csv'):
        msg = run_command('.', [self.zerodsolver, run_file, out_file])
        return msg != 0

    def run_solver_1d(self, run_folder, run_file='solver.inp'):
        run_command(run_folder, [self.onedsolver, run_file])  # 'mpirun', '-np', '4',
        return ' ', True

    def run_python(self, command):
        return subprocess.run([self.sv, ' --python -- '] + command)

    def run_python_legacyio(self, command):
        p = subprocess.Popen([self.sv_legacy_io, ' --python -- '] + command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        m = ''
        for s in [self.sv_legacy_io, ' --python -- '] + command:
            m += s + ' '
        # print(m)
        return p.communicate()

    def run_python_debug(self, command):
        command = [self.sv, ' --python -- '] + command
        out_str = ''
        for c in command:
            out_str += c + ' '
        print(out_str)
        # return subprocess.run(['gdb', self.sv_debug])

    def run_perigee_sv_converter(self):
        run_command(run_folder, [self.onedsolver, run_file])


def get_tcl_paths(fpath, geo, offset):
    assert len(geo) == 9 and geo[4] == '_' and is_int(geo[:4]) and is_int(geo[5:]), geo + ' not in OSMSC format'
    ids = geo.split('_')
    geo_bc = ids[0] + '_' + str(int(ids[1]) + offset).zfill(4)
    return os.path.join(fpath, geo_bc + '.tcl'), os.path.join(fpath, geo_bc + '-bc.tcl')


class SVProject:
    def __init__(self):
        self.dir = {'images': 'Images', 'paths': 'Paths', 'segmentations': 'Segmentations', 'models': 'Models',
                    'meshes': 'Meshes', 'simulations': 'Simulations', 'flow': 'flow-files'}
        self.t = '    '


class Post:
    def __init__(self):
        # self.fields = ['pressure', 'flow', 'area']
        self.fields = ['pressure', 'flow']
        self.units = {'pressure': 'mmHg', 'flow': 'l/min', 'area': 'mm$^2$'}
        self.styles = {'3d': '-', '3d_rerun': '-', '3d_rerun_bc': '-', '1d': '-.', '0d': '--'}
        self.color = {'3d': 'k', '3d_rerun': 'tab:blue', '3d_rerun_bc': 'C1', '1d': 'tab:orange', '0d': 'r'}
        # self.color = {'3d': 'k', '3d_rerun': 'C0', '3d_rerun_bc': 'C1', '1d': 'C1', '0d': 'C2'}

        self.cgs2mmhg = 7.50062e-4
        self.mlps2lpmin = 60 / 1000
        self.convert = {'pressure': self.cgs2mmhg, 'flow': self.mlps2lpmin, 'area': 100}

        # sets the plot order
        # self.models = ['3d', '1d', '0d']
        # self.models = ['3d_rerun', '0d']
        # self.models = ['3d', '0d']
        # self.models = ['1d', '0d']
        self.models = ['3d_rerun', '3d', '1d', '0d']
        # self.models = ['3d', '3d_rerun']
        # self.models = ['3d', '3d_rerun_bc']
        # self.model_names = {'3d': '3d_legacy', '3d_rerun': 'svSolver', '1d': 'svOneDSolver', '0d': 'svZeroDSolver'}
        self.model_names = {'3d': '3d legacy', '3d_rerun': '3D', '1d': '1D', '0d': '0D'}

        self.colors = {'Cerebrovascular': 'k',
                       'Coronary': 'r',
                       'Aortofemoral': 'm',
                       'Pulmonary': 'c',
                       'Congenital Heart Disease': 'y',
                       'Aorta': 'b',
                       'Animal and Misc': '0.75'}


def run_command(run_folder, command):
    process = subprocess.Popen(command, cwd=run_folder, stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def exists(fpath):
    if os.path.exists(fpath):
        return fpath
    else:
        return None
