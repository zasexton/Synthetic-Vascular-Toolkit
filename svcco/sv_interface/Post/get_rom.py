#!/usr/bin/env python

import glob
import io
import os
import pdb
import re
import shutil
import sys
import time
import json

import numpy as np
from pandas import read_csv

from get_database import Database, SimVascular, Post, input_args, run_command
from get_sv_project import Project
from simulation_io import read_results_1d, get_caps_db
from common import coronary_sv_to_oned


import sv_rom_simulation as oned
import svzerodsolver as run0d


def get_params(db, geo):
    # store parameters for export
    params = {}

    if db.get_bcs(geo) is None:
        raise RuntimeError('No boundary conditions')

    n_inlet = db.count_inlets(geo)
    if n_inlet == 0:
        raise RuntimeError('3d geometry has no inlet')
    elif n_inlet > 1:
        raise RuntimeError('3d geometry has multiple inlets (' + repr(n_inlet) + ')')

    # simvascular project
    project = Project(db, geo, '')

    params['fpath_1d'] = db.get_solve_dir_1d(geo)
    fpath_geo = os.path.join(params['fpath_1d'], 'geometry')
    os.makedirs(fpath_geo, exist_ok=True)

    # reference pressure (= initial pressure?)
    if os.path.exists(db.get_bc_flow_path(geo)):
        res_3d = db.read_results(db.get_bc_flow_path(geo))
        params['pref'] = res_3d['pressure'][-1, 0]
    else:
        raise RuntimeError('3d results do not exist')

    if db.has_loop(geo):
        raise RuntimeError('3d geometry contains a loop')

    # copy surface model to folder if it exists
    if os.path.exists(db.get_surfaces(geo, 'all_exterior')):
        shutil.copy2(db.get_surfaces(geo, 'all_exterior'), fpath_geo)
    else:
        raise RuntimeError('3d geometry does not exist')

    # assume pre-computed centerlines from SimVascular C++
    params['centerlines_input_file'] = db.get_centerline_path(geo)
    if not os.path.exists(params['centerlines_input_file']):
        raise RuntimeError('centerline does not exist')

    # get boundary conditions
    params['bc_def'] = db.get_bcs(geo)
    if params['bc_def'] is None:
        raise RuntimeError('no boundary conditions')

    # write outlet boundary conditions to file if they exist
    params['bc_types'], err = project.write_bc(params['fpath_1d'], model='1d')
    if err:
        raise RuntimeError(err)

    # get inflow
    time, _ = db.get_inflow_smooth(geo)
    if time is None:
        raise RuntimeError('inflow does not exist')
    params['inflow_input_file'] = db.get_sv_flow_path(geo, '1d')

    # get outlets
    caps = get_caps_db(db, geo)
    del caps['inflow']
    params['outlets'] = list(caps.keys())

    # material parameters
    params['olufsen_k1'] = 1e7
    params['olufsen_k2'] = -20
    params['olufsen_k3'] = 1e7
    params['linear_ehr'] = 3e5
    if not deformable:
        params['mat_model'] = 'OLUFSEN'
    else:
        params['mat_model'] = 'LINEAR'

    # number of cycles to run
    params['n_cycle'] = project.estimate_n_cycle(n_default=50)

    # sub-segment size
    params['seg_min_num'] = 1
    params['seg_size'] = 999

    # FEM size
    params['min_num_elems'] = 10
    params['element_size'] = 0.01

    # mesh adaptive?
    params['seg_size_adaptive'] = True

    # set simulation time as end of 3d simulation
    params['save_data_freq'] = 1
    params['dt'] = 1.0e-3

    # number of time steps per cycle
    params['num_dts_cycle'] = int(time[-1] / params['dt'])

    # run all cycles
    params['num_dts'] = int(time[-1] * params['n_cycle'] / params['dt'] + 1.0)

    # model order (0 or 1)
    params['model_order'] = model_order
    params['solver_output_file'] = geo + '_' + str(model_order) + 'd.in'

    print('tau [cycles] = ' + str(params['n_cycle']))
    return params


def generate_1d_api(db, geo):
    import sv

    # get parameters
    params_db = get_params(db, geo)

    # create 1d simulation
    rom_simulation = sv.simulation.ROM()

    # Create 1D simulation parameters
    params = sv.simulation.ROMParameters()

    # Mesh parameters
    mesh_params = params.MeshParameters()
#    mesh_params.use_adaptive_meshing = params_db['seg_size_adaptive']
#    mesh_params.num_branch_segments = params_db['seg_min_num']
    mesh_params.element_size = params_db['element_size']

    # Model parameters
    model_params = params.ModelParameters()
    model_params.name = geo
    model_params.inlet_face_names = ['inflow']
    model_params.outlet_face_names = params_db['outlets']
    model_params.centerlines_file_name = params_db['centerlines_input_file']

    # Fluid properties
    fluid_props = params.FluidProperties()

    # Set wall properties
    if params_db['mat_model'] == 'OLUFSEN':
        material = params.WallProperties.OlufsenMaterial()
        material.k1 = params_db['olufsen_k1']
        material.k2 = params_db['olufsen_k2']
        material.k3 = params_db['olufsen_k3']
        material.exponent = 2.0
        material.pressure = params_db['pref']
    elif params_db['mat_model'] == 'LINEAR':
        material = params.WallProperties.LinearMaterial()
        material.eh_r = params_db['linear_ehr']
        material.pressure = params_db['pref']
    else:
        raise ValueError('Unknonwn material ' + params_db['mat_model'])

    # Set boundary conditions
    bcs = params.BoundaryConditions()

    # add inlet
    bcs.add_velocities(face_name='inlet', file_name=params_db['inflow_input_file'])

    # add outlets
    for cp in params_db['outlets']:
        t = params_db['bc_def']['bc_type'][cp]
        bc = params_db['bc_def']['bc'][cp]
        if 'Po' in bc and bc['Po'] != 0.0:
            raise ValueError('RCR reference pressure not implemented')
        if t == 'rcr':
            bcs.add_rcr(face_name=cp, Rp=bc['Rp'], C=bc['C'], Rd=bc['Rd'])
        elif t == 'resistance':
            bcs.add_resistance(face_name=cp, resistance=bc['R'])
        elif t == 'coronary':
            raise ValueError('Coronary BC not implemented')

    # Set solution parameters
    solution_params = params.Solution()
    solution_params.time_step = params_db['dt']
    solution_params.num_time_steps = params_db['num_dts']
    solution_params.save_frequency = params_db['save_data_freq']

    # Write a 1D solver input file
    rom_simulation.write_input_file(model=model_params,
                                    model_order=model_order,
                                    mesh=mesh_params,
                                    fluid=fluid_props,
                                    material=material,
                                    boundary_conditions=bcs,
                                    solution=solution_params,
                                    directory=params_db['fpath_1d'])
    return None


def generate_1d(db, geo):
    # get parameters
    params = get_params(db, geo)

    # simvascular project
    project = Project(db, geo, '')

    # set simulation paths
    fpath_geo = os.path.join(params['fpath_1d'], 'geometry')
    fpath_surf = os.path.join(fpath_geo, 'surfaces')
    os.makedirs(fpath_surf, exist_ok=True)

    # copy outlet surfaces
    for f in db.get_surfaces(geo, 'caps'):
        shutil.copy2(f, fpath_surf)

    # copy outlet names
    fpath_outlets = os.path.join(params['fpath_1d'], 'outlets')
    shutil.copy(db.get_centerline_outlet_path(geo), fpath_outlets)

    # write inflow
    project.write_inflow('1d')

    # try:
    if True:
        oned.run(model_order=params['model_order'],
                 boundary_surfaces_directory=fpath_surf,
                 centerlines_input_file=params['centerlines_input_file'],
                 centerlines_output_file=None,
                 compute_centerlines=False,
                 compute_mesh=True,
                 density=params['bc_def']['params']['sim_density'],
                 element_size=params['element_size'],
                 inlet_face_input_file='inflow.vtp',
                 inflow_input_file=db.get_sv_flow_path(geo, '1d'),
                 linear_material_ehr=params['linear_ehr'],
                 linear_material_pressure=params['pref'],
                 material_model=params['mat_model'],
                 mesh_output_file=geo + '.vtp',
                 min_num_elements=params['min_num_elems'],
                 model_name=geo,
                 num_time_steps=params['num_dts'],
                 olufsen_material_k1=params['olufsen_k1'],
                 olufsen_material_k2=params['olufsen_k2'],
                 olufsen_material_k3=params['olufsen_k3'],
                 olufsen_material_exp=2.0,
                 olufsen_material_pressure=params['pref'],
                 outflow_bc_input_file=params['fpath_1d'],
                 outflow_bc_type=','.join(params['bc_types']),
                 outlet_face_names_input_file=fpath_outlets,
                 output_directory=params['fpath_1d'],
                 seg_min_num=params['seg_min_num'],
                 seg_size=params['seg_size'],
                 seg_size_adaptive=params['seg_size_adaptive'],
                 solver_output_file=params['solver_output_file'],
                 save_data_frequency=params['save_data_freq'],
                 surface_model=os.path.join(fpath_geo, 'all_exterior.vtp'),
                 time_step=params['dt'],
                 uniform_bc=True,
                 units='cm',
                 viscosity=params['bc_def']['params']['sim_viscosity'],
                 wall_properties_input_file=None,
                 wall_properties_output_file=None,
                 write_mesh_file=True,
                 write_solver_file=True)
    # except Exception as e:
    #     return repr(e)

    return None

model_order = 0
zerod_cpp = True
zerod_cpp_fast = True
deformable = False

# from profilehooks import profile
# @profile
def main(db, geometries):
    # simvascular object
    sv = SimVascular()

    for geo in geometries:
        print('Running geometry ' + geo)

        # if (model_order == 0 and os.path.exists(db.get_0d_flow_path(geo))) or (model_order == 1 and os.path.exists(db.get_1d_flow_path(geo))):
        #     print('Results exist. Skipping...')
        #     continue

        msg = None
        # msg = generate_1d_api(db, geo)
        # sys.exit(0)
        # get parameters
        try:
            params = get_params(db, geo)
            # msg = generate_1d(db, geo)
        except Exception as e:
            msg = 'error: ' + str(e)

        # continue
        if not msg:
            if model_order == 1:
                # run oneDSolver
                start = time.time()
                sv.run_solver_1d(db.get_solve_dir_1d(geo), params['solver_output_file'])
                end = time.time()

                # extract results
                res_dir = db.get_solve_dir_1d(geo)
                params_file = db.get_1d_params(geo)
                results_1d = read_results_1d(res_dir, params_file)

                # save results
                if results_1d['flow']:
                    # save results in dict
                    np.save(db.get_1d_flow_path(geo), results_1d)

                    # remove 1d output files
                    for f in glob.glob(os.path.join(res_dir, geo + 'branch*seg*_*.dat')):
                        os.remove(f)

                    msg = 'success'
                else:
                    msg = 'unconverged'
            elif model_order == 0:
                # parameters for 0d simulation
                zerod_in = os.path.join(db.get_solve_dir_1d(geo), params['solver_output_file'])

                # run 0d simulation
                # try:
                if True:
                    if zerod_cpp:
                        # quick output?
                        with open(zerod_in, 'r') as file:
                            inp = json.load(file)
                            if zerod_cpp_fast:
                                label = True
                            else:
                                label = False
                            inp['simulation_parameters']['output_variable_based'] = label
                            inp['simulation_parameters']['output_last_cycle_only'] = label
                        with open(zerod_in, 'w') as file:
                            json.dump(inp, file, indent=4, sort_keys=True)

                        # run cpp
                        zerod_out = db.get_0d_flow_path(geo).replace('npy', 'csv')

                        start = time.time()
                        err = sv.run_solver_0d(zerod_in, zerod_out)
                        end = time.time()

                        if err:
                            msg = 'cpp error'
                        else:
                            msg = 'success'

                            if not zerod_cpp_fast:
                                # convert to numpy
                                df = read_csv(zerod_out)
                                out = {"flow": {}, "pressure": {}, "distance": {}}

                                # read input file
                                with open(zerod_in, 'r') as file:
                                    inp = json.load(file)

                                # loop branches and segments
                                names = list(sorted(set(df["name"])))
                                for name in names:
                                    # extract ids
                                    br, seg = [int(s) for s in re.findall(r'\d+', name)]

                                    # add 0d results
                                    for field in ['flow', 'pressure']:
                                        if seg == 0:
                                            out[field][br] = [list(df[df.name == name][field + '_in'])]
                                        out[field][br] += [list(df[df.name == name][field + '_out'])]
                                    out["time"] = list(df[df.name == name]["time"])

                                    # add path distance
                                    for vessel in inp['vessels']:
                                        if vessel['vessel_name'] == name:
                                            if seg == 0:
                                                out["distance"][br] = [0]
                                            l_new = out["distance"][br][-1] + vessel['vessel_length']
                                            out["distance"][br] += [l_new]

                                # convert to numpy
                                for field in ['flow', 'pressure', 'distance']:
                                    for br in out[field].keys():
                                        out[field][br] = np.array(out[field][br])
                                out['time'] = np.array(out['time'])

                                # save to file
                                np.save(db.get_0d_flow_path(geo), out)

                    else:
                        ini_steady = True
                        if 'coronary' in db.get_bcs(geo)['bc_type'].values():
                            ini_steady = False

                        start = time.time()
                        run0d.solver.set_up_and_run_0d_simulation(zerod_in,
                                                                  use_steady_soltns_as_ics=ini_steady)
                        end = time.time()

                        # move output file
                        src = os.path.join(db.get_solve_dir_1d(geo), geo + '_0d_branch_results.npy')
                        dst = db.get_0d_flow_path(geo)
                        shutil.move(src, dst)

                        msg = 'success'
                # except Exception as e:
                #     msg = '0d failed: ' + str(e)
        if msg != 'success':
            print('  skipping (1d model creation failed)\n  ' + msg)
        print('\nTime elapsed: ' + '{:.2f}'.format(end - start) + '\n')

        # store errors in file
        if model_order == 1:
            db.add_log_file_1d(geo, msg)
            db.add_log_file_1d(geo + '_time', end - start)
        elif model_order == 0:
            db.add_log_file_0d(geo, msg)
            db.add_log_file_0d(geo + '_time', end - start)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
