#!/usr/bin/env python

import pdb
import os
import shutil
import glob
import matplotlib.cm as cm
from collections import OrderedDict, defaultdict

import numpy as np
import scipy.interpolate

from get_database import input_args, Database, SVProject, SimVascular, Post
from common import coronary_sv_to_oned, get_dict
from get_sv_meshes import get_meshes


class Project:
    def __init__(self, db, geo, mode):
        self.db = db
        self.geo = geo
        self.mode = mode
        self.opt = self.get_sv_opt()

    def create_sv_project(self):
        print('Estimated cycles: ' + str(self.opt['n_cycle']))
        self.check_files()
        self.make_folders()
        self.write_inflow('1d')
        self.write_inflow('3d')
        self.copy_files()
        self.write_svproj_file()
        self.write_model()
        self.write_simulation()
        self.write_mesh()
        self.write_pre()
        self.write_solver()
        self.write_bc(os.path.join(self.db.get_solve_dir_3d(self.geo)), False)
        self.write_path_segmentation()

    def get_sv_opt(self):
        # get boundary conditions
        bc_def = self.db.get_bcs(self.geo)
        if 'steady' in self.mode:
            bc_def = rc_to_r(bc_def)

        # number of cycles
        n_cycle = self.estimate_n_cycle()

        # number of time steps
        numstep = self.db.get_3d_numstep(self.geo)
    
        # time step
        dt = self.db.get_3d_timestep(self.geo)
    
        # time increment
        nt_out = self.db.get_3d_increment(self.geo)
    
        # read inflow conditions
        time, inflow = self.db.get_inflow_smooth(self.geo)

        if self.mode == '':
            pass
        elif self.mode == 'ini_1d_quad':
            n_cycle = 5
        elif self.mode == 'steady':
            n_cycle = 1
            numstep = 100
            nt_out = 1
            inflow = np.mean(inflow) * np.ones(len(time))
        elif self.mode == 'steady0':
            n_cycle = 1
            inflow = inflow[0] * np.ones(len(time))
        elif self.mode == 'irene':
            n_cycle = 1
    
            # sample inflow at fine time step
            dt_fine = dt# * 1e-3
            time_fine = np.arange(0, time[-1] + dt_fine, dt_fine)
            inflow_fine = scipy.interpolate.interp1d(time, inflow, fill_value="extrapolate")(time_fine)
    
            # find (first) time where inflow reaches mean flow
            f_mean = np.mean(inflow)
            t_mean = np.argwhere(np.diff(np.sign(inflow_fine - f_mean))).flatten()[0]
            assert np.abs(inflow_fine[t_mean]/f_mean - 1) < 5e-2, 'could not find mean-flow'
    
            # shift inflow to start at mean flow
            inflow_fine_shift = np.roll(inflow_fine, -t_mean)
    
            # interpolate back to original time step
            inflow = scipy.interpolate.interp1d(time_fine, inflow_fine_shift)(time)
            assert np.abs(inflow[0]/f_mean - 1) < 5e-2, 'inflow does not start at mean flow'
    
            p_str = 'time post nt=' + str(numstep-t_mean)
            p_str += ' mean inflow f=' + str(f_mean * Post().convert['flow']) + ' [l/]'
            p_str += ' at t=' + str(t_mean * dt_fine) + ' s'
            print(p_str)
        else:
            raise ValueError('Unknown mode ' + self.mode)

        # number of time steps
        n_time = n_cycle * numstep + nt_out
    
        # create inflow string
        inflow_str = array_to_sv(np.vstack((time, inflow)).T)
    
        # number of fourier modes
        n_t = time.shape[0]
        if n_t % 2 == 0:
            n_fourier = int(n_t / 2 + 1)
        else:
            n_fourier = int((n_t + 1) / 2)
        n_fourier = np.min([200, n_fourier])

        # number of points
        n_point = len(time)
    
        # get inflow type
        inflow_type = bc_def['bc']['inflow']['type']
        if inflow_type == 'womersley':
            inflow_type = 'plug'
        if inflow_type not in ['parabolic', 'plug', 'womersley']:
            raise ValueError('unknown inflow type ' + inflow_type)

        # set svsolver options
        opt = {'bc': bc_def,
               'density': '1.06',
               'viscosity': '0.04',
               'backflow': '0.2',
               'advection': 'Convective',
               'inflow': 'inflow.flow',
               'inflow_data': np.vstack((time, inflow)).T,
               'inflow_str': inflow_str,
               'inflow_type': inflow_type,
               'fourier_modes': str(n_fourier),
               'fourier_period': str(time[-1]),
               'fourier_points': str(n_point),
               'max_iter_continuity': '400',
               'max_iter_momentum': '10',
               'max_iter_ns_solver': '10',
               'min_iter': '3',
               'num_krylov': '300',
               'num_solve': '1',
               'num_time': str(int(n_time)),
               'num_restart': '1',#str(nt_out),
               'n_cycle': n_cycle,
               'bool_surf_stress': 'True',
               'coupling': 'Implicit',
               'print_avg_sol': 'True',
               'print_err': 'False',
               'quad_boundary': '3',
               'quad_interior': '2',
               'residual_control': 'True',
               'residual_criteria': '0.01',
               'residual_tolerance': '1.0e12',
               'step_construction': '5',
               'time_int_rho': '0.5',
               'time_int_rule': 'Second Order',
               'time_step': str(dt),
               'tol_continuity': '0.01',
               'tol_momentum': '0.01',
               'tol_ns_solver': '0.01',
               'svls_type': 'NS',
               'mesh_initial': os.path.join('mesh-complete', 'initial.vtu'),
               'mesh_vtu': os.path.join('mesh-complete', 'mesh-complete.mesh.vtu'),
               'mesh_vtp': os.path.join('mesh-complete', 'mesh-complete.exterior.vtp'),
               'mesh_inflow': os.path.join('mesh-complete', 'mesh-surfaces', 'inflow.vtp'),
               'mesh_walls': os.path.join('mesh-complete', 'walls_combined.vtp')}
        return opt
    
    def estimate_n_cycle(self, n_extra=1, n_default=5):
        # get numerical time constants
        db = Database('1spb_length')
        res_num = get_dict(db.get_convergence_path())

        params = db.get_bcs(self.geo)
        cat = params['params']['deliverable_category']

        # pure resistance
        bc_types = np.unique(list(params['bc_type'].values()))
        if len(bc_types) == 1 and bc_types[0] == 'resistance':
            return 2

        # time constant
        if self.geo not in res_num:
            tau_list = []
            for outlet, val in params['bc'].items():
                if outlet in params['bc_type'] and params['bc_type'][outlet] == 'rcr':
                    tau_list += [val['C'] * val['Rd']]
            tau = np.max(tau_list)
        else:
            tau = np.mean(res_num[self.geo]['tau']['pressure'])

        # tolerance for asymptotic convergence
        tol = 0.01

        # number of cardiac cycles required to reach tolerance (+ extra) from zero ICs
        n_cycle_zero = int(- np.log(tol) * tau + 0.5) + n_extra

        # estimated number of cycles from steady state ICs
        n_cycle_mean = int(4 + 2 * (tau + 0.5))

        if cat == 'Coronary':
            return n_cycle_zero
        else:
            return np.min([n_cycle_zero, n_cycle_mean])
    
    def write_svproj_file(self):
        t = str(self.db.svproj.t)
        proj_head = ['<?xml version="1.0" encoding="UTF-8"?>',
                     '<projectDescription version="1.0">']
        proj_end = ['</projectDescription>']
    
        with open(self.db.get_svproj_file(self.geo), 'w+') as f:
            # write header
            for s in proj_head:
                f.write(s + '\n')
    
            # write images/segmentations
            for k, s in self.db.svproj.dir.items():
                f.write(t + '<' + k + ' folder_name="' + s + '"')
                if k == 'images' and self.db.get_img(self.geo) is not None:
                    img = os.path.basename(self.db.get_img(self.geo))
                    f.write('>\n')
                    f.write(
                        t * 2 + '<image name="' + os.path.splitext(img)[0] + '" in_project="yes" path="' + img + '"/>\n')
                    f.write(t + '</' + k + '>\n')
                # elif k == 'segmentations':
                else:
                    f.write('/>\n')
    
            # write end
            for s in proj_end:
                f.write(s + '\n')
    
    def write_model(self):
        t = str(self.db.svproj.t)
        model_head = ['<?xml version="1.0" encoding="UTF-8" ?>',
                      '<format version="1.0" />',
                      '<model type="PolyData">',
                      t + '<timestep id="0">',
                      t * 2 + '<model_element type="PolyData" num_sampling="0">']
        model_end = [t * 3 + '<blend_radii />',
                     t * 3 + '<blend_param blend_iters="2" sub_blend_iters="3" cstr_smooth_iters="2" lap_smooth_iters="50" '
                             'subdivision_iters="1" decimation="0.01" />',
                     t * 2 + '</model_element>',
                     t + '</timestep>',
                     '</model>']
    
        # read boundary conditions
        bc_def = self.opt['bc']
        bc_def['preid']['wall'] = 0
    
        # get cap names
        caps = self.db.get_surface_names(self.geo, 'caps')
        caps += ['wall']
    
        # sort caps according to face id
        ids = np.array([repr(int(float(bc_def['preid'][c] + 1))) for c in caps])
        order = np.argsort(ids)
        caps = np.array(caps)[order]
        ids = ids[order]
    
        # display colors for caps
        colors = cm.jet(np.linspace(0, 1, len(caps)))
    
        # write model file
        with open(self.db.get_svproj_mdl_file(self.geo), 'w+') as f:
            # write header
            for s in model_head:
                f.write(s + '\n')
    
            #             <segmentations>
            #                 <seg name="aorta_final_new" />
            #                 <seg name="btrunk_final" />
            #                 <seg name="carotid_final" />
            #                 <seg name="subclavian_final_new" />
            #             </segmentations>
    
            # write faces
            f.write(t * 3 + '<faces>\n')
            for i, c in enumerate(caps):
                c_str = t * 4 + '<face id="' + ids[i] + '" name="' + c + '" type='
                if c == 'wall':
                    c_str += '"wall"'
                else:
                    c_str += '"cap"'
                for j in range(3):
                    c_str += ' color' + repr(j + 1) + '="' + repr(colors[i, j]) + '"'
                f.write(c_str + ' visible="true" opacity="1" />\n')
            f.write(t * 3 + '</faces>\n')
    
            # write end
            for s in model_end:
                f.write(s + '\n')
    
    def write_path_segmentation(self):
        # SimVascular instance
        sv = SimVascular()
    
        # get paths
        p = OrderedDict()
        p['f_path_in'] = self.db.get_path_file(self.geo)
        p['f_path_out'] = os.path.join(self.db.get_svproj_dir(self.geo), self.db.svproj.dir['paths'])
    
        seg_dir = self.db.get_seg_dir(self.geo)
        segments = glob.glob(os.path.join(seg_dir, '*'))
    
        err_seg = ''
        for s in segments:
            p['f_seg_in'] = s
            p['f_seg_out'] = os.path.join(self.db.get_svproj_dir(self.geo), self.db.svproj.dir['segmentations'])
    
            if '.tcl' in s:
                continue
    
            # assemble call string
            sv_string = [os.path.join(os.getcwd(), 'sv_get_path_segmentation.py')]
            for v in p.values():
                sv_string += [v]
    
            err = sv.run_python_legacyio(sv_string)[1]
            if err:
                err_seg += os.path.basename(s).split('.')[0] + '\n'
    
        # execute SimVascular-Python
        return err_seg
    
    def write_mesh(self):
        t = str(self.db.svproj.t)
        mesh_generic = ['<?xml version="1.0" encoding="UTF-8" ?>',
                        '<format version="1.0" />',
                        '<mitk_mesh type="TetGen" model_name="' + self.geo + '">',
                        t + '<timestep id="0">',
                        t * 2 + '<mesh type="TetGen">',
                        t * 3 + '<command_history>',
                        t * 4 + '<command content="option surface 1" />',
                        t * 4 + '<command content="option volume 1" />',
                        t * 4 + '<command content="option UseMMG 1" />',
                        t * 4 + '<command content="setWalls" />',
                        # t * 4 + '<command content="option Optimization 3" />',
                        # t * 4 + '<command content="option QualityRatio 1.4" />',
                        t * 4 + '<command content="option NoBisect" />',
                        # t * 4 + '<command content="AllowMultipleRegions 0" />',
                        t * 4 + '<command content="generateMesh" />',
                        t * 4 + '<command content="writeMesh" />',
                        t * 3 + '</command_history>',
                        t * 2 + '</mesh>',
                        t + '</timestep>',
                        '</mitk_mesh>']
    
        fname = os.path.join(self.db.get_svproj_dir(self.geo), self.db.svproj.dir['meshes'], self.geo + '.msh')
    
        # write generic mesh file
        with open(fname, 'w+') as f:
            for s in mesh_generic:
                f.write(s + '\n')
    
    def write_inflow(self, model, n_mode=10, n_sample_real=256):
        # read inflow conditions
        opt = self.opt
        time = opt['inflow_data'][:, 0]
        inflow = opt['inflow_data'][:, 1]

        if time is None:
            raise ValueError('no inflow')

        # save inflow file
        fpath = self.db.get_sv_flow_path(self.geo, model)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        # reverse flow for svOneDSolver
        fmt = '%.10e'
        if model == '1d':
            np.savetxt(fpath, np.vstack((time, - inflow)).T, fmt=fmt)
        else:
            np.savetxt(fpath, np.vstack((time, inflow)).T, fmt=fmt)

        return len(inflow), time[-1]
    
    def write_pre(self, solver='svsolver'):
        """
        Create input file for svpre
        """
        # get boundary conditions
        bc_def = self.opt['bc']
    
        # read inflow conditions
        time, _ = self.db.get_inflow_smooth(self.geo)
    
        # outlet names
        outlets = self.db.get_outlet_names(self.geo)
    
        # get solver options
        opt = self.opt
    
        with open(self.db.get_svpre_file(self.geo, solver), 'w+') as f:
            # enter debug mode
            # f.write('verbose true\n')
    
            # write volume mesh
            f.write('mesh_and_adjncy_vtu ' + opt['mesh_vtu'] + '\n')
    
            # write surface mesh
            fpath_surf = os.path.join('mesh-complete', 'mesh-surfaces')
    
            # write surfaces (sort according to surface ID for readability)
            f.write('set_surface_id_vtp ' + opt['mesh_vtp'] + ' 1\n')
            f.write('set_surface_id_vtp ' + opt['mesh_inflow'] + ' 2\n')
            for k in outlets:
                v = bc_def['preid'][k] + 1
                if int(v) > 1:
                    f_surf = os.path.join(fpath_surf, k + '.vtp')
    
                    # check if mesh file exists
                    f_surf_full = os.path.join(self.db.get_solve_dir_3d(self.geo), f_surf)
                    assert os.path.exists(f_surf_full), 'file ' + f_surf + ' does not exist'
    
                    f.write('set_surface_id_vtp ' + f_surf + ' ' + repr(int(v)) + '\n')
            f.write('\n')
    
            if solver == 'perigee':
                return
    
            # write inlet bc
            f.write('prescribed_velocities_vtp ' + opt['mesh_inflow'] + '\n\n')
    
            # generate inflow
            f.write('bct_analytical_shape ' + opt['inflow_type'] + '\n')
            f.write('bct_period ' + opt['fourier_period'] + '\n')
            f.write('bct_point_number ' + opt['fourier_points'] + '\n')
            f.write('bct_fourier_mode_number ' + opt['fourier_modes'] + '\n')
            # f.write('bct_create ' + opt['mesh_inflow'] + ' ' + self.db.get_sv_flow_path_rel(self.geo, '3d_constant') + '\n')
            f.write('bct_create ' + opt['mesh_inflow'] + ' ' + opt['inflow'] + '\n')
            f.write('bct_write_dat bct.dat\n')
            f.write('bct_write_vtp bct.vtp\n\n')
    
            # write default parameters
            f.write('fluid_density ' + opt['density'] + '\n')
            f.write('fluid_viscosity ' + opt['viscosity'] + '\n\n')
    
            # reference pressure
            for cap in outlets:
                bc = bc_def['bc'][cap]
                if cap == 'inflow' or cap == 'wall':
                    continue
                if 'Po' in bc and bc_def['bc_type'][cap] == 'resistance':
                    p0 = str(bc['Po'])
                else:
                    p0 = '0.0'
                f.write('pressure_vtp ' + os.path.join(fpath_surf, cap + '.vtp') + ' ' + p0 + '\n')
            f.write('\n')
    
            # set previous results as initial condition
            f.write('read_pressure_velocity_vtu ' + opt['mesh_initial'] + '\n\n')
            # f.write('initial_pressure 0\n')
            # f.write('initial_velocity 0.0001 0.0001 0.0001\n\n')
    
            # no slip boundary condition
            f.write('noslip_vtp ' + opt['mesh_walls'] + '\n\n')
    
            # request outputs
            f.write('write_geombc geombc.dat.1\n')
            f.write('write_restart restart.0.1\n')
            f.write('write_numstart 0\n\n')
    
        # write start file
        fname_start = os.path.join(self.db.get_solve_dir_3d(self.geo), 'numstart.dat')
        with open(fname_start, 'w+') as f:
            f.write('0')
    
    def write_solver(self):
        # get boundary conditions
        bc_def = self.opt['bc']
    
        # ordered outlets
        outlets = self.db.get_outlet_names(self.geo)
    
        # get solver options
        opt = self.opt
    
        with open(self.db.get_solver_file(self.geo), 'w+') as f:
            # write default parameters
            # todo: get from tcl
            f.write('Density: ' + opt['density'] + '\n')
            f.write('Viscosity: ' + opt['viscosity'] + '\n\n')
    
            # time step
            f.write('Number of Timesteps: ' + opt['num_time'] + '\n')
            f.write('Time Step Size: ' + opt['time_step'] + '\n\n')
    
            # output
            f.write('Number of Timesteps between Restarts: ' + opt['num_restart'] + '\n')
            f.write('Number of Force Surfaces: 1\n')
            f.write('Surface ID\'s for Force Calculation: 0\n')
            f.write('Force Calculation Method: Velocity Based\n')
            f.write('Print Average Solution: ' + opt['print_avg_sol'] + '\n')
            f.write('Print Error Indicators: ' + opt['print_err'] + '\n\n')
    
            f.write('Time Varying Boundary Conditions From File: True\n\n')
    
            f.write('Step Construction:')
            for i in range(int(opt['step_construction'])):
                f.write(' 0 1')
            f.write('\n\n')
    
            # collect faces for each boundary condition type
            bc_ids = defaultdict(list)
            for cap in outlets:
                bc_ids[bc_def['bc_type'][cap]] += [int(bc_def['preid'][cap]) + 1]
    
            # boundary conditions
            names = {'rcr': 'RCR', 'resistance': 'Resistance', 'coronary': 'COR'}
            for t, v in bc_ids.items():
                f.write('Number of ' + names[t] + ' Surfaces: ' + str(len(v)) + '\n')
                f.write('List of ' + names[t] + ' Surfaces: ' + str(v).replace(',', '')[1:-1] + '\n')
    
                if t == 'rcr' or t == 'coronary':
                    f.write(names[t] + ' Values From File: True\n\n')
                elif t == 'resistance':
                    f.write('Resistance Values: ')
                    for cap in self.db.get_outlet_names(self.geo):
                        if bc_def['bc_type'][cap] == 'resistance':
                            f.write(str(bc_def['bc'][cap]['R']) + ' ')
                    f.write('\n\n')
                else:
                    raise ValueError('Boundary condition ' + t + ' unknown')
    
            f.write('Pressure Coupling: ' + opt['coupling'] + '\n')
            f.write('Number of Coupled Surfaces: ' + str(len(bc_def['bc']) - 2) + '\n\n')
    
            f.write('Backflow Stabilization Coefficient: ' + opt['backflow'] + '\n')
    
            # nonlinear solver
            f.write('Residual Control: ' + opt['residual_control'] + '\n')
            f.write('Residual Criteria: ' + opt['residual_criteria'] + '\n')
            f.write('Residual Tolerance: ' + opt['residual_tolerance'] + '\n')
            f.write('Minimum Required Iterations: ' + opt['min_iter'] + '\n')
    
            # linear solver
            f.write('svLS Type: ' + opt['svls_type'] + '\n')
            f.write('Number of Krylov Vectors per GMRES Sweep: ' + opt['num_krylov'] + '\n')
            f.write('Number of Solves per Left-hand-side Formation: ' + opt['num_solve'] + '\n')
    
            f.write('Tolerance on Momentum Equations: ' + opt['tol_momentum'] + '\n')
            f.write('Tolerance on Continuity Equations: ' + opt['tol_continuity'] + '\n')
            f.write('Tolerance on svLS NS Solver: ' + opt['tol_ns_solver'] + '\n')
    
            f.write('Maximum Number of Iterations for svLS NS Solver: ' + opt['max_iter_ns_solver'] + '\n')
            f.write('Maximum Number of Iterations for svLS Momentum Loop: ' + opt['max_iter_momentum'] + '\n')
            f.write('Maximum Number of Iterations for svLS Continuity Loop: ' + opt['max_iter_continuity'] + '\n')
    
            # time integration
            f.write('Time Integration Rule: ' + opt['time_int_rule'] + '\n')
            f.write('Time Integration Rho Infinity: ' + opt['time_int_rho'] + '\n')
    
            f.write('Flow Advection Form: ' + opt['advection'] + '\n')
    
            f.write('Quadrature Rule on Interior: ' + opt['quad_interior'] + '\n')
            f.write('Quadrature Rule on Boundary: ' + opt['quad_boundary'] + '\n')
    
    def write_simulation(self):
        # get boundary conditions
        bc_def = self.opt['bc']

        # get outlet names
        outlets = self.db.get_outlet_names(self.geo)
    
        # get solver options
        opt = self.opt
    
        # tab
        t = str(self.db.svproj.t)
    
        sim_header = ['<?xml version="1.0" encoding="UTF-8" ?>',
                      '<format version="1.0" />',
                      '<mitk_job model_name="' + self.geo + '" mesh_name="' + self.geo + '" status="Simulation failed">',
                      t + '<job>']
    
        basic_props = [['Fluid Density', opt['density']],
                       ['Fluid Viscosity', opt['viscosity']],
                       ['IC File', opt['mesh_initial']],
                       ['Initial Pressure', '0'],
                       ['Initial Velocities', '0.0001 0.0001 0.0001']]
    
        inflow_props = [['Analytic Shape', bc_def['bc']['inflow']['type']],
                        ['BC Type', 'Prescribed Velocities'],
                        ['Flip Normal', 'False'],
                        ['Flow Rate', opt['inflow_str']],
                        ['Fourier Modes', opt['fourier_modes']],
                        ['Original File', 'inflow.flow'],
                        ['Period', opt['fourier_period']],
                        ['Point Number', opt['fourier_points']]]
    
        wall_props = [['Type', 'rigid']]
    
        solver_props = [['Backflow Stabilization Coefficient', opt['backflow']],
                        ['Flow Advection Form', opt['advection']],
                        ['Force Calculation Method', 'Velocity Based'],
                        ['Maximum Number of Iterations for svLS Continuity Loop', opt['max_iter_continuity']],
                        ['Maximum Number of Iterations for svLS Momentum Loop', opt['max_iter_momentum']],
                        ['Maximum Number of Iterations for svLS NS Solver', opt['max_iter_ns_solver']],
                        ['Minimum Required Iterations', opt['min_iter']],
                        ['Number of Krylov Vectors per GMRES Sweep', opt['num_krylov']],
                        ['Number of Solves per Left-hand-side Formation', opt['num_solve']],
                        ['Number of Timesteps', opt['num_time']],
                        ['Number of Timesteps between Restarts', opt['num_restart']],
                        ['Output Surface Stress', opt['bool_surf_stress']],
                        ['Pressure Coupling', opt['coupling']],
                        ['Print Average Solution', opt['print_avg_sol']],
                        ['Print Error Indicators', opt['print_err']],
                        ['Quadrature Rule on Boundary', opt['quad_boundary']],
                        ['Quadrature Rule on Interior', opt['quad_interior']],
                        ['Residual Control', opt['residual_control']],
                        ['Residual Criteria', opt['residual_criteria']],
                        ['Residual Tolerance', opt['residual_tolerance']],
                        ['Step Construction', opt['step_construction']],
                        ['Time Integration Rho Infinity', opt['time_int_rho']],
                        ['Time Integration Rule', opt['time_int_rule']],
                        ['Time Step Size', opt['time_step']],
                        ['Tolerance on Continuity Equations', opt['tol_continuity']],
                        ['Tolerance on Momentum Equations', opt['tol_momentum']],
                        ['Tolerance on svLS NS Solver', opt['tol_ns_solver']],
                        ['svLS Type', opt['svls_type']]]
    
        run_props = [['Number of Processes', '8']]
    
        with open(self.db.get_svproj_sjb_file(self.geo), 'w+') as f:
            for h in sim_header:
                f.write(h + '\n')
    
            f.write(t * 2 + '<basic_props>\n')
            print_props(f, basic_props, t * 3)
            f.write(t * 2 + '</basic_props>\n')
    
            # bcs
            f.write(t * 2 + '<cap_props>\n')
    
            # outflow
            for k in outlets:
                f.write(t * 3 + '<cap name="' + k + '">\n')
    
                tp = bc_def['bc_type'][k]
                bc = bc_def['bc'][k]
    
                if tp == 'rcr':
                    rcr_val = ' '.join([str(bc[v]) for v in ['Rp', 'C', 'Rd']])
                    f.write(t * 4 + '<prop key="BC Type" value="RCR" />\n')
                    f.write(t * 4 + '<prop key="C Values" value="" />\n')
                    if 'Po' in bc:
                        f.write(t * 4 + '<prop key="Pressure" value="' + str(bc['Po']) + '" />\n')
                    else:
                        f.write(t * 4 + '<prop key="Pressure" value="0.0" />\n')
                    f.write(t * 4 + '<prop key="R Values" value="" />\n')
                    f.write(t * 4 + '<prop key="Values" value="' + rcr_val + '" />\n')
                elif tp == 'resistance':
                    f.write(t * 4 + '<prop key="BC Type" value="Resistance" />\n')
                    if 'Po' in bc:
                        f.write(t * 4 + '<prop key="Pressure" value="' + str(bc['Po']) + '" />\n')
                    else:
                        f.write(t * 4 + '<prop key="Pressure" value="0.0" />\n')
                    f.write(t * 4 + '<prop key="Values" value="' + str(bc['R']) + '" />\n')
                elif tp == 'coronary':
                    # convert parameters to SimVascular format
                    bc_sv = coronary_sv_to_oned(bc)
    
                    c_val = ' '.join([str(bc_sv[v]) for v in ['Ca', 'Cc']])
                    r_val = ' '.join([str(bc_sv[v]) for v in ['Ra1', 'Ra2', 'Rv1']])
                    p_val = str(bc_sv['P_v'])
                    a_val = ' '.join([str(bc_sv[v]) for v in ['Ra1', 'Ca', 'Ra2', 'Cc', 'Rv1']])
    
                    p_v = bc_def['coronary'][bc['Pim']]
    
                    # save ventricular pressure to file
                    f_out = os.path.join(self.db.get_solve_dir_3d(self.geo), bc['Pim'])
                    np.savetxt(f_out, p_v)
    
                    f.write(t * 4 + '<prop key="BC Type" value="Coronary" />\n')
                    f.write(t * 4 + '<prop key="C Values" value="' + c_val + '" />\n')
                    f.write(t * 4 + '<prop key="Original File" value="' + os.path.join(self.geo, bc['Pim']) + '" />\n')
                    f.write(t * 4 + '<prop key="Pressure" value="' + p_val + '" />\n')
                    f.write(t * 4 + '<prop key="Pressure Period" value="' + str(p_v[-1, 0]) + '" />\n')
                    f.write(t * 4 + '<prop key="Pressure Scaling" value="1.0" />\n')
                    f.write(t * 4 + '<prop key="R Values" value="' + r_val + '" />\n')
                    f.write(t * 4 + '<prop key="Timed Pressure" value="' + array_to_sv(p_v) + '" />\n')
                    f.write(t * 4 + '<prop key="Values" value="' + a_val + '" />\n')
                else:
                    raise ValueError('Boundary condition ' + tp + ' unknown')
    
                f.write(t * 3 + '</cap>\n')
    
            # inflow
            f.write(t * 3 + '<cap name="inflow">\n')
            print_props(f, inflow_props, t * 4)
            f.write(t * 3 + '</cap>\n')
    
            f.write(t * 2 + '</cap_props>\n')
    
            # wall
            f.write(t * 2 + '<wall_props>\n')
            print_props(f, wall_props, t * 3)
            f.write(t * 2 + '</wall_props>\n')
    
            # various
            f.write(t * 2 + '<var_props />\n')
    
            # solver
            f.write(t * 2 + '<solver_props>\n')
            print_props(f, solver_props, t * 3)
            f.write(t * 2 + '</solver_props>\n')
    
            # run
            f.write(t * 2 + '<run_props>\n')
            print_props(f, run_props, t * 3)
            f.write(t * 2 + '</run_props>\n')
    
            # close
            f.write(t + '</job>\n')
            f.write('</mitk_job>')
    
    def write_bc(self, fdir, write_face=True, model='3d'):
        # get boundary conditions
        bc_def = self.opt['bc']
    
        # check if bc-file exists
        if not bc_def:
            return None, 'boundary conditions do not exist'
    
        # get outlet names
        outlets = self.db.get_outlet_names(self.geo)
    
        # names expected by svsolver for different boundary conditions
        bc_file_names = {'rcr': 'rcrt.dat', 'resistance': 'resistance.dat', 'coronary': 'cort.dat'}
    
        # keyword to indicate a new boundary condition
        keywords = {'rcr': '2', 'coronary': '1001'}
    
        # create bc-files for every bc type
        u_bc_types = list(set(bc_def['bc_type'].values()))
        files = {}
        fnames = []
        for t in u_bc_types:
            if t in bc_file_names:
                fname = os.path.join(fdir, bc_file_names[t])
                files[t] = open(fname, 'w+')
                fnames += [fname]
    
                # write keyword for new faces in first line
                if t == 'rcr' or t == 'coronary':
                    files[t].write(keywords[t] + '\n')
            else:
                return None, 'boundary condition not implemented (' + t + ')'
    
        # write boundary conditions
        for s in outlets:
            bc = bc_def['bc'][s]
            t = bc_def['bc_type'][s]
            f = files[t]
            write_vals = lambda names: f.write('\n'.join([str(bc[v]) for v in names]))
            if t == 'rcr':
                f.write(keywords[t] + '\n')
                if write_face:
                    f.write(s + '\n')
                write_vals(['Rp', 'C', 'Rd'])
                if 'Po' in bc and bc['Po'] != 0.0:
                    p_ref = bc['Po']
                else:
                    p_ref = 0.0
                f.write('\n0.0 ' + str(p_ref) + '\n')
                f.write('1.0 ' + str(p_ref) + '\n')
            elif t == 'resistance':
                f.write(s + ' ')
                f.write(str(bc['R']) + ' ')
                f.write(str(bc['Po']) + '\n')
            elif t == 'coronary':
                f.write(keywords[t] + '\n')
                if model == '1d':
                    f.write(s + '\n')
                write_vals(['q0', 'q1', 'q2', 'p0', 'p1', 'p2', 'b0', 'b1', 'b2', 'dQinidT', 'dPinidT'])
                f.write('\n')
    
                # write time and pressure pairs
                for m in bc_def['coronary'][bc['Pim']]:
                    f.write(str(m[0]) + ' ' + str(m[1]) + '\n')
    
        # close all opened files
        for t in u_bc_types:
            files[t].close()
    
        return fnames, False
    
    def copy_files(self):
        # get solver options
        opt = self.opt
    
        # define paths
        sim_dir = self.db.get_solve_dir_3d(self.geo)
        fpath_surf = os.path.join(sim_dir, 'mesh-complete', 'mesh-surfaces')
    
        # create simulation folder
        os.makedirs(fpath_surf, exist_ok=True)
    
        # copy inflow
        np.savetxt(os.path.join(sim_dir, 'inflow.flow'), opt['inflow_data'])
        # shutil.copy(self.db.get_sv_flow_path(self.geo, '3d'), os.path.join(sim_dir, 'inflow.flow'))
    
        # copy cap meshes
        for f in glob.glob(os.path.join(self.db.get_sv_meshes(self.geo), 'caps', '*.vtp')):
            shutil.copy(f, fpath_surf)
    
        # copy surface and volume mesh
        shutil.copy(self.db.get_sv_surface(self.geo), os.path.join(sim_dir, opt['mesh_vtp']))
        shutil.copy(self.db.get_volume_mesh(self.geo), os.path.join(sim_dir, opt['mesh_vtu']))
    
        # copy initial condition mesh
        shutil.copy(self.db.get_initial_conditions(self.geo), os.path.join(sim_dir, opt['mesh_initial']))
    
        # copy wall mesh
        shutil.copy(os.path.join(self.db.get_sv_meshes(self.geo), 'walls_combined.vtp'), os.path.join(sim_dir, opt['mesh_walls']))
    
    def copy_file(self, src, trg_dir):
        trg = os.path.join(self.db.get_svproj_dir(self.geo), self.db.svproj.dir[trg_dir], os.path.basename(src))
        shutil.copy2(src, trg)
    
    def make_folders(self):
        # make all project sub-folders
        for s in self.db.svproj.dir.values():
            os.makedirs(os.path.join(self.db.get_svproj_dir(self.geo), s), exist_ok=True)
    
        # copy image
        if self.db.get_img(self.geo) is not None:
            self.copy_file(self.db.get_img(self.geo), 'images')
    
        # copy volume mesh
        self.copy_file(self.db.get_volume_mesh(self.geo), 'meshes')
    
        # copy surface mesh
        self.copy_file(self.db.get_sv_surface(self.geo), 'meshes')
        self.copy_file(self.db.get_sv_surface(self.geo), 'models')
    
        return True
    
    def check_files(self):
        # check if files exist
        if self.db.get_volume_mesh(self.geo) is None:
            raise RuntimeError('no volume mesh')
        if self.db.get_sv_surface(self.geo) is None:
            raise RuntimeError('no SV surface mesh')
        # if self.db.get_img(self.geo) is None:
        #     raise RuntimeError('no medical image')


def print_props(f, props, t):
    for h in props:
        f.write(t + '<prop key="' + h[0] + '" value="' + h[1] + '" />\n')


def array_to_sv(array):
    sv_str = ''
    for j, (t, i) in enumerate(array):
        sv_str += str(t) + ' ' + str(i)
        if j < array.shape[0] - 1:
            sv_str += '&#x0A;'
    return sv_str


def rc_to_r(bc_def):
    """
    Convert all boundary conditions to resistance (for steady flow simulation)
    """
    for s, t in bc_def['bc_type'].items():
        bc = bc_def['bc'][s]
        bc_def['bc_type'][s] = 'resistance'
        if t == 'resistance':
            pass
        elif t == 'rcr':
            bc_def['bc'][s] = {'R': bc['Rp'] + bc['Rd'], 'Po': bc['Po']}
        elif t == 'coronary':
            bc_sv = coronary_sv_to_oned(bc)
            bc_def['bc'][s] = {'R': bc_sv['Ra1'] + bc_sv['Ra2'] + bc_sv['Rv1'], 'Po': bc_sv['P_v']}
    return bc_def


def main(db, geometries, params):
    for geo in geometries:

        if params.mode is not None:
            mode = params.mode
        else:
            mode = ''

        print('Running geometry ' + geo)
        try:
            # get meshes
            get_meshes(db, geo)

            # create sv project
            pj = Project(db, geo, mode)
            pj.create_sv_project()
        except Exception as e:
            # print(e)
            continue


if __name__ == '__main__':
    descr = 'Generate an svproject folder'
    d, g, p = input_args(descr)
    main(d, g, p)
