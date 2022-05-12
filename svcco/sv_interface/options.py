import os
import sys
import platform
import glob

class file_options:

    def __init__(self,time=[0,1],flow=[1,1],gui=True,distal_resistance=0):
        """
        Set the parameters for vascular construction
        and simulation for SimVascular python interface.
        """
        self.set_geometry_options()
        self.set_loft_options()
        self.set_solid_options()
        self.set_mesh_options()
        self.set_pipeline_steps(gui=gui)
        self.set_directories()
        if len(time) > 2:
            self.simulation_parameters(fourier_mode=10,period=time[-1],bct_point_number=2*len(time),distal_resistance=distal_resistance)
        else:
            self.simulation_parameters(distal_resistance=distal_resistance)
        self.flow_parameters(time=time,flow=flow)
        self.rom_parameters()

    def set_geometry_options(self,number_samples=50,use_distance_alignment_method=True):
        geometry_options = {'number_samples':                number_samples,
                            'use_distance_alignment_method': use_distance_alignment_method}
        self.geometry_options = geometry_options
        return
    def set_loft_options(self,u_knot_span_type=None,v_parametric_span_type=None,
                         u_parametric_span_type=None,v_knot_span_type=None,
                         v_degree=None,u_degree=None,boundary_face_angle=45):
        loft_param = [u_knot_span_type,v_parametric_span_type,
                      u_parametric_span_type,v_knot_span_type,
                      v_degree,u_degree]
        loft_param = [param is None for param in loft_param]
        if all(loft_param):
            loft_options = {'default':             True,
                            'changes':               '',
                            'angle':boundary_face_angle}
        else:
            loft_options = {'default': False,
                            'changes':    '',
                            'angle':boundary_face_angle}
            # Check Knot Span Values
            if u_knot_span_type is not None:
                if u_knot_span_type not in ['derivative','equal','average']:
                    print('Invalid knot span type\n\nMust be one of:\n  "derivative"\n  "equal"\n  "average"\n')
                    print('Will default to "derivative" type.')
                else:
                    loft_options['changes'] += 'options.u_knot_span_type = {}\n'.format(u_knot_span_type)
            if v_knot_span_type is not None:
                if v_knot_span_type not in ['derivative','equal','average']:
                    print('Invalid knot span type\n\nMust be one of:\n  "derivative"\n  "equal"\n  "average"\n')
                    print('Will default to "average" type.')
                else:
                    loft_options['changes'] += 'options.v_knot_span_type = {}\n'.format(v_knot_span_type)
            # Check Parametric Span Values
            if v_parametric_span_type is not None:
                if v_parametric_span_type not in ['centripetal','equal','chord']:
                    print('Invalid knot span type\n\nMust be one of:\n  "centripetal"\n  "equal"\n  "chord"\n')
                    print('Will default to "chord" type.')
                else:
                    loft_options['changes'] += 'options.v_parametric_span_type = {}\n'.format(v_parametric_span_type)
            if u_parametric_span_type is not None:
                if u_parametric_span_type not in ['centripetal','equal','chord']:
                    print('Invalid knot span type\n\nMust be one of:\n  "centripetal"\n  "equal"\n  "chord"\n')
                    print('Will default to "centripetal" type.')
                else:
                    loft_options['changes'] += 'options.u_parametric_span_type = {}\n'.format(u_parametric_span_type)
            if v_degree is not None:
                if not isinstance(v_degree,int):
                    print('v_degree must be an integer')
                    print('Will default to v_degree = 2')
                else:
                    loft_options['changes'] += 'options.v_degree = {}\n'.format(v_degree)
            if u_degree is not None:
                if not isinstance(u_degree,int):
                    print('v_degree must be an integer')
                    print('Will default to u_degree = 2')
                else:
                    loft_options['changes'] += 'options.u_degree = {}\n'.format(u_degree)
        self.loft_options = loft_options
        return
    def set_solid_options(self,minimum_face_cells=200,hmin=0.02,hmax=0.02,
                          face_edge_size=0.02,boundary_face_angle=45):
        #Check input types
        if not isinstance(minimum_face_cells,int):
            print('minimum_face_cells must be a positive int > 0')
            print('Using default value')
            minimum_face_cells=200
        if not isinstance(hmin,float):
            print('hmin must be a float < 1')
            print('Using default value')
            hmin=0.02
        if not isinstance(hmax,float):
            print('hmax must be a float hmin =< hmax < 1')
            print('Using default value')
            hmax=0.02
        if not isinstance(face_edge_size,float):
            print('face_edge_size must be a float < 1')
            print('Using default value')
            face_edge_size=0.02
        if boundary_face_angle > 90 or boundary_face_angle < 0:
            print('boundary_face_angle must be greater than 0 and less than 90')
            print('Using default value')
            boundary_face_angle = 45
        solid_options = {'minimum_face_cells':  minimum_face_cells,
                         'hmin':                             hmin,
                         'hmax':                             hmax,
                         'face_edge_size':          face_edge_size,
                         'angle':              boundary_face_angle}
        self.solid_options = solid_options
        return
    def set_mesh_options(self,global_edge_size=0.01,surface_mesh_flag=True,
                         volume_mesh_flag=True,no_merge=False,optimization_level=5,
                         minimum_dihedral_angle=18.0):
        if not isinstance(global_edge_size,float):
            print('global_edge_size must be a float less than 1')
            print('Using default: global_edge_size = 0.01')
            global_edge_size=0.01
        if not isinstance(surface_mesh_flag,bool):
            print('surface_mesh_flag must be boolean logic')
            print('Using default: surface_mesh_flag = True')
            surface_mesh_flag = True
        if not isinstance(volume_mesh_flag,bool):
            print('volume_mesh_flag must be boolean logic')
            print('Using default: volume_mesh_flag = True')
            volume_mesh_flag = True
        if not isinstance(no_merge,bool):
            print('no_merge must be boolean logic')
            print('Using default: no_merge = False')
            no_merge = False
        if not isinstance(optimization_level,int):
            print('optimization_level must be an integer between 0 and 7, inclusive')
            print('Using default: optimization_level = 5')
            optimization_level = 5
        elif optimization_level > 7 or optimization_level < 0:
            print('optimization_level must be an integer between 0 and 7, inclusive')
            print('Using default: optimization_level = 5')
            optimization_level = 5
        if not isinstance(minimum_dihedral_angle,float):
            print('minimum_dihedral_angle must be a float')
            print('Using default: minimum_dihedral_angle = 18')
            minimum_dihedral_angle = 18
        mesh_options = {'global_edge_size':            global_edge_size,
                        'surface_mesh_flag':          surface_mesh_flag,
                        'volume_mesh_flag':            volume_mesh_flag,
                        'no_merge':                            no_merge,
                        'optimization_level':        optimization_level,
                        'minimum_dihedral_angle':minimum_dihedral_angle}
        self.mesh_options = mesh_options
        return
    def set_pipeline_steps(self,files='all',gui=False):
        file_types = ['path','contour','solid',
                      'mesh','simulation','rom']
        # Check parameter values
        if not isinstance(files,str):
            print('{files} parameter must be a str')
            return
        if not isinstance(gui,bool):
            print('{gui} parameter must be True or False')
            return
        # Assign Model Constructor Values
        file_constructor = {'path':        True,
                            'contour':     True,
                            'solid':       True,
                            'mesh':        True,
                            'simulation':  True,
                            'rom':         True,
                            'gui':          gui}
        if files == 'all':
            pass
        else:
            if files in file_types:
               file_index = file_types.index(files)
               for i,f in enumerate(file_types):
                   if i <= file_index:
                       pass
                   else:
                       file_constructor[f] = False
        self.file_constructor = file_constructor
        return

    def set_directories(self,outdir=None):
        if isinstance(outdir,type(None)):
            outdir = os.getcwd()
        if isinstance(outdir,str):
            if not os.path.isdir(outdir):
                print('{} either does not exist or is not accessible'.format(outdir))
                return
            else:
                pass
        else:
            print('{outdir} should be a str argument if provided')
            return
        # Assigning Directory Constructor Values
        directory_constructor = {'main_folder':          None,
                                 'create_main_folder':  False,
                                 'data_folder':          None,
                                 'mesh_complete_folder': None,
                                 'mesh_surfaces_folder': None,
                                 'postprocessing_folder':None}
        if not os.path.isdir(outdir+'/CCO_DATA'):
            directory_constructor['create_main_folder'] = True
        else:
            directory_constructor['create_main_folder'] = False
        directory_constructor['main_folder'] = outdir+os.sep+'CCO_DATA'
        number = len(glob.glob(directory_constructor['main_folder']+os.sep+'CCO_*'))
        directory_constructor['data_folder'] = directory_constructor['main_folder']+os.sep+'simulation_data{}'.format(number)
        directory_constructor['mesh_complete_folder'] = directory_constructor['data_folder']+os.sep+'mesh-complete'
        directory_constructor['mesh_surfaces_folder'] = directory_constructor['mesh_complete_folder']+os.sep+'mesh-surfaces'
        directory_constructor['centerline_folder'] = directory_constructor['data_folder']+os.sep+'centerlines'
        directory_constructor['postprocessing_folder'] = directory_constructor['data_folder']+os.sep+'post'
        directory_constructor['rom_folder'] = directory_constructor['data_folder']+os.sep+'rom'
        directory_constructor['rom_0D_folder'] = directory_constructor['rom_folder']+os.sep+'zeroD'
        directory_constructor['rom_1D_folder'] = directory_constructor['rom_folder']+os.sep+'oneD'
        directory_constructor['main_folder']           = directory_constructor['main_folder'].replace('\\','\\\\')
        directory_constructor['data_folder']           = directory_constructor['data_folder'].replace('\\','\\\\')
        directory_constructor['mesh_complete_folder']  = directory_constructor['mesh_complete_folder'].replace('\\','\\\\')
        directory_constructor['mesh_surfaces_folder']  = directory_constructor['mesh_surfaces_folder'].replace('\\','\\\\')
        directory_constructor['centerline_folder']     = directory_constructor['centerline_folder'].replace('\\','\\\\')
        directory_constructor['postprocessing_folder'] = directory_constructor['postprocessing_folder'].replace('\\','\\\\')
        directory_constructor['rom_folder']            = directory_constructor['rom_folder'].replace('\\','\\\\')
        directory_constructor['rom_0D_folder']         = directory_constructor['rom_0D_folder'].replace('\\','\\\\')
        directory_constructor['rom_1D_folder']         = directory_constructor['rom_1D_folder'].replace('\\','\\\\')
        self.directory_constructor = directory_constructor
        self.number = number
        self.filename = directory_constructor['main_folder']+os.sep+'CCO_{}.py'.format(number)
        self.filename = self.filename.replace('\\','\\\\')
        self.name = 'CCO_{}'.format(number)
        return

    def simulation_parameters(self,svpre_name='cco',fluid_density=1.06,
                              fluid_viscosity=0.04, initial_pressure=0,
                              initial_velocity=[0.0001,0.0001,0.0001],
                              bct_analytical_shape='parabolic',
                              inflow_file='inflow',
                              period=1,bct_point_number=2,
                              fourier_mode=1,pressures=0,
                              svsolver='solver',number_timesteps=200,
                              timestep_size=0.02,number_restarts=50,
                              number_force_surfaces=1,
                              surface_id_force_calc=1,
                              force_calc_method='Velocity Based',
                              print_avg_solution=True,
                              print_error_indicators=False,
                              varying_time_from_file=True,
                              step_construction='0 1 0 1',
                              pressure_coupling='Implicit',
                              backflow_stabilization=0.2,
                              residual_control=True,
                              residual_criteria=0.01,
                              minimum_req_iter=2,
                              svLS_type='NS',num_krylov=100,
                              num_solves_per_left=1,
                              tolerance_momentum=0.05,
                              tolerance_continuity=0.4,
                              tolerance_svLS_NS=0.4,
                              max_iter_NS=2,max_iter_momentum=4,
                              max_iter_continuity=400,
                              time_integration_rule='Second Order',
                              time_integration_rho=0.5,
                              flow_advection_form='Convective',
                              quadrature_interior=2,
                              quadrature_boundary=3,procs=24,
                              svpost='cco',start=0,vtu=True,
                              vtp=False,vtkcombo=False,
                              all_arg=True,wss=False,
                              sim_units_mm=False,
                              sim_units_cm=True,
                              global_edge_size=0.01,
                              distal_resistance=0):
        # Assign PreSolver Constructor Values
        presolver_constructor = {'name':                           svpre_name,
                                 'fluid_density':               fluid_density,
                                 'fluid_viscosity':           fluid_viscosity,
                                 'initial_pressure':         initial_pressure,
                                 'initial_velocity':         initial_velocity,
                                 'bct_analytical_shape': bct_analytical_shape,
                                 'inflow_file':        inflow_file+'_3d.flow',
                                 'period':                             period,
                                 'bct_point_number':         bct_point_number,
                                 'fourier_mode':                 fourier_mode,
                                 'pressures':                       pressures}
        # Assign Solver Constructor Values
        solver_constructor = {'name':                                svsolver,
                              'fluid_density':                  fluid_density,
                              'fluid_viscosity':              fluid_viscosity,
                              'number_timesteps':            number_timesteps,
                              'timestep_size':                  timestep_size,
                              'number_restarts':              number_restarts,
                              'number_force_surfaces':  number_force_surfaces,
                              'surface_id_force_calc':  surface_id_force_calc,
                              'force_calc_method':          force_calc_method,
                              'print_avg_solution':        print_avg_solution,
                              'print_error_indicators':print_error_indicators,
                              'varying_time_from_file':varying_time_from_file,
                              'step_construction':          step_construction,
                              'pressure_coupling':          pressure_coupling,
                              'backflow_stabilization':backflow_stabilization,
                              'residual_control':            residual_control,
                              'residual_criteria':          residual_criteria,
                              'minimum_req_iter':            minimum_req_iter,
                              'svLS_type':                          svLS_type,
                              'num_krylov':                        num_krylov,
                              'num_solves_per_left':      num_solves_per_left,
                              'tolerance_momentum':        tolerance_momentum,
                              'tolerance_continuity':    tolerance_continuity,
                              'tolerance_svLS_NS':          tolerance_svLS_NS,
                              'max_iter_NS':                     max_iter_NS,
                              'max_iter_momentum':          max_iter_momentum,
                              'max_iter_continuity':      max_iter_continuity,
                              'time_integration_rule':  time_integration_rule,
                              'time_integration_rho':    time_integration_rho,
                              'flow_advection_form':      flow_advection_form,
                              'quadrature_interior':      quadrature_interior,
                              'quadrature_boundary':      quadrature_boundary,
                              'global_edge_size':            global_edge_size,
                              'distal_resistance':          distal_resistance}
        indir = self.directory_constructor['data_folder']+'/{}-procs_case'.format(procs)
        outdir = self.directory_constructor['postprocessing_folder']
        if sim_units_mm and sim_units_cm:
            print('Must pick one unit scale. Not both.')
            print('Exiting without assigning postprocessing parameters.')
            return
        postsolver_constructor = {'name':                              svpost,
                                  'start':                              start,
                                  'stop':                    number_timesteps,
                                  'incr':                     number_restarts,
                                  'vtu':                                  vtu,
                                  'vtp':                                  vtp,
                                  'vtkcombo':                        vtkcombo,
                                  'all':                              all_arg,
                                  'wss':                                  wss,
                                  'indir':                              indir,
                                  'sim_units_mm':                sim_units_mm,
                                  'sim_units_cm':                sim_units_cm,
                                  'outdir':                            outdir}
        self.presolver_constructor   = presolver_constructor
        self.solver_constructor      = solver_constructor
        self.postsolver_constructor  = postsolver_constructor
        return
    def flow_parameters(self,inflow_file='inflow',time=[0,1],flow=[1,1]):
        flow_constructor = {'inflow_file':                        inflow_file,
                            'time':                                      time,
                            'flow':                                      flow}
        self.flow_constructor = flow_constructor

    def rom_parameters(self,inflow='inflow_rom.flow',zeroD=True,oneD=True,time_step=0.001,
                       number_time_steps=1000):
        rom0D_constructor = {'create': zeroD,
                             'model_order': 0,
                             'time_step': time_step,
                             'number_time_steps': number_time_steps,
                             'inflow_file':  inflow}

        rom1D_constructor = {'create': oneD,
                             'model_order': 1,
                             'time_step': time_step,
                             'number_time_steps': number_time_steps,
                             'inflow_file':   inflow}

        self.rom_constructor = {'0D': rom0D_constructor,
                                '1D': rom1D_constructor}
        return
