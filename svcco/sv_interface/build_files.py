import os
from copy import deepcopy
from .setup import setup as setup_file
from .paths import paths as path_file
from .contour import contour as contour_file
from .geometry import geometry as geometry_file
from .loft import loft as loft_file
from .solid import solid as solid_file
from .mesh import mesh as mesh_file
from .presolver import presolver as presolver_file
from .solver import solver as solver_file
from .centerline import centerline as centerline_file
from .rom import rom as rom_file
from .flow import flow as flow_file

def build(paths,radii,normals,options,setup_file=setup_file,
          path_file=path_file,contour_file=contour_file.split("\n"),
          geometry_file=geometry_file,loft_file=loft_file,
          solid_file=solid_file,mesh_file=mesh_file,
          presolver_file=presolver_file,solver_file=solver_file,
          centerline_file=centerline_file,rom_file=rom_file,flow_file=flow_file):
    cco_filename = options.filename
    if options.directory_constructor['create_main_folder']:
        os.mkdir(options.directory_constructor['main_folder'])
    cco_file = open(cco_filename,'w+')
    #setup_file = open(os.path.dirname(__file__)+'/setup.txt','r').read()
    if options.file_constructor['path']:
        #path_file = open(os.path.dirname(__file__)+'/paths.txt','r').read()
        path_file = path_file.format(options.file_constructor['gui'])
    else:
        path_file = ''
    if options.file_constructor['contour']:
        #contour_file = open(os.path.dirname(__file__)+'/contour.txt','r').readlines()
        tmp_contour_file = deepcopy(contour_file)
        tmp_contour_file[8] = tmp_contour_file[8].format(options.file_constructor['gui'])
        #print(contour_file[8])
        for i in range(len(tmp_contour_file)):
            tmp_contour_file[i] += "\n"
    else:
        contour_file = ''
    if options.file_constructor['solid']:
        #geometry_file = open(os.path.dirname(__file__)+'/geometry.txt','r').read()
        #geometry_file = geometry_file.format(options.geometry_options['number_samples'],
        #                                     options.geometry_options['use_distance_alignment_method'])
        #loft_file = open(os.path.dirname(__file__)+'/loft.txt','r').read()
        #loft_file = loft_file.format(options.loft_options['changes'],
        #                             options.loft_options['angle'])
        #solid_file = open(os.path.dirname(__file__)+'/solid.txt','r').read()
        #solid_file = solid_file.format(options.solid_options['minimum_face_cells'],
        #                               options.solid_options['hmin'],
        #                               options.solid_options['hmax'],
        #                               options.solid_options['face_edge_size'],
        #                               options.solid_options['face_edge_size'],
        #                               options.solid_options['angle'],
        #                               options.file_constructor['gui'],
        #                               options.name)
        loft_file = loft_file
        solid_file = solid_file.format(options.solid_options['num_caps'],options.solid_options['angle'],
                                       options.file_constructor['gui'],
                                       options.name)
    else:
        geometry_file = ''
        loft_file = ''
        solid_file = ''
    if options.file_constructor['mesh']:
        #mesh_file = open(os.path.dirname(__file__)+'/mesh.txt','r').read()
        mesh_file = mesh_file.format(options.mesh_options['global_edge_size'],
                                     options.mesh_options['surface_mesh_flag'],
                                     options.mesh_options['volume_mesh_flag'],
                                     options.mesh_options['no_merge'],
                                     options.mesh_options['optimization_level'],
                                     options.mesh_options['minimum_dihedral_angle'],
                                     options.file_constructor['gui'],
                                     options.name,
                                     options.name,
                                     options.directory_constructor['create_main_folder'],
                                     options.directory_constructor['main_folder'],
                                     options.directory_constructor['data_folder'],
                                     options.directory_constructor['mesh_complete_folder'],
                                     options.directory_constructor['mesh_surfaces_folder'],
                                     options.directory_constructor['centerline_folder'],
                                     options.directory_constructor['mesh_complete_folder'],
                                     options.directory_constructor['mesh_complete_folder'],
                                     options.directory_constructor['mesh_complete_folder'],
                                     options.directory_constructor['mesh_complete_folder'],
                                     options.directory_constructor['mesh_surfaces_folder'])
        #centerline_file = open(os.path.dirname(__file__)+'/centerline.txt','r').read()
        centerline_file = centerline_file.format(options.directory_constructor['centerline_folder'])
    else:
        mesh_files = ''
        centerline_file = ''
    #presolver_file = open(os.path.dirname(__file__)+'/presolver.txt','r').read()
    presolver_file = presolver_file.format(options.directory_constructor['data_folder'],
                                           options.presolver_constructor['name'],
                                           options.presolver_constructor['fluid_density'],
                                           options.presolver_constructor['fluid_viscosity'],
                                           options.presolver_constructor['initial_pressure'],
                                           options.presolver_constructor['initial_velocity'][0],
                                           options.presolver_constructor['initial_velocity'][1],
                                           options.presolver_constructor['initial_velocity'][2],
                                           options.presolver_constructor['bct_analytical_shape'],
                                           options.presolver_constructor['period'],
                                           options.presolver_constructor['bct_point_number'],
                                           options.presolver_constructor['fourier_mode'],
                                           options.presolver_constructor['inflow_file'],
                                           options.presolver_constructor['pressures'])
    #solver_file = open(os.path.dirname(__file__)+'/solver.txt','r').read()
    solver_file = solver_file.format(options.directory_constructor['data_folder'],
                                     options.solver_constructor['name'],
                                     options.solver_constructor['fluid_density'],
                                     options.solver_constructor['fluid_viscosity'],
                                     options.solver_constructor['number_timesteps'],
                                     options.solver_constructor['timestep_size'],
                                     options.solver_constructor['number_restarts'],
                                     options.solver_constructor['number_force_surfaces'],
                                     options.solver_constructor['surface_id_force_calc'],
                                     options.solver_constructor['force_calc_method'],
                                     options.solver_constructor['print_avg_solution'],
                                     options.solver_constructor['print_error_indicators'],
                                     options.solver_constructor['varying_time_from_file'],
                                     options.solver_constructor['step_construction'],
                                     options.solver_constructor['distal_resistance'],
                                     options.solver_constructor['pressure_coupling'],
                                     options.solver_constructor['backflow_stabilization'],
                                     options.solver_constructor['residual_control'],
                                     options.solver_constructor['residual_criteria'],
                                     options.solver_constructor['minimum_req_iter'],
                                     options.solver_constructor['svLS_type'],
                                     options.solver_constructor['num_krylov'],
                                     options.solver_constructor['num_solves_per_left'],
                                     options.solver_constructor['tolerance_momentum'],
                                     options.solver_constructor['tolerance_continuity'],
                                     options.solver_constructor['tolerance_svLS_NS'],
                                     options.solver_constructor['max_iter_NS'],
                                     options.solver_constructor['max_iter_momentum'],
                                     options.solver_constructor['max_iter_continuity'],
                                     options.solver_constructor['time_integration_rule'],
                                     options.solver_constructor['time_integration_rho'],
                                     options.solver_constructor['flow_advection_form'],
                                     options.solver_constructor['quadrature_interior'],
                                     options.solver_constructor['quadrature_boundary'],
                                     options.directory_constructor['data_folder'])
    flow_file = flow_file.format(options.directory_constructor['data_folder'],
                                 options.flow_constructor['inflow_file'],
                                 options.directory_constructor['data_folder'],
                                 options.flow_constructor['inflow_file'],
                                 options.flow_constructor['time'],
                                 options.flow_constructor['flow'])
    rom_files = 'import sv\n'
    if options.rom_constructor['0D']['create'] or options.rom_constructor['1D']['create']:
        rom_files += "os.mkdir('{}')\n".format(options.directory_constructor['rom_folder'])
    if options.rom_constructor['0D']['create']:
        rom_files += "os.mkdir('{}')\n".format(options.directory_constructor['rom_0D_folder'])
        rom0D = rom_file.format(options.name,
                                '0D',
                                options.directory_constructor['centerline_folder'],
                                options.directory_constructor['data_folder'],
                                options.rom_constructor['0D']['inflow_file'],
                                options.rom_constructor['0D']['time_step'],
                                options.rom_constructor['0D']['number_time_steps'],
                                options.directory_constructor['rom_0D_folder'],
                                options.rom_constructor['0D']['model_order'],
                                options.rom_constructor['0D']['model_order'],
                                options.directory_constructor['rom_0D_folder'],
                                options.directory_constructor['rom_0D_folder'],
                                options.rom_constructor['0D']['model_order'],
                                options.directory_constructor['rom_0D_folder'])
        rom_files += rom0D
    if options.rom_constructor['1D']['create']:
        rom_files += "os.mkdir('{}')\n".format(options.directory_constructor['rom_1D_folder'])
        rom1D = rom_file.format(options.name,
                                '1D',
                                options.directory_constructor['centerline_folder'],
                                options.directory_constructor['data_folder'],
                                options.rom_constructor['1D']['inflow_file'],
                                options.rom_constructor['1D']['time_step'],
                                options.rom_constructor['1D']['number_time_steps'],
                                options.directory_constructor['rom_1D_folder'],
                                options.rom_constructor['1D']['model_order'],
                                options.rom_constructor['1D']['model_order'],
                                options.directory_constructor['rom_1D_folder'],
                                options.directory_constructor['rom_1D_folder'],
                                options.rom_constructor['1D']['model_order'],
                                options.directory_constructor['rom_1D_folder'])
        rom_files += rom1D
    #postsolver_file = open('postsolver.txt','r').read()
    ############################################################
    # BUILDING SV PATHLINES
    ############################################################
    cco_file_text = ''
    cco_file_text += setup_file
    for idx, points in enumerate(paths):
        cco_file_text += path_file.format(idx,idx,points,idx,idx)
    ############################################################
    ############################################################

    ############################################################
    # BUILDING SV CONTOURS
    ############################################################
    cco_file_text += tmp_contour_file[0]
    cco_file_text += tmp_contour_file[1]
    for idx, path in enumerate(paths):
        cco_file_text += tmp_contour_file[2].format(idx)
        cco_file_text += tmp_contour_file[3].format(idx)
        contours = zip(paths[idx],radii[idx],normals[idx])
        for jdx,contour in enumerate(contours):
            cco_file_text += tmp_contour_file[4].format(idx,contour[1],contour[0],contour[2])
            cco_file_text += tmp_contour_file[5].format(idx,idx)
        cco_file_text += tmp_contour_file[6].format(idx)
        cco_file_text += tmp_contour_file[7].format(idx)
        cco_file_text += tmp_contour_file[8]
        cco_file_text += tmp_contour_file[9].format(idx,idx,idx)
    ############################################################
    ############################################################

    ############################################################
    # COMPUTING GEOMETRY
    ############################################################
    #cco_file_text += geometry_file
    ############################################################
    ############################################################

    ############################################################
    # LOFTING CONTOURS
    ############################################################
    cco_file_text += loft_file
    ############################################################
    ############################################################

    ############################################################
    # BUILDING SV SOLID
    ############################################################
    cco_file_text += solid_file
    ############################################################
    ############################################################

    ############################################################
    # BUILDING SV MESH
    ############################################################
    cco_file_text += mesh_file
    ############################################################
    ############################################################

    ############################################################
    # BUILDING SVPRE FILE
    ############################################################
    cco_file_text += presolver_file
    ############################################################
    ############################################################

    ############################################################
    # BUILDING SOLVER FILE
    ############################################################
    cco_file_text += solver_file
    ############################################################

    ############################################################
    # BUILD INFLOW FILES
    ############################################################
    cco_file_text += flow_file
    ############################################################

    ############################################################
    # EXTRACT CENTERLINE
    ############################################################
    cco_file_text += centerline_file
    ############################################################


    ############################################################
    # BUILDING ROM INPUT FILES
    ############################################################
    cco_file_text += rom_files
    ############################################################
    cco_file.writelines([cco_file_text])
    cco_file.close()
