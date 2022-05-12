presolver="""if not terminating:
    svpre_file = open('{}'+'/{}.svpre','w+')
    svpre_construction = 'mesh_and_adjncy_vtu mesh-complete/mesh-complete.mesh.vtu\\n'
    svpre_construction += 'set_surface_id_vtp mesh-complete/mesh-complete.exterior.vtp 1\\n'
    resistance = dict([])
    mass = vtk.vtkMassProperties()
    max_surface_area = 0
    total_res = 0
    for i,face in enumerate(mesher.get_model_face_ids()):
        if face == walls[0]:
            continue
        f_poly = mesher.get_face_polydata(face)
        mass.SetInputData(f_poly)
        resistance[face] = mass.GetSurfaceArea()
        total_res += 1/resistance[face]
        if resistance[face] > max_surface_area:
            skip = i + 1
            inlet = face
            max_surface_area = resistance[face]
        svpre_construction += 'set_surface_id_vtp mesh-complete/mesh-surfaces/cap_{{}}.vtp {{}}\\n'.format(face,i+1)
    total_res -= 1/max_surface_area
    svpre_construction += 'fluid_density {}\\n'
    svpre_construction += 'fluid_viscosity {}\\n'
    svpre_construction += 'initial_pressure {}\\n'
    svpre_construction += 'initial_velocity {} {} {}\\n'
    svpre_construction += 'prescribed_velocities_vtp mesh-complete/mesh-surfaces/cap_{{}}.vtp\\n'.format(inlet)
    svpre_construction += 'bct_analytical_shape {}\\n'
    svpre_construction += 'bct_period {}\\n'
    svpre_construction += 'bct_point_number {}\\n'
    svpre_construction += 'bct_fourier_mode_number {}\\n'
    svpre_construction += 'bct_create mesh-complete/mesh-surfaces/cap_{{}}.vtp {}\\n'.format(inlet)
    svpre_construction += 'bct_write_dat bct.dat\\n'
    svpre_construction += 'bct_write_vtp bct.vtp\\n'
    for i, outlet in enumerate(mesher.get_model_face_ids()):
        if outlet == walls[0]:
            continue
        elif outlet == inlet:
            continue
        svpre_construction += 'pressure_vtp mesh-complete/mesh-surfaces/cap_{{}}.vtp {}\\n'.format(outlet)
    svpre_construction += 'noslip_vtp mesh-complete/walls_combined.vtp\\n'
    svpre_construction += 'write_geombc geombc.dat.1\\n'
    svpre_construction += 'write_restart restart.0.1\\n'
    svpre_file.writelines([svpre_construction])
    svpre_file.close()
"""
