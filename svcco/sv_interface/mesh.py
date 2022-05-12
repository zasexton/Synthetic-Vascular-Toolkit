mesh="""if not terminating:
    faces = model.get_face_ids()
    mesher = meshing.create_mesher(meshing.Kernel.TETGEN)
    tet_options = meshing.TetGenOptions({},{},{})
    tet_options.no_merge = {}
    tet_options.optimization = {}
    tet_options.minimum_dihedral_angle = {}
    mesher.set_model(model)
    mesher.set_walls(walls)
    mesher.generate_mesh(tet_options)
    msh = mesher.get_mesh()
    if {}:
        dmg.add_mesh('{}',msh,'{}')
    if {}:
        os.mkdir('{}')
    os.mkdir('{}')
    os.mkdir('{}')
    os.mkdir('{}')
    os.mkdir('{}')
    model.write('{}'+os.sep+'model_tmp','vtp')
    mesher.write_mesh('{}'+os.sep+'mesh-complete.mesh.vtu')
    mesh_out = modeling.PolyData()
    mesh_out.set_surface(mesher.get_surface())
    mesh_out.write('{}'+os.sep+'mesh-complete','exterior.vtp')
    mesh_out.set_surface(mesher.get_face_polydata(walls[0]))
    mesh_out.write('{}'+os.sep+'walls_combined','vtp')
    for face in mesher.get_model_face_ids():
        if face == walls[0]:
            continue
        mesh_out.set_surface(mesher.get_face_polydata(face))
        mesh_out.write('{}'+os.sep+'cap_{{}}'.format(face),'vtp')
"""
