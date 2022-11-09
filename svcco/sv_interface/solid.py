solid="""###############################
# INITIALIZE MODELING KERNEL
###############################
def robust_union(model_1,model_2):
    modeler = modeling.Modeler(modeling.Kernel.POLYDATA)
    model_1_be = bad_edges(model_1)
    model_2_be = bad_edges(model_2)
    print("Model 1 Bad Edges: {{}}\\n Model 2 Bad Edges: {{}}".format(model_1_be,model_2_be))
    if model_1_be == 0 and model_2_be == 0:
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        if bad_edges(unioned_model) > 0:
            print('Unioned Model Bad Edges: {{}}'.format(bad_edges(unioned_model)))
            print('Filling')
            unioned_model = fill(unioned_model)
            print('Unioned Model Bad Edges: {{}}'.format(bad_edges(unioned_model)))
            print('Cleaning')
            unioned_model = clean(unioned_model)
            print('Unioned Model Bad Edges: {{}}'.format(bad_edges(unioned_model)))
            unioned_model = tri(unioned_model)
            print('Unioned Model Bad Edges: {{}}'.format(bad_edges(unioned_model)))
        print('union successful')
        return unioned_model
    else:
        print('1 or both models have bad edges.')
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        return unioned_model

def union_all(solids,n_cells=100):
    for i in range(len(solids)):
        solids[i] = norm(solids[i])
        solids[i] = remesh(solids[i])
        solids[i] = remesh_caps(solids[i])
    joined = robust_union(solids[0],solids[1])
    for i in range(2,len(solids)):
        print("UNION NUMBER: "+str(i)+"/"+str(len(solids)-1))
        joined = robust_union(joined,solids[i])
        if joined is None:
            print("unioning failed")
            return None,True
    print("unioning passed")
    return joined,False

unioned_model,terminating = union_all(capped_vessels)
model = modeling.PolyData()
tmp = unioned_model.get_polydata()
NUM_CAPS = {}
############################
# COMBINE FACES
############################
if not terminating:
    model.set_surface(tmp)
    model.compute_boundary_faces({})
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    while len(walls) > 1:
        target = walls[0]
        lose = walls[1]
        model.combine_faces(target,[lose])
        #combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
        #model.set_surface(combined)
        ids = model.get_face_ids()
        caps = model.identify_caps()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        print(walls)
    ids = model.get_face_ids()
    if {}:
        dmg.add_model('{}',model)
    if len(ids) > NUM_CAPS:
        face_cells = []
        for idx in ids:
            face = model.get_face_polydata(idx)
            cells = face.GetNumberOfCells()
            print(cells)
            face_cells.append(cells)
        data_to_remove = len(ids) - NUM_CAPS
        remove_list = []
        for i in range(data_to_remove):
            remove_list.append(ids[face_cells.index(min(face_cells))])
            face_cells[face_cells.index(min(face_cells))] += 1000
        print(remove_list)
        while len(remove_list) > 0:
            target = walls[0]
            lose = remove_list.pop(-1)
            model.combine_faces(target,[lose])
            #combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
            #model.set_surface(combined)
            print(remove_list)
        print(model.get_face_ids())
    ###############################
    # LOCAL SMOOTHING (not included)
    ###############################
    #smoothing_params = {{'method':'constrained', 'num_iterations':5, 'constrain_factor':0.2, 'num_cg_solves':30}}
    smooth_model = model.get_polydata()
    for idx, contour_set in enumerate(contour_list):
         if idx == 0:
              continue
         smoothing_params = {{'method':'constrained', 'num_iterations':3, 'constrain_factor':0.1+(0.9*(1-contour_set[0].get_radius()/contour_list[0][0].get_radius())), 'num_cg_solves':30}}
         smooth_model = geometry.local_sphere_smooth(smooth_model,contour_set[0].get_radius()*2,contour_set[0].get_center(),smoothing_params)
         print('local sphere smoothing {{}}'.format(idx))
    model.set_surface(smooth_model)

model = clean(model)
#model = remesh_wall(model,walls[0])
"""


old_solid="""###############################
# INITIALIZE MODELING KERNEL
###############################
modeler = modeling.Modeler(modeling.Kernel.POLYDATA)
walls = []
faces = []
refined = []
better = []
n_cells = {}
###############################
# REMESH FACES
###############################
cap_model = modeling.PolyData()
for vessel in capped_solids:
    faces = vessel.identify_caps()
    face_ids = [i+1 for i,x in enumerate(faces) if x]
    tmp = mesh_utils.remesh(vessel.get_polydata(),hmin={},hmax={})
    out = mesh_utils.remesh_faces(tmp,face_ids,{})
    cap_model.set_surface(out)
    all_faces = cap_model.get_face_ids()
    faces = cap_model.identify_caps()
    face_ids = [all_faces[i] for i,x in enumerate(faces) if x]
    for cap in face_ids:
        face_edge_size = {}
        cap_model.set_surface(out)
        face_elements = cap_model.get_face_polydata(cap).GetNumberOfCells()
        if face_elements < n_cells:
            print('Face Edge Size: {{}} ---> {{}}'.format(face_edge_size,round(face_edge_size*(face_elements/n_cells),5)))
            out = mesh_utils.remesh_faces(out,[cap],round(face_edge_size*(face_elements/n_cells),5))
    model = modeling.PolyData()
    model.set_surface(out)
    better.append(model.get_polydata())
##############################
# INITIAL CLEAN BAD EDGES
##############################
fe = vtk.vtkFeatureEdges()
fe.FeatureEdgesOff()
fe.BoundaryEdgesOn()
fe.NonManifoldEdgesOn()
clean = vtk.vtkCleanPolyData()
clean2 = vtk.vtkCleanPolyData()
fill = vtk.vtkFillHolesFilter()
tri = vtk.vtkTriangleFilter()
for idx,vessel in enumerate(better):
    fe.SetInputDataObject(vessel)
    fe.Update()
    BAD_CELLS = fe.GetOutput().GetNumberOfCells()
    print(BAD_CELLS)
    if BAD_CELLS > 0:
        print('Attempt initial clean')
        clean.SetInputDataObject(vessel)
        clean.Update()
        tmp_vessel = clean.GetOutput()
        fe.SetInputDataObject(tmp_vessel)
        BAD_CELLS = fe.GetOutput().GetNumberOfCells()
        print(BAD_CELLS)
        if BAD_CELLS > 0:
            print('Attempt initial hole filling')
            fill.SetInputDataObject(tmp_vessel)
            fill.Update()
            tri.SetInputDataObject(fill.GetOutput())
            tri.Update()
            clean2.SetInputDataObject(tri.GetOutput())
            clean2.SetConvertLinesToPoints(1)
            clean2.SetConvertPolysToLines(1)
            clean2.SetConvertStripsToPolys(1)
            clean2.SetPointMerging(1)
            tmp_vessel = clean2.GetOutput()
            fe.SetInputDataObject(tmp_vessel)
            BAD_CELLS = fe.GetOutput().GetNumberOfCells()
            print(BAD_CELLS)
        better[idx] = tmp_vessel
##############################
# UNION
##############################
model_1 = modeling.PolyData()
model_1.set_surface(better[0])
model_2 = modeling.PolyData()
model_2.set_surface(better[1])
unioned_model = modeler.union(model_1,model_2)
fill.SetHoleSize(fill.GetHoleSizeMaxValue())
clean.SetInputDataObject(unioned_model.get_polydata())
clean.Update()
#clean2 = vtk.vtkCleanPolyData()
region = vtk.vtkPolyDataConnectivityFilter()
strip = vtk.vtkStripper()
poly = vtk.vtkPolyData()
tmp = clean.GetOutput()
fe.SetInputDataObject(tmp)
fe.Update()
terminating = False
print('Union: {{}}\\nBad Edges: {{}}'.format(1,fe.GetOutput().GetNumberOfCells()))
for i in range(2,len(better)):
    model_1.set_surface(tmp)
    #model_1.compute_boundary_faces(angle=45)
    tmp = mesh_utils.remesh(model_1.get_polydata(),hmin=0.01,hmax=0.02)
    model_1.set_surface(tmp)
    model_2.set_surface(better[i])
    print('models created')
    unioned_model = modeler.union(model_1,model_2)
    print('union successful')
    fe.SetInputDataObject(unioned_model.get_polydata())
    fe.Update()
    print('Initial Bad Edges: {{}}'.format(fe.GetOutput().GetNumberOfCells()))
    clean.SetInputDataObject(unioned_model.get_polydata())
    clean.Update()
    tmp = clean.GetOutput()
    fe.SetInputDataObject(tmp)
    fe.Update()
    print('Union: {{}}\\nBad Edges: {{}}'.format(i,fe.GetOutput().GetNumberOfCells()))
    if fe.GetOutput().GetNumberOfCells() == 0:
        run_remesh = False
    if fe.GetOutput().GetNumberOfCells() > 0:
        number_bad_edges = fe.GetOutput().GetNumberOfLines()
        fill.SetInputDataObject(clean.GetOutput())
        fill.Update()
        #region.SetInputDataObject(fe.GetOutput())
        #region.SetExtractionMode(6)
        #region.Update()
        #strip.SetInputDataObject(region.GetOutput())
        #strip.Update()
        tri.SetInputDataObject(fill.GetOutput())
        tri.Update()
        clean2.SetInputDataObject(tri.GetOutput())
        clean2.SetConvertLinesToPoints(1)
        clean2.SetConvertPolysToLines(1)
        clean2.SetConvertStripsToPolys(1)
        clean2.SetPointMerging(1)
        clean2.Update()
        fe.SetInputDataObject(clean2.GetOutput())
        fe.Update()
        number_bad_edges = fe.GetOutput().GetNumberOfLines()
        tmp = clean2.GetOutput()
        print('Pedantic Cleaning: '+str(number_bad_edges))
        colors = vtk.vtkNamedColors()
        actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputDataObject(tmp)
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('red'))
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('white'))
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('Polydata')
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        renderer.AddActor(actor)
        render_window.Render()
        interactor.Start()
        colors = vtk.vtkNamedColors()
        actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputDataObject(better[i])
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('blue'))
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('white'))
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('Polydata')
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        renderer.AddActor(actor)
        render_window.Render()
        interactor.Start()
        if number_bad_edges:
            terminating = True
        else:
            terminating = False
    #colors = vtk.vtkNamedColors()
    #actor = vtk.vtkActor()
    #mapper = vtk.vtkPolyDataMapper()
    #mapper.SetInputDataObject(tmp)
    #actor.SetMapper(mapper)
    #actor.GetProperty().SetColor(colors.GetColor3d('red'))
    #renderer = vtk.vtkRenderer()
    #renderer.SetBackground(colors.GetColor3d('white'))
    #render_window = vtk.vtkRenderWindow()
    #render_window.AddRenderer(renderer)
    #render_window.SetWindowName('Polydata')
    #interactor = vtk.vtkRenderWindowInteractor()
    #interactor.SetRenderWindow(render_window)
    #renderer.AddActor(actor)
    #render_window.Render()
    #interactor.Start()
############################
# COMBINE FACES
############################
if not terminating:
    model.set_surface(tmp)
    model.compute_boundary_faces({})
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    while len(walls) > 1:
        target = walls[0]
        lose = walls[1]
        combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
        model.set_surface(combined)
        ids = model.get_face_ids()
        caps = model.identify_caps()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        print(walls)
    ids = model.get_face_ids()
    if {}:
        dmg.add_model({},model)
    if len(ids) > len(better)+2:
        face_cells = []
        for idx in ids:
            face = model.get_face_polydata(idx)
            cells = face.GetNumberOfCells()
            print(cells)
            face_cells.append(cells)
        data_to_remove = len(ids) - (len(better)+2)
        remove_list = []
        for i in range(data_to_remove):
            remove_list.append(ids[face_cells.index(min(face_cells))])
            face_cells[face_cells.index(min(face_cells))] += 1000
        print(remove_list)
        while len(remove_list) > 0:
            target = walls[0]
            lose = remove_list.pop(-1)
            combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
            model.set_surface(combined)
            print(remove_list)
        print(model.get_face_ids())
    ###############################
    # LOCAL SMOOTHING (not included)
    ###############################
    #smoothing_params = {{'method':'constrained', 'num_iterations':5, 'constrain_factor':0.2, 'num_cg_solves':30}}
    smooth_model = model.get_polydata()
    for idx, contour_set in enumerate(contour_list):
         if idx == 0:
              continue
         smoothing_params = {{'method':'constrained', 'num_iterations':3, 'constrain_factor':0.1+(0.9*(1-contour_set[0].get_radius()/contour_list[0][0].get_radius())), 'num_cg_solves':30}}
         smooth_model = geometry.local_sphere_smooth(smooth_model,contour_set[0].get_radius()*2,contour_set[0].get_center(),smoothing_params)
         print('local sphere smoothing {{}}'.format(idx))
    model.set_surface(smooth_model)

"""
