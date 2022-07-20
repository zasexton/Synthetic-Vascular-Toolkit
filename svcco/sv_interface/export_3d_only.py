export_3d_only = """import sv
import vtk
import os

Modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA)
model = Modeler.read("{}")

mass = vtk.vtkMassProperties()

model.compute_boundary_faces(45)

face_types = []
face_areas = []

face_ids = model.get_face_ids()
caps = model.identify_caps()

inlet_cap_area = -100
inlet_cap = 1

for id in face_ids:
     mass.SetInputData(model.get_face_polydata(id))
     mass.Update()
     SA = mass.GetSurfaceArea()
     face_areas.append(SA)
     if not caps[id-1]:
         face_types.append("Wall")
         print("Wall found")
     else:
         face_types.append("Cap")
         print("Cap found")
         if SA > inlet_cap_area:
             inlet_cap = id
             inlet_cap_area = SA

face_types[inlet_cap-1] = "Inlet"

#---------------------
# REMESHING
#---------------------
elements_on_smallest_face = 20

hmin = min(face_areas)/elements_on_smallest_face
hmax = hmin

optimized_polydata = sv.mesh_utils.remesh(model.get_polydata(),hmin=hmin,hmax=hmax)

model.set_surface(optimized_polydata)
"""
