loft="""options = geometry.LoftNurbsOptions()
{}
loft_surfaces = []
loft_solids = []
capped = []
capped_solids = []
faces = []
for contour_polydata_list in contour_polydata:
    loft_surfaces.append(geometry.loft_nurbs(polydata_list=contour_polydata_list,loft_options=options))
    loft_solids.append(modeling.PolyData())
    loft_solids[-1].set_surface(surface=loft_surfaces[-1])
    capped.append(vmtk.cap(surface=loft_solids[-1].get_polydata(),use_center=False))
    capped_solids.append(modeling.PolyData())
    capped_solids[-1].set_surface(surface=capped[-1])
    faces.append(capped_solids[-1].compute_boundary_faces(angle={}))
"""
