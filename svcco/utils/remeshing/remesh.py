# Remeshing utility based on MMG executables

import os
import platform
import subprocess
import pyvista as pv

filepath = os.path.abspath(__file__)
dirpath  = os.path.dirname(filepath)

def remesh_surface(pv_polydata_object,hausd=0.01):
    pv.save_meshio("tmp.mesh",pv_polydata_object)
    if platform.system() == 'Windows':
        _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmgs_O3.exe"
    elif platform.system() == "Linux":
        _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmgs_O3"
    elif platform.system() == "Darwin":
        _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmgs_O3"
    subprocess.check_call([_EXE_,"tmp.mesh","-hausd",str(hausd)])
    remeshed = pv.read("tmp.o.mesh")
    remeshed_surface = remeshed.extract_surface()
    remeshed_surface.triangulate(inplace=True)
    remeshed_surface.clear_data()
    os.remove("tmp.mesh")
    os.remove("tmp.o.sol")
    os.remove("tmp.o.mesh")
    return remeshed_surface
