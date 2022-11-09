# Remeshing utility based on MMG executables

import os
import stat
import platform
import subprocess
import pyvista as pv
import io
import sys
filepath = os.path.abspath(__file__)
dirpath  = os.path.dirname(filepath)

def remesh_surface(pv_polydata_object,auto=True,hausd=0.01,verbosity=1):
    _mesh_ = pv.PolyData(pv_polydata_object.points,pv_polydata_object.faces)
    pv.save_meshio("tmp.mesh",_mesh_)
    if platform.system() == 'Windows':
        _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmgs_O3.exe"
    elif platform.system() == "Linux":
        _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmgs_O3"
    elif platform.system() == "Darwin":
        _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmgs_O3"
    devnull = open(os.devnull, 'w')
    if verbosity == 0:
        try:
            subprocess.check_call([_EXE_,"tmp.mesh","-hausd",str(hausd),"-v",str(verbosity)],stdout=devnull,stderr=devnull)
        except:
            os.chmod(_EXE_,stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)
            subprocess.check_call([_EXE_,"tmp.mesh","-hausd",str(hausd),"-v",str(verbosity)],stdout=devnull,stderr=devnull)
    else:
        try:
            subprocess.check_call([_EXE_,"tmp.mesh","-hausd",str(hausd),"-v",str(verbosity)])
        except:
            os.chmod(_EXE_,stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)
            subprocess.check_call([_EXE_,"tmp.mesh","-hausd",str(hausd),"-v",str(verbosity)])
    clean_medit("tmp.o.mesh")
    remeshed = pv.read("tmp.o.mesh")
    remeshed_surface = remeshed.extract_surface()
    remeshed_surface.triangulate(inplace=True)
    remeshed_surface.clear_data()
    os.remove("tmp.mesh")
    os.remove("tmp.o.sol")
    os.remove("tmp.o.mesh")
    return remeshed_surface

def clean_medit(filename):
    file = open(filename)
    lines = file.readlines()
    file.close()
    keywords_index = []
    for i,s in enumerate(lines):
        if s[0].isnumeric():
            pass
        elif s[0] == '-':
            pass
        elif s[0] == '\n':
            pass
        elif s[0] == '\n':
            pass
        else:
            keywords_index.append(i)
    new_file = open(filename,'w+')
    new_lines = []
    for i,o in enumerate(keywords_index):
        if lines[o] == 'RequiredVertices\n':
            pass
        elif lines[o] == 'Ridges\n':
            pass
        elif lines[o] == 'Tangents\n':
            pass
        elif lines[o] == 'TangentAtVertices\n':
            pass
        else:
            if o == keywords_index[-1]:
                new_lines.append(lines[o])
            else:
                new_lines.extend(lines[o:keywords_index[i+1]])
    new_file.writelines(new_lines)
    new_file.close()
