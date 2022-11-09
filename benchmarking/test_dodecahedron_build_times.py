# Script to obtain build-time data for convex cone geometry
import pyvista as pv
import os
from pickle import dump
import svcco
import numpy as np

name = 'dodecahedron'
shape = pv.Dodecahedron().triangulate().subdivide(3)

######################
number = 100
######################

FOLDER = 'convex_data'

if not os.path.isdir(os.getcwd()+os.sep+FOLDER):
    os.mkdir(os.getcwd()+os.sep+FOLDER)

if not os.path.isdir(os.getcwd()+os.sep+FOLDER+os.sep+name):
    os.mkdir(os.getcwd()+os.sep+FOLDER+os.sep+name)
    os.mkdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'times')
    os.mkdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data')

s = svcco.surface()
s.set_data(10*shape.points,normals=shape.point_normals)
s.solve()
s.build()


# Warm-up build to catch jit-compilation of functions
t = svcco.tree()
t.set_boundary(s)
t.parameters['Qterm'] *= 40
t.set_assumptions(convex=True)
t.set_root()
t.n_add(400)

EXISTING_TIMES     = len(os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'times'))
EXISTING_DATA      = len(os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'))
EXISTING_PERFUSION = len(list(filter(lambda n: 'perfusion' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'))))
EXISTING_MESH      = len(list(filter(lambda n: 'mesh' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'))))
EXISTING_TREE      = len(list(filter(lambda n: 'tree' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'))))
EXISTING_RAW       = len(list(filter(lambda n: 'raw' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'))))

for i in range(number):
    t = svcco.tree()
    t.set_boundary(s)
    t.parameters['Qterm'] *= 40
    t.set_assumptions(convex=True)
    t.set_root()
    t.n_add(8000)
    file = open(os.getcwd() + os.sep +FOLDER +os.sep+name+os.sep+'times'+os.sep+'{}_times_8000_vessels_{}.pkl'.format(name,i+EXISTING_TIMES),"wb")
    dump(t.time,file)
    file.close()

    id,vol = svcco.perfusion_territory(t,mesh_file=os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'+os.sep+name+'_mesh_{}.vtu'.format(i+EXISTING_MESH),
                                         tree_file=os.getcwd()+os.sep+FOLDER+os.sep+name+os.sep+'data'+os.sep+name+'_tree_{}.vtp'.format(i+EXISTING_TREE))
    file = open(os.getcwd() + os.sep + FOLDER +os.sep+name+os.sep+'data'+os.sep+name+'_perfusion_{}.pkl'.format(i+EXISTING_PERFUSION),"wb")
    dump(vol,file)
    file.close()

    np.savetxt(os.getcwd() + os.sep + FOLDER +os.sep+name+os.sep+'data'+os.sep+name+'_raw_{}.csv'.format(i+EXISTING_RAW),t.data,delimiter=',')

grid = s.pv_polydata.delaunay_3d()
surf = grid.extract_surface()
stats = [s.pv_polydata.area,s.pv_polydata.volume,surf.area,grid.volume]
file = open(os.getcwd() + os.sep + FOLDER+os.sep+name+os.sep+'data'+os.sep+'stats.pkl',"wb")
dump(stats,file)
file.close()
