import os

brain_files = os.listdir(os.getcwd()+os.sep+'brain_testing')

if not os.path.isdir(os.getcwd()+os.sep+'brain_data'):
    os.mkdir('brain_data')

CODE ="""
# Code to generate vascular networks for Third Ventricle

import svcco
import os
import numpy as np
from pickle import dump

######################
number = 100
######################

FOLDER   = '{}'
OBJ_FILE = '{}'
NAME     = '{}_8000_vessels_'

if not os.path.isdir(os.getcwd()+os.sep+FOLDER):
    os.mkdir(os.getcwd()+os.sep+FOLDER)
    os.mkdir(os.getcwd()+os.sep+FOLDER+os.sep+'times')
    os.mkdir(os.getcwd()+os.sep+FOLDER+os.sep+'data')

FILE_TIME_PREFIX = os.getcwd()+os.sep+FOLDER+os.sep+'times'
FILE_DATA_PREFIX = os.getcwd()+os.sep+FOLDER+os.sep+'data'

s = svcco.surface()
s.load(OBJ_FILE)

s.solve()
s.build()

t = svcco.tree()
t.set_boundary(s)

t.set_root()
t.n_add(500)

EXISTING_TIMES     = len(os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'times'))
EXISTING_DATA      = len(os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'data'))
EXISTING_PERFUSION = len(list(filter(lambda n: 'perfusion' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'data'))))
EXISTING_MESH      = len(list(filter(lambda n: 'mesh' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'data'))))
EXISTING_TREE      = len(list(filter(lambda n: 'tree' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'data'))))
EXISTING_RAW       = len(list(filter(lambda n: 'raw' in n,os.listdir(os.getcwd()+os.sep+FOLDER+os.sep+'data'))))

for i in range(number):
    t = svcco.tree()
    t.set_boundary(s)
    t.set_assumptions(convex=True)
    t.set_root()
    t.n_add(8000)
    file = open(os.getcwd() + os.sep + FOLDER +os.sep+'times'+os.sep+NAME+'{{}}.pkl'.format(i+EXISTING_TIMES),"wb")
    dump(t.time,file)
    file.close()

    id,vol = svcco.perfusion_territory(t,mesh_file=os.getcwd()+os.sep+FOLDER+os.sep+'data'+os.sep+NAME+'mesh_{{}}.vtu'.format(i+EXISTING_MESH),
                                                 tree_file=os.getcwd()+os.sep+FOLDER+os.sep+'data'+os.sep+NAME+'tree_{{}}.vtp'.format(i+EXISTING_TREE))
    file = open(os.getcwd() + os.sep + FOLDER +os.sep+'data'+os.sep+NAME+'perfusion_{{}}.pkl'.format(i+EXISTING_PERFUSION),"wb")
    dump(vol,file)
    file.close()

    np.savetxt(os.getcwd() + os.sep + FOLDER +os.sep+'data'+os.sep+NAME+'raw_{{}}.csv'.format(i+EXISTING_RAW),t.data,delimiter=',')

grid = s.pv_polydata.delaunay_3d()
surf = grid.extract_surface()
stats = [s.pv_polydata.area,s.pv_polydata.volume,surf.area,grid.volume]
file = open(os.getcwd() + os.sep + FOLDER +os.sep+'data'+os.sep+'stats.pkl',"wb")
dump(stats,file)
file.close()
"""

for file in brain_files:
    part_name = file.split('_')[-1].split('.')[0].replace(' ','_')
    full_part_path = (os.getcwd()+os.sep+'brain_testing'+os.sep+file).replace(os.sep,os.sep+os.sep)
    f = open(os.getcwd()+os.sep+'brain_data'+os.sep+part_name+'.py','w+')
    f.writelines([CODE.format(part_name,full_part_path,part_name)])
    f.close()
