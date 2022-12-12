#!/usr/bin/env python

import numpy as np
import os
import glob
import pdb
import vtk

from get_database import Database, input_args
from vtk_functions import read_geo, write_geo


def invert_traingles(surf, name):
    """
    Invert triangles by flipping nodes from  to
    """
    cells = vtk.vtkCellArray()
    for i in range(surf.GetNumberOfCells()):
        assert surf.GetCellType(i) == 5, 'not a triangle mesh'

        # could be done easier with DeepCopy but throws seg fault
        ids = surf.GetCell(i).GetPointIds()
        id0 = ids.GetId(0)
        id1 = ids.GetId(1)
        id2 = ids.GetId(2)

        tri = vtk.vtkTriangle()
        if 'all_exterior' in name or 'wall' in name:
            tri.GetPointIds().SetId(0, id2)
            tri.GetPointIds().SetId(1, id1)
            tri.GetPointIds().SetId(2, id0)
        else:
            tri.GetPointIds().SetId(0, id1)
            tri.GetPointIds().SetId(1, id2)
            tri.GetPointIds().SetId(2, id0)
        cells.InsertNextCell(tri)
    surf.SetPolys(cells)
    surf.Modified()


def fix_surf_orientation(db, geo):
    # get all surfaces
    surfaces = glob.glob(os.path.join(db.get_surface_dir(geo), '*.vtp'))

    for f_surf in surfaces:
        name = os.path.basename(f_surf)
        # if name == 'btrunk.vtp' or name == 'carotid.vtp':
        print('  fixing ' + name)

        surf = read_geo(f_surf).GetOutput()
        invert_traingles(surf, name)

        f_out = os.path.join(db.fpath_gen, 'surfaces', geo + '_flipped', name)
        write_geo(f_out, surf)


def main(db, geometries):
    for geo in geometries:
        print('Fixing geometry ' + geo)

        fix_surf_orientation(db, geo)


if __name__ == '__main__':
    descr = 'Fix orientation of surface triangles'
    d, g, _ = input_args(descr)
    main(d, g)
