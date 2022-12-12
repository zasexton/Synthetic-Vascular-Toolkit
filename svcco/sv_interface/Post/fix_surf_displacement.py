#!/usr/bin/env python

import numpy as np
import os
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import Database, input_args
from vtk_functions import read_geo, write_geo


def main(db, geometries):
    for geo in geometries:
        print('Fixing geometry ' + geo)

        # read volumetric geometry
        vol = read_geo(db.get_volume(geo))
        vol_id = v2n(vol.GetOutput().GetPointData().GetArray('GlobalNodeID')).astype(int)
        vol_points = v2n(vol.GetOutput().GetPoints().GetData())

        for surf_path in db.get_surfaces(geo):
            # read surface geometry
            surf = read_geo(surf_path)
            surf_id = v2n(surf.GetOutput().GetPointData().GetArray('GlobalNodeID')).astype(int)
            surf_points = v2n(surf.GetOutput().GetPoints().GetData())

            # find surface indices in volume
            index = np.argsort(vol_id)
            search = np.searchsorted(vol_id[index], surf_id)
            ind_surf = index[search]

            # get displacement between meshes
            disp = vol_points[ind_surf] - surf_points

            # check if displacement is rigid body displacement
            disp_norm = np.linalg.norm(disp, axis=1)
            disp_diff = np.max(np.abs(disp_norm - np.mean(disp_norm)))
            assert disp_diff < 1.0e-3, 'displacement is not a rigid body translation'

            # move points
            surf.GetOutput().GetPoints().SetData(n2v(vol_points[ind_surf]))

            # write to file
            write_geo(surf_path, surf.GetOutput())


if __name__ == '__main__':
    descr = 'Fix displacement between volume and surface geometry'
    d, g, _ = input_args(descr)
    main(d, g)
