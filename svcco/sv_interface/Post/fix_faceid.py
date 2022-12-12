#!/usr/bin/env python

import pdb
import numpy as np
import os
import shutil
import glob

from get_database import Database, input_args
from vtk_functions import read_geo, write_geo

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v


def main(db, geometries):

    for geo in geometries:
        # get all surfaces
        surfaces = glob.glob(os.path.join(db.get_surface_dir(geo), '*.vtp'))
        surf_geos = {}

        for f_surf in surfaces:
            name_osmsc = os.path.basename(f_surf)
            surf = read_geo(f_surf).GetOutput()
            surf_c = surf.GetCellData()
            face_id = v2n(surf_c.GetArray('BC_FaceID'))
            face_id_u = np.unique(face_id)
            surf_geos[name_osmsc] = surf

            # wall has not faceid 0 as it should
            if 'all_exterior' in name_osmsc and 0 not in face_id_u:
                break
        else:
            continue

        print('Fixing geometry ' + geo)
        id_wall = v2n(surf_geos['wall.vtp'].GetCellData().GetArray('GlobalElementID')).astype(np.int64)
        id_all = v2n(surf_geos['all_exterior.vtp'].GetCellData().GetArray('GlobalElementID')).astype(np.int64)
        face_id = v2n(surf_geos['all_exterior.vtp'].GetCellData().GetArray('BC_FaceID')).astype(np.int64)

        # find surface indices in all_exterior
        index = np.argsort(id_all)
        search = np.searchsorted(id_all[index], id_wall)
        ind_wall = index[search]

        # set wall indices to zero
        face_id[ind_wall] = 0

        # export BC_FaceID
        out = n2v(face_id)
        out.SetName('BC_FaceID')
        surf_geos['all_exterior.vtp'].GetCellData().AddArray(out)

        # # backup original file
        f_out = os.path.join(db.get_surface_dir(geo), 'all_exterior.vtp')
        shutil.copy(f_out, f_out + '_original')

        # # save new file
        write_geo(f_out, surf_geos['all_exterior.vtp'])


if __name__ == '__main__':
    descr = 'Fix wrong scale'
    d, g, _ = input_args(descr)
    main(d, g)
