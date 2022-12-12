#!/usr/bin/env python

import numpy as np
import os
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import Database
from vtk_functions import read_geo, write_geo, extract_surface
from common import input_args


def main(db, geometries):
    for geo in geometries:
        print('Fixing geometry ' + geo)

        # # read volume mesh and delete result arrays
        # read_all, node_all, cell_all = read_geo(db.get_volume(geo))
        # for i in range(node_all.GetNumberOfArrays() - 1):
        #     node_all.RemoveArray(1)
        # for i in range(cell_all.GetNumberOfArrays() - 1):
        #     cell_all.RemoveArray(1)
        #
        # # extract surface from volume mesh
        # surf = extract_surface(read_all)
        read_all, node_all, cell_all = read_geo(os.path.join(db.fpath_gen, 'surfaces', geo, 'all_exterior.vtp'))

        # get GlobalElementID to match with surfaces
        surf_eid = v2n(cell_all.GetArray('GlobalElementID'))

        # initialize BC_FaceID
        surf_fid = np.zeros(surf_eid.shape, dtype='int32')

        # loop all surface meshes
        for s in db.get_surfaces(geo):
            # skip the broken surface:
            sname = os.path.basename(s)
            if sname == 'all_exterior.vtp' or sname == 'wall.vtp':
                continue

            # read surface mesh
            _, _, cell_s = read_geo(s)
            s_fid = v2n(cell_s.GetArray('BC_FaceID'))
            s_eid = v2n(cell_s.GetArray('GlobalElementID'))

            # find surface indices in all_exterior
            index = np.argsort(surf_eid)
            search = np.searchsorted(surf_eid[index], s_eid)
            ind_surf = index[search]

            # make sure there is only one BC_FaceID in surface
            s_id = np.unique(s_fid)
            assert s_id.shape[0] == 1, 'surface ' + s + ' BC_FaceID not unique'

            # change BC_FaceID
            surf_fid[ind_surf] = int(s_id[0])

        # export
        out_array = n2v(surf_fid)
        out_array.SetName('BC_FaceID')
        cell_all.AddArray(out_array)
        write_geo(os.path.join(db.fpath_gen, 'surfaces', geo, 'all_exterior_BC_FaceID.vtp'), read_all)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
