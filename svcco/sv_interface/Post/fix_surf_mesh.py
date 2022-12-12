#!/usr/bin/env python

import paraview.simple as pv
import numpy as np
import os
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from common import input_args
from get_bc_integrals import read_geo, write_geo
from get_database import Database


def write_geo_pv(fname_in, fname_out, ele_id, aname):
    """
    Read/write geometry and remove/add cell array.
    Not sure how replacing array works with ParaView reader, so read again here using vtk
    :param fname_in: input filename
    :param fname_out: output filename
    :param ele_id: cell array to be written
    :param aname: cell array name
    :return: void
    """
    out_array = n2v(ele_id)
    out_array.SetName(aname)
    reader, _, reader_cell = read_geo(fname_in)
    reader_cell.RemoveArray(aname)
    reader_cell.AddArray(out_array)
    write_geo(fname_out, reader)


def read_geo_pv(fname):
    """
    Read geometry using built-in ParaView functions. Distinguish between 2D/3D meshes.
    In case of 3D mesh, extract surface
    :param fname: geometry file name
    :return: reader (for writer), reader (to extract data), point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = pv.XMLPolyDataReader(FileName=[fname])
    elif ext == '.vtu':
        reader_3d = pv.XMLUnstructuredGridReader(FileName=[fname])
        reader = pv.ExtractSurface(Input=reader_3d)
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader_fetch = pv.servermanager.Fetch(reader)
    return reader, reader_fetch, reader_fetch.GetPointData(), reader_fetch.GetCellData()


def is_unique(ids):
    """
    Check if list of ids is unqiue
    :param ids: 1d numpy array of ids
    :return: True if ids is unique
    """
    return np.unique(ids).shape[0] == ids.shape[0]


def read_array(data, name):
    """
    Read vtk array from data and convert to numpy int array
    :param data: vtk cell/point data
    :param name: name of array
    :return: 1-d numpy int array
    """
    return v2n(data.GetArray(name)).astype(int)


def get_connectivity(reader):
    """
    Extract surface cell connectivity from mesh
    :param reader: vtk reader
    :return: [N,3] numpy array with N surface elements in mesh containing point ids
    """
    # loop all cells
    cells = []
    for i in range(reader.GetNumberOfCells()):
        c = reader.GetCell(i)

        # loop all points of current cell
        points = []
        for j in range(c.GetNumberOfPoints()):
            points.append(c.GetPointIds().GetId(j))

        # extract only surface elements
        if len(points) == 3:
            cells.append(points)
    return np.array(cells)


def get_ids(fpath):
    """
    Read geometry and extract relevant node/element ids
    :param fpath: filename
    :return: vtk reader,  GlobalElementID, element connectivity in GlobalNodeID (points sorted for each element)
    """
    reader, reader_fetch, point_data, cell_data = read_geo_pv(fpath)

    node_id = read_array(point_data, 'GlobalNodeID')
    assert is_unique(node_id), 'GlobalNodeID is not unique'

    cell_id = read_array(cell_data, 'GlobalElementID')
    connectivity = get_connectivity(reader_fetch)

    return reader, cell_id, np.sort(node_id[connectivity], axis=1)


def fix_surfaces(fpath_vol, fpath_surf, folder_out):
    """
    Problem: GlobalElementID in 2D meshes only stored with 6-digit precision
    Solution: Compare elements in 2D and 3D meshes to restore GlobalElementID
    :param fpath_vol: filename of 3D mesh
    :param fpath_surf: filenames of corresponding 2D meshes
    :param folder_out: output folder
    :return: True of all 2D meshes were converted successfully
    """
    # read volume mesh
    vol_reader, vol_cell, vol_conn = get_ids(fpath_vol)

    # loop all corresponding surface meshes
    for f in fpath_surf:
        surf_fname = os.path.basename(f)
        if not surf_fname == 'wall.vtp':
            continue
        print('  mesh ' + surf_fname)

        # get surface mesh
        surf_reader, surf_cell, surf_conn = get_ids(f)

        # skipping (GlobalElementID already unique)
        if is_unique(surf_cell):
            surf_cell_new = surf_cell

        # match GlobalElementID in surface mesh with volume mesh
        else:
            # todo: this can probably be done without loops
            surf_cell_new = np.zeros(surf_cell.shape, dtype=int)
            for i, cell in enumerate(surf_conn):
                found = vol_cell[(vol_conn == cell).all(axis=1)]

                # check if element was found in both 2d and 3d mesh
                if not found.shape[0] == 1:
                    print('    error: 2d and 3d mesh are not identical')
                    return False

                surf_cell_new[i] = found[0]

            # compare original 2D GlobalElementID to the fixed one
            assert np.max(np.abs(surf_cell_new - surf_cell)) <= 5, 'round-off error bigger than 5'

        # export surface
        fpath_out = os.path.join(folder_out, surf_fname)
        write_geo_pv(f, fpath_out, surf_cell_new, 'GlobalElementID')
        del surf_reader
    else:
        del vol_reader
        return True


def is_fixed(folder_out, fpath_surf):
    """
    Check if all surfaces meshes have been fixed for a geometry
    :param folder_out: output folder for fixed meshes
    :param fpath_surf: list of surface mesh files
    :return: True if ALL surface meshes have been fixed
    """
    for f in fpath_surf:
        if not os.path.exists(os.path.join(folder_out, os.path.basename(f))):
            return False
    else:
        return True


def main(db, geometries):
    # loop all geometries in repository
    database_surf_fixed = {}
    for geo in geometries:
        print('Fixing geometry ' + geo)
        folder_out = os.path.join(db.fpath_gen, 'surfaces', geo)
        try:
            os.mkdir(folder_out)
        except OSError:
            pass

        # get volume mesh path
        fpath_vol = db.get_volume(geo)

        # get all surface mesh paths
        fpath_surf = db.get_surfaces_upload(geo)

        if is_fixed(folder_out, fpath_surf):
            print('  skipping (already fixed)')
            database_surf_fixed[geo] = True
        elif not os.path.exists(fpath_vol):
            print('  skipping (no 3D mesh found)')
            database_surf_fixed[geo] = False
        else:
            database_surf_fixed[geo] = fix_surfaces(fpath_vol, fpath_surf, folder_out)

    fpath_report = os.path.join(db.fpath_gen, 'database', 'surf_fixed')
    np.save(fpath_report, database_surf_fixed)


if __name__ == '__main__':
    descr = 'Fix wrong GlobalElementID'
    d, g, _ = input_args(descr)
    main(d, g)