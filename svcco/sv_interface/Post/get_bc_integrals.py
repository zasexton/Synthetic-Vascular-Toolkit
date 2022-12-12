#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from .get_database import Database, input_args
from .vtk_functions import Integration, read_geo, write_geo, threshold, calculator, cut_plane
from .vtk_to_xdmf import split


def transfer_solution(node_trg, node_src, res_fields):
    """
    Transfer point data from volume mesh to surface mesh using GlobalNodeID
    Args:
        node_trg: surface mesh
        node_src: volume mesh
        res_fields: point data names to transfer
    """
    # get global node ids in both meshes
    nd_id_trg = v2n(node_trg.GetArray('GlobalNodeID')).astype(int)
    nd_id_src = v2n(node_src.GetArray('GlobalNodeID')).astype(int)

    # map volume mesh to surface mesh
    index = np.argsort(nd_id_src)
    search = np.searchsorted(nd_id_src[index], nd_id_trg)
    mask = index[search]

    # transfer results from volume mesh to surface mesh
    for i in range(node_src.GetNumberOfArrays()):
        res_name = node_src.GetArrayName(i)
        if res_name.split('_')[0] in res_fields:
            # read results from volume mesh
            res = v2n(node_src.GetArray(res_name))

            # create array to output surface mesh results
            out_array = n2v(res[mask])
            out_array.SetName(res_name)
            node_trg.AddArray(out_array)


def sort_faces(res_faces, area):
    """
    Arrange results from surface integration in matrix
    Args:
        res_faces: dictionary with key: cap id, value: result at a certain time step
        area: cross-sectional area of each cap

    Returns:
        dictionary with results as keys and matrix at all surfaces/time steps as values
    """
    # get time steps
    times = []
    for n in res_faces[list(res_faces)[0]].keys():
        times += [split(n)[0]]
    times = np.unique(np.array(times))

    # sort data in arrays according to time steps
    res_array = {'time': times}
    dim = (times.shape[0], max(list(res_faces.keys())))

    for f, f_res in res_faces.items():
        for res_name, res in f_res.items():
            time, name = split(res_name)
            if name not in res_array:
                res_array[name] = np.zeros(dim)
            res_array[name][float(time) == times, f - 1] = res

    # repeat area for all time steps to match format
    res_array['area'] = np.zeros(dim)
    for f, f_res in area.items():
        res_array['area'][:, f - 1] = f_res

    return res_array


def get_res_names(inp, res_fields):
    # result name list
    res = []

    # get integral for each result
    for i in range(inp.GetPointData().GetNumberOfArrays()):
        res_name = inp.GetPointData().GetArrayName(i)
        field = res_name.split('_')[0]
        num = res_name.split('_')[-1]

        # check if field should be added to output
        if field in res_fields:
            try:
                float(num)
                res += [res_name]
            except ValueError:
                pass

    return res


def integrate_surfaces(surf, cell_surf, res_fields, face_array='BC_FaceID'):
    """
    Integrate desired fields on all caps of surface mesh (as defined by BC_FaceID)
    Args:
        surf: reader for surface mesh
        cell_surf: surface mesh cell data
        res_fields: result fields to extract
        face_array: name of array containing face ids
    Returns:
        dictionary with result fields as keys and matrices with all faces and time steps as matrices
    """
    # generate surface normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf)
    normals.Update()

    # recursively add calculators for normal velocities
    calc = normals
    for v in get_res_names(surf, 'velocity'):
        calc = calculator(calc, 'Normals.' + v, ['Normals', v], 'normal_' + v)

    # get all output array names
    res_names = get_res_names(surf, res_fields)

    # boundary faces
    faces = np.unique(v2n(cell_surf.GetArray(face_array)).astype(int))
    res = {}
    area = {}

    # loop boundary faces
    for f in faces:
        # skip face 0 (vessel wall)
        if f:
            # threshhold face
            thresh = threshold(calc.GetOutput(), f, face_array)

            # integrate over selected face (separately for pressure and velocity)
            integrator = Integration(thresh)

            # perform integration
            res[f] = {}
            for r in res_names:
                res[f][r] = integrator.evaluate(r)

            # store cross-sectional area
            area[f] = integrator.area()

    return sort_faces(res, area)


def integrate_bcs(fpath_surf, fpath_vol, res_fields, debug=False, debug_out='', face_array='BC_FaceID'):
    """
    Perform all steps necessary to get results averaged on caps
    Args:
        fpath_surf: surface geometry file
        fpath_vol: volume geometry file
        res_fields: results to extract
        debug: bool if debug geometry should be written
        debug_out: path for debug geometry
        face_array: name of array containing face ids
    Returns:
        dictionary with result fields as keys and matrices with all faces and time steps as matrices
    """
    if not os.path.exists(fpath_surf) or not os.path.exists(fpath_vol):
        return None

    # read surface and volume meshes
    surf = read_geo(fpath_surf).GetOutput()
    vol = read_geo(fpath_vol).GetOutput()

    # transfer solution from volume mesh to surface mesh
    transfer_solution(surf.GetPointData(), vol.GetPointData(), res_fields)

    # integrate data on boundary surfaces
    res_faces = integrate_surfaces(surf, surf.GetCellData(), res_fields, face_array=face_array)

    # write results for debugging in paraview
    if debug:
        write_geo(debug_out, surf)
    return res_faces


def main(db, geometries):
    """
    Loop all geometries in database
    """
    for geo in geometries:
        print('Processing ' + geo)

        # file paths
        fpath_surf = db.get_surfaces(geo, 'all_exterior')
        fpath_vol = db.get_volume(geo)

        bc_flow = integrate_bcs(fpath_surf, fpath_vol, ['pressure', 'velocity'])

        if bc_flow is not None:
            np.save(db.get_bc_flow_path(geo), bc_flow)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
