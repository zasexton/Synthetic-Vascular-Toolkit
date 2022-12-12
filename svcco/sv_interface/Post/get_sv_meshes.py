#!/usr/bin/env python
import vtk
import os
import meshio
import argparse
import glob
import numpy as np
import pdb
from collections import defaultdict

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_bc_integrals import get_res_names, read_geo, write_geo
from get_database import Database, input_args
from vtk_functions import read_geo, write_geo, cell_connectivity, collect_arrays


def jacobian_positive(points, tets):
    """
    Check if Jacobian determinant is positive everywhere
    """
    jac = np.zeros((tets.shape[0], 3, 3))
    for i, p in enumerate([[1, 0], [2, 0], [3, 0]]):
        jac[:, :, i] = points[tets[:, p[0]]] - points[tets[:, p[1]]]
    return np.sum(np.linalg.det(jac) <= 0.0) == 0


def get_last_timestep(res, field):
    """
    Return the last time step of a field
    """
    return [k for k in sorted(res.keys()) if field in k][-1]


def get_last_result(fpath):
    """
    Return the results of the last time step for pressure and velocity
    """
    if not os.path.exists(fpath):
        raise ValueError('No results found in ' + fpath)
    geo = read_geo(fpath).GetOutput()
    res = collect_arrays(geo.GetPointData())
    return res[get_last_timestep(res, 'pressure')], res[get_last_timestep(res, 'velocity')]


def get_initial_conditions(db, geo, point_data, ini):
    """
    Generate initial conditions from database for pressure and velocity
    """
    # load initial conditions
    ini_val = defaultdict(dict)

    # zero initial conditions (as in SimVascular)
    if 'zero' in ini.values():
        n_points = len(point_data['GlobalNodeID'])
        ini_val['pressure']['zero'] = np.zeros(n_points)
        ini_val['velocity']['zero'] = 0.0001 * np.ones((n_points, 3))

    # initial conditions from file
    ini_paths = {'steady': db.get_initial_conditions_steady(geo),
                 'steady0': db.get_initial_conditions_steady0(geo),
                 'irene': db.get_initial_conditions_irene(geo),
                 'osmsc': db.get_volume(geo),
                 'asymp': db.get_asymptotic(geo)}

    for i, f_path in ini_paths.items():
        if i in ini.values():
            ini_val['pressure'][i], ini_val['velocity'][i] = get_last_result(f_path)

    fpath_red = db.get_initial_conditions_pressure(geo)
    if '1d' in ini.values() and os.path.exists(fpath_red):
        data_1d = read_geo(fpath_red).GetOutput().GetPointData()
        for f in ini.keys():
            if data_1d.HasArray(f):
                ini_val[f]['1d'] = v2n(data_1d.GetArray(f))

    # apply initial conditions
    for f, i in ini.items():
        if i not in ini_val[f]:
            raise RuntimeError('Unknown ' + f + ' initialization ' + i)

        print('  initial condition ' + f + ': ' + i)
        point_data[f] = ini_val[f][i]


def get_vol(db, geo, ini):
    """
    Generate volume mesh for SimVascular: remove all unused arrays and reoder tet nodes
    """
    f_vol = db.get_volume(geo)
    f_out = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
    f_ini = db.get_initial_conditions(geo)

    if not os.path.exists(f_vol):
        print('  no volume mesh')
        return

    # read volume mesh
    vol = read_geo(f_vol).GetOutput()

    # get geometry
    points = v2n(vol.GetPoints().GetData())
    cells = cell_connectivity(vol)

    # reorder nodes in tets to fix negative Jacobian
    if not jacobian_positive(points, cells['tetra']):
        cells['tetra'] = cells['tetra'][:, [0, 1, 3, 2]]
        print('  tets flipped')
    else:
        print('  tets ok')
    # assert jacobian_positive(points, cells['tetra']), 'Jacobian negative after flipping tets'

    if not jacobian_positive(points, cells['tetra']):
        print('  Jacobian negative after flipping tets')

    # get arrays
    point_data = {'GlobalNodeID': np.expand_dims(v2n(vol.GetPointData().GetArray('GlobalNodeID')), axis=1)}
    cell_data = {'GlobalElementID': np.expand_dims(v2n(vol.GetCellData().GetArray('GlobalElementID')), axis=1)}

    # write raw write to file
    mesh = meshio.Mesh(points, [('tetra', cells['tetra'])], point_data=point_data, cell_data=cell_data)
    meshio.write(f_out, mesh)

    # get initial conditions
    get_initial_conditions(db, geo, point_data, ini)

    # write initial conditions to file
    mesh = meshio.Mesh(points, [('tetra', cells['tetra'])], point_data=point_data, cell_data=cell_data)
    meshio.write(f_ini, mesh)


def get_indices(a, b):
    """
    Elementwise True if an entry in a is in any of b
    """
    i = np.zeros(a.shape, dtype=bool)
    for j in b:
        i |= (a == j)
    return i


def get_surf(db, geo):
    """
    Generate arrays in surface mesh used by SimVascular
    """
    # get all surfaces
    surfaces = glob.glob(os.path.join(db.get_surface_dir(geo), '*.vtp'))

    for f_surf in surfaces:
        # read volume mesh with results
        surf = read_geo(f_surf).GetOutput()
        surf_p = surf.GetPointData()
        surf_c = surf.GetCellData()

        # get output name
        name_osmsc = os.path.basename(f_surf)
        if 'all_exterior' in name_osmsc:
            name = geo + '.vtp'
        elif 'wall' in name_osmsc:
            name = 'walls_combined.vtp'
        else:
            name = os.path.join('caps', name_osmsc)

        f_out = os.path.join(db.get_sv_meshes(geo), name)

        # reconstruct SimVascular arrays from BC_FaceID
        face_id = v2n(surf_c.GetArray('BC_FaceID'))

        # read surface ids
        try:
            caps = get_indices(face_id, db.get_surface_ids(geo, 'caps'))
            inflow = get_indices(face_id, db.get_surface_ids(geo, 'inflow'))
            outlets = get_indices(face_id, db.get_surface_ids(geo, 'outlets'))
        except (KeyError, TypeError):
            print('  face missing in boundary conditions')
            return

        # initialize new arrays
        n_names = ['GlobalBoundaryPoints']
        c_names = ['GlobalBoundaryCells', 'CapID', 'BadTriangle', 'FreeEdge', 'BooleanRegion', 'ModelFaceID',
                   'Normals', 'ActiveCells']
        arrays = {}
        for n in n_names:
            arrays[n] = {'handle': surf_p, 'array': np.zeros(surf.GetNumberOfPoints(), dtype=np.int64)}
        for n in c_names:
            arrays[n] = {'handle': surf_c, 'array': np.zeros(surf.GetNumberOfCells(), dtype=np.int64)}

        # rename
        arrays['ModelFaceID']['array'] = face_id + 1

        # all caps
        arrays['ActiveCells']['array'][caps] = 1

        # inflow is 1, outflow is 2
        arrays['CapID']['array'][inflow] = 1
        arrays['CapID']['array'][outlets] = 2

        # remove old array
        surf_c.RemoveArray('BC_FaceID')

        # add new arrays
        for n, v in arrays.items():
            out_array = n2v(v['array'])
            out_array.SetName(n)
            v['handle'].AddArray(out_array)

        # generate normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surf)
        normals.ComputePointNormalsOff()
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.Update()

        # export to generated folder
        write_geo(f_out, normals.GetOutput())


def get_meshes(db, geo):
    ini = {}
    if 'ini_' in db.study:
        ini_type = db.study.split('_')[1]
        ini['pressure'] = ini_type
        ini['velocity'] = ini_type
    elif 'steady' in db.study:
        ini['pressure'] = 'zero'
        ini['velocity'] = 'zero'
    elif 'irene' == db.study:
        ini['pressure'] = 'steady'
        ini['velocity'] = 'steady'
    else:
        raise ValueError('Unknown intialization for study ' + db.study)

    get_vol(db, geo, ini)
    get_surf(db, geo)


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        get_meshes(db, geo)


if __name__ == '__main__':
    descr = 'Generate all meshes for SimVascular'
    d, g, _ = input_args(descr)
    main(d, g)
