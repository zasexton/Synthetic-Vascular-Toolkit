#!/usr/bin/env python

import os
import meshio
import numpy as np
import pdb

from collections import defaultdict
from vtk.util.numpy_support import vtk_to_numpy as v2n

from .vtk_functions import read_geo, get_all_arrays, cell_connectivity
from .get_database import Database, input_args


def split(array):
    """
    Split array name in name and time step if possible
    """
    comp = array.split('_')
    num = comp[-1]

    # check if array name has a time step
    try:
        time = float(num)
        name = '_'.join([c for c in comp[:-1]])
    except ValueError:
        time = 0
        name = array
    return time, name


def convert_data(data):
    """
    Change array dimensions to comply with meshio
    """
    if len(data.shape) == 1:
        return np.expand_dims(data, axis=1)
    elif len(data.shape) == 2 and data.shape[1] == 4:
        return data[:, 1:]
    else:
        return data


def collect_arrays(array_type, input_arrays, arrays):
    for array_name, data in input_arrays.items():
        time, name = split(array_name)
        arrays[time][array_type][name] = convert_data(data)


def get_time_series(geo):
    # read geometry
    point_arrays, cell_arrays = get_all_arrays(geo)

    # collect all arrays
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()
    collect_arrays('point', point_arrays, arrays)
    collect_arrays('cell', cell_arrays, arrays)
    return arrays


def write_xdmf(geo, arrays, f_out):
    """
    Convert .vtu/.vtp to xdmf
    """
    # extract connectivity
    cells = cell_connectivity(geo)
    points = v2n(geo.GetPoints().GetData())

    # collect time steps
    times = np.unique(list(arrays.keys())).tolist()
    if 0 in times:
        times.remove(0)
    if len(times) == 0:
        times = [0.0]

    with meshio.xdmf.TimeSeriesWriter(f_out) as writer:
        # write points and cells
        writer.write_points_cells(points, cells)

        # write arrays
        for time, data in arrays.items():
            if time == '0':
                for t in times:
                    writer.write_data(t, point_data=data['point'], cell_data=data['cell'])
            else:
                writer.write_data(time, point_data=data['point'], cell_data=data['cell'])


def osmsc_to_xdmf(f_in, f_out):
    # read geometry
    geo = read_geo(f_in).GetOutput()

    # extract time data
    arrays = get_time_series(geo)

    # write to file
    write_xdmf(geo, arrays, f_out)


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        f_in = db.get_3d_flow(geo)
        f_out = os.path.splitext(f_in)[0] + '.xdmf'

        if not os.path.exists(f_in):
            continue

        osmsc_to_xdmf(f_in, f_out)


if __name__ == '__main__':
    descr = 'Extract 3d-results at 1d-locations'
    d, g, _ = input_args(descr)
    main(d, g)
