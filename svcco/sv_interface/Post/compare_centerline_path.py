#!/usr/bin/env python

import pdb
import numpy as np

from vtk_functions import read_geo, write_geo

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n


def main():
    f_in = '/home/pfaller/work/osmsc/data_generated/centerlines/0075_1001.vtp'
    f_out = '0075_1001.vtp'

    # read geometry from file
    cent = read_geo(f_in).GetOutput()

    # get number of points
    n_point = cent.GetNumberOfPoints()

    # create dummy array
    array = 42 * np.ones(n_point)

    # add array to centerline
    out_array = n2v(array)
    out_array.SetName('Answer to the Ultimate Question of Life, The Universe, and Everything')
    cent.GetPointData().AddArray(out_array)

    # write geometry to file
    write_geo(f_out, cent)

    # set break-point for debugging
    pdb.set_trace()


if __name__ == '__main__':
    main()
