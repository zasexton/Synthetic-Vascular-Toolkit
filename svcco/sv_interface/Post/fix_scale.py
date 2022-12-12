#!/usr/bin/env python

import pdb
import numpy as np
import os
import shutil

from get_database import Database, input_args
from vtk_functions import read_geo, write_geo


def main(db, geometries):
    # scale for geometry
    scales = {'0164_0001': 10}

    for geo in geometries:
        if geo not in scales:
            continue
        print('Fixing geometry ' + geo)

        # get points from geometry
        f_surf = db.get_surfaces(geo, 'all_exterior')
        surf = read_geo(f_surf).GetOutput()
        points = surf.GetPoints()

        # scale all points
        for i in range(points.GetNumberOfPoints()):
            point = np.array(points.GetPoint(i)) * scales[geo]
            points.SetPoint(i, tuple(point))

        # backup original file
        f_path, ext = os.path.splitext(f_surf)
        shutil.copy(f_surf, f_path + '_original' + ext)

        # save new file
        write_geo(f_surf, surf)


if __name__ == '__main__':
    descr = 'Fix wrong scale'
    d, g, _ = input_args(descr)
    main(d, g)
