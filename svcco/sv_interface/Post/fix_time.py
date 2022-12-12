#!/usr/bin/env python

import pdb
import numpy as np
import os

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from common import get_dict
from get_database import Database, input_args
from vtk_functions import read_geo, write_geo


def main(db, geometries):
    steps_new = {'0074_0001': 600,
                 '0075_1001': 1600,
                 '0090_0001': 1200,
                 '0063_0001': 5750,
                 '0064_0001': 5750,
                 '0065_0001': 5750,
                 '0075_0001': 5750,
                 '0076_0001': 5750,
                 '0068_0001': 2000,
                 '0099_0001': 1000,
                 '0092_0001':  850}

    # get bcs of all models
    bc_all = get_dict(db.db_params)

    for geo in steps_new.keys():
        # 3d exported time
        time, _ = db.get_inflow(geo)
        if time is None:
            continue

        # number of steps
        numstep = db.get_3d_numstep(geo)

        # output increment
        nt_out = db.get_3d_increment(geo)

        if not numstep/nt_out % 1 == 0.0:
            print(geo + ' output step not integer: ' + str(numstep/nt_out), nt_out, numstep, steps_new[geo])
            bc_all[geo]['params']['sim_steps_per_cycle'] = str(steps_new[geo])
    np.save(db.db_params, bc_all)


if __name__ == '__main__':
    descr = 'Fix wrong time step / numstep'
    d, g, _ = input_args(descr)
    main(d, g)
