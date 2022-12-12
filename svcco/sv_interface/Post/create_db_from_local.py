#!/usr/bin/env python

import os
import pdb
import csv
from collections import defaultdict, OrderedDict
import numpy as np

from get_database import Database


def get_params():
    # paths where results are exported
    db_dir = 'database'
    par_file = 'parameters'

    db = Database()
    geometries = db.get_geometries()

    bc_def = defaultdict(OrderedDict)

    # extract simulation parameters
    for geo in geometries:
        print('Extracting geometry ' + geo)
        bc_def[geo] = db.get_bcs_local(geo)

    np.save(os.path.join(db_dir, par_file), bc_def)


def main():
    get_params()


if __name__ == '__main__':
    main()
