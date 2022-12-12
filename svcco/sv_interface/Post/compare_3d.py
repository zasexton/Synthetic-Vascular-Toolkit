#!/usr/bin/env python

import numpy as np
import sys
import argparse
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from collections import OrderedDict, defaultdict
from scipy.interpolate import interp1d

from common import rec_dict
from get_database import Database, Post, input_args
from simulation_io import get_caps_db, collect_results_db_1d_3d, collect_results_db_3d_3d, \
    collect_results_db_3d_3d_spatial


def calc_error_spatial(res, time):
    # interpolate 1d to 3d in space and time (allow extrapolation due to round-off errors at bounds)
    interp = lambda x_1d, y_1d, x_3d: interp1d(x_1d, y_1d.T, fill_value='extrapolate')(x_3d)

    fields = ['pressure', 'velocity']

    err = defaultdict(dict)
    for f in fields:
        err[f]['avg'] = []
        err[f]['max'] = []
        for i in np.arange(1, time['3d_rerun_n_cycle'] + 1):
            res_3d_osmsc = res['3d'][f]
            res_3d_rerun_time = res['3d_rerun'][f][time['3d_rerun_i_cycle_' + str(i)]]
            res_3d_rerun = interp(time['3d_rerun_cycle_' + str(i)], res_3d_rerun_time, time['3d'][1:]).T

            # calculate spatial error
            if f == 'velocity':
                delta = np.linalg.norm(res_3d_osmsc - res_3d_rerun, axis=-1)
                norm = np.mean(np.linalg.norm(res_3d_osmsc, axis=-1))
            elif f == 'pressure':
                delta = res_3d_osmsc - res_3d_rerun
                norm = np.mean(res_3d_osmsc)
            else:
                raise ValueError('Unknown field ' + f)

            diff = delta ** 2
            err[f]['avg'] += [np.sqrt(np.mean(diff)) / norm]
            err[f]['max'] += [np.sqrt(np.max(diff)) / norm]

            # calculate periodicity error
            err[f]['cyc'] += [np.abs(res_3d_rerun[-1] - res_3d_rerun[0]) / (np.max(res_3d_rerun) - np.min(res_3d_rerun))]

    for f in fields:
        for e in err[f]:
            print(f, e, err[f][e])

    # todo: write spatial error to geometry
    return err


def main(db, geometries):
    for geo in geometries:
        print('Comparing geometry ' + geo)

        # read results
        res, time = collect_results_db_3d_3d_spatial(db, geo)
        if res is None:
            continue

        err = calc_error_spatial(res, time)
        db.add_3d_3d_comparison(geo, err)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
