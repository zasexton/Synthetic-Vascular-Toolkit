#!/usr/bin/env python

import contextlib
import csv
import glob
import io
import os
import pdb
import re
import shutil
import sys
import argparse
import subprocess

import numpy as np
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from simulation_io import get_dict, get_caps_db, collect_results_db_3d_3d
from vtk_functions import read_geo, write_geo, collect_arrays
from compare_1d import plot_1d_3d_caps

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_error_spatial(db, geometries):
    # get post-processing constants
    post = Post()
    fields = post.fields
    fields.remove('area')

    # set global plot options
    fig, ax = plt.subplots(len(fields), len(geometries), figsize=(16, 8), dpi=200, sharey=True, sharex=True)  #
    plt.rcParams['axes.linewidth'] = 2

    # get 1d/3d map
    cycles = []
    for j, geo in enumerate(geometries):
        # get results
        res, time = collect_results_db_3d_3d(db, geo)

        # number of cardiac cycles
        n_cycle = time['3d_rerun_n_cycle']
        cycles += [n_cycle]

        # get caps
        caps = get_caps_db(db, geo)

        for i, f in enumerate(post.fields):
            # plot location
            if len(geometries) == 1:
                pos = i
            else:
                pos = (i, j)

            # calculate errors
            err = []
            err_last = []
            val = []
            for c, br in caps.items():
                # get results for branch at all time steps
                res_br = res[br][f]['3d_rerun_all']

                # get last cycle
                res_last = res[br][f]['3d_rerun_cap']

                # normalize
                if f == 'pressure':
                    norm = np.mean(res_last)
                elif f == 'flow':
                    norm = np.max(res_last) - np.min(res_last)

                # get start and end step of each cardiac cycle
                cycle_range = []
                for k in range(1, n_cycle + 1):
                    i_cycle = np.where(time['3d_rerun_i_cycle_' + str(k)])[0]
                    cycle_range += [i_cycle]

                # calculate cycle error
                err_br = []
                err_br_last = []
                val_br = []
                t0 = []
                for k in range(1, n_cycle):
                    t_prev = cycle_range[k - 1]
                    t_this = cycle_range[k]
                    t_last = cycle_range[-1]
                    diff = np.mean(res_br[t_this] - res_br[t_prev])
                    err_br += [np.abs(diff / norm)]
                    diff = np.mean(res_br[t_last] - res_br[t_prev])
                    err_br_last += [np.abs(diff / norm)]
                    val_br += [res_br[t_this[0]]]
                    t0 += [time['3d_rerun_all'][t_prev[0]]]
                err += [err_br]
                err_last += [err_br_last]
                val += [val_br]
                t0 = np.array(t0)
            err = np.array(err).T
            err_last = np.array(err_last).T
            val = np.array(val)

            # mean error over all outlets
            err_all = np.mean(err, axis=1)

            # how much does the error decrease in each cycle?
            slope = np.mean(np.diff(np.log10(err_all)))

            # error threshold for a converged solution
            thresh = 1.0e-3

            # how many cycles are needed to reach convergence
            i_conv = np.where(np.all(err < thresh, axis=1))[0]
            if not i_conv.any():
                i_conv = n_cycle + int((np.log10(thresh) - np.log10(err_all[-1])) / slope + 1.0)
            else:
                i_conv = i_conv[0] + 2

            # time constants per outlet [s]
            if f == 'pressure':
                dt = np.mean(np.diff(t0))
                dval_dt = np.diff(val) / np.diff(t0)

                # select all time steps where error decreases
                i_decr = np.where(np.all(dval_dt > 0, axis=0) == False)[0]
                if np.any(i_decr):
                    dval_dt = dval_dt[:, :i_decr[0]]
                    t0 = t0[:i_decr[0]]
                else:
                    t0 = t0[:-1]

                tau_all = - 1 / np.mean(np.diff(np.log(dval_dt)) / dt, axis=1)
                val0 = np.zeros(len(caps))
                for cap, tau in enumerate(tau_all):
                    val0[cap] += [np.mean(tau * dval_dt[cap] / np.exp(-t0 / tau))]
                out_str = geo + ' pres time constant ' + '{:2.1e}'.format(np.max(tau_all)) + ' [s]'
                out_str += ' = ' '{:2.1e}'.format(np.max(tau_all) / dt) + ' [cycles]'
                print(out_str)

            # calculate periodic value
            f_conv = norm
            for m in range(10000 - n_cycle):
                f_conv += err_all[-1] * norm * 10 ** ((m + 1) * slope)

            # print errors
            max_err = np.max(err[-1])
            max_outlet = db.get_cap_names(geo)[list(caps.keys())[np.argmax(err[-1])]]
            f_delta = (f_conv - norm) * post.convert[f]
            out_str = geo + ' ' + f[:4]
            out_str += '\tdelta to periodic ' + '{:2.1e}'.format(f_delta) + ' [' + post.units[f] + '] '
            out_str += '\tperiodic error {:.2e}'.format(max_err * 100) + '%'
            out_str += '\tconverged in {:>2}'.format(str(i_conv)) + ' cycles'
            if f == 'pressure':
                out_str += ' ' + '\t{:.2e}'.format(i_conv / np.max(tau_all) * dt) + ' times tau'
            else:
                out_str += '\t\t\t'
            out_str += '\tat outlet ' + max_outlet
            print(out_str)

            # plot data points
            ax[pos].plot(np.arange(2, len(err) + 2), err, 'o-')

            # plot threshold
            ax[pos].plot([0, 999], [thresh, thresh], 'k-')

            # set plot options
            if i == 0:
                ax[pos].set_title(geo)
            if i == len(fields) - 1:
                ax[pos].set_xlabel('Cardiac cycle')
            if j == 0:
                ax[pos].set_ylabel(f.capitalize() + ' cyclic error')
            ax[pos].grid(True)
            ax[pos].ticklabel_format(axis='y')
            ax[pos].set_yscale('log')
            ax[pos].set_ylim([1.0e-5, 1])
            # ax[pos].set_xlim([2, np.max(cycles)])
            ax[pos].set_xlim([2, 50])
            ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))

    plt.subplots_adjust(right=0.8)
    fname = 'outlets.png'
    fpath = os.path.join(db.get_statistics_dir(), fname)
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def main(db, geometries, params):
    geos = []
    for geo in geometries:
        f_path = db.get_3d_flow_rerun(geo)
        if not os.path.exists(f_path):
            continue
        geos += [geo]

    # geos = ['0003_0001', '0067_0001']
    plot_error_spatial(db, geos)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    d, g, p = input_args(descr)
    main(d, g, p)
