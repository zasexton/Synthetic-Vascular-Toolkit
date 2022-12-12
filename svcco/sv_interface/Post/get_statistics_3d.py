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

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from simulation_io import get_dict
from vtk_functions import read_geo, write_geo, collect_arrays

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_error_spatial(db, geometries):
    # set global plot options
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=200, sharey='all')#, sharex='all'
    plt.rcParams['axes.linewidth'] = 2

    # read errors
    f_path = db.get_3d_3d_comparison()
    if not os.path.exists(f_path):
        return
    err = get_dict(f_path)

    # select geometries
    geos = [geo for geo in geometries if geo in err]
    legend = [geo[:4] for geo in geometries if geo in err]

    for i, f in enumerate(['pressure', 'velocity']):
        for j, c in enumerate(['avg']):#, 'max'
            # plot location
            pos = i#(i, j)

            # plot data points
            lengths = []
            for geo in geos:
                e = err[geo][f][c]
                x = np.arange(1, len(e) + 1)
                lengths += [len(e)]
                ax[pos].plot(x, e, 'o-')

            # set plot options
            ax[pos].grid(True)
            ax[pos].set_xlabel('Cardiac cycle')
            if f == 'pressure':
                err_name = r'$\epsilon_{p,avg}$'
            elif f == 'velocity':
                err_name = r'$\epsilon_{u,avg}$'
            ax[pos].set_ylabel(f.capitalize() + ' relative ' + c + '. error ' + err_name)
            ax[pos].ticklabel_format(axis='y') #
            # r"$\epsilon_{p,\text{avg}$"
            # ax[pos].legend(legend)
            ax[pos].set_yscale('log')
            ax[pos].set_xticks(np.arange(1, np.max(lengths) + 1))
            ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    lgd = ax[(0)].legend(legend, bbox_to_anchor=(-0.2, 0), loc='right') #
    # fig.subplots_adjust(left=0.2)
    # plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    plt.subplots_adjust(right=0.8)
    fname = '3d_3d_comparison.png'
    fpath = os.path.join(db.get_statistics_dir(), fname)
    fig.savefig(fpath, bbox_extra_artists=(lgd,), bbox_inches='tight')#
    plt.close(fig)


def main(db, geometries, params):
    plot_error_spatial(db, geometries)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    d, g, p = input_args(descr)
    main(d, g, p)
