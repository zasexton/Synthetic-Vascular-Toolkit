#!/usr/bin/env python

import os
import re
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database
from vtk_functions import read_geo, write_geo, get_all_arrays
from simulation_io import get_dict


fsize = 20
plt.rcParams.update({'font.size': fsize})

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})

def plot_area():
    db = Database()
    f_in = db.get_centerline_path('0003_0001')
    f_out = os.path.join('.', 'OSMSC_0003_0001_branch0')

    # get centerline
    geo = read_geo(f_in).GetOutput()
    arrays, _ = get_all_arrays(geo)

    # extract branch
    br = 0
    mask = arrays['BranchId'] == br

    # get plot quantities
    path = arrays['Path'][mask]
    area_slice = arrays['CenterlineSectionArea'][mask]
    area_vmtk = arrays['MaximumInscribedSphereRadius'][mask] ** 2 * np.pi

    print('factor', area_slice[0] / area_vmtk[0])

    # make plot
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    ax.plot(path, area_slice, 'r-')
    ax.plot(path, area_vmtk, 'b-')
    ax.legend(['Area from slicing', 'Area from MISR'])
    ax.set_xlim(left=0)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['Inlet', '2', '4'])
    plt.xlabel('Branch path [cm]')
    plt.ylabel('Area [cm$^2$]')
    plt.grid()
    fig.savefig(f_out, bbox_inches='tight')


def plot_model_statistics(db, geometries_paper):

    pie = defaultdict(lambda: defaultdict(int))
    cats = ['deliverable_category', 'vascular_state', 'treatment', 'image_data_modality', 'paper_reference', 'gender']

    names = {'deliverable_category': 'Vascular anatomy',
             'vascular_state': 'Vascular state',
             'treatment': 'Treatment',
             'image_data_modality': 'Imaging',
             'paper_reference': 'Literature reference',
             'gender': 'Gender'}

    # count all categories
    for geo in geometries_paper:
        bc_def = db.get_bcs(geo)
        if bc_def is not None:
            pie['has_bc']['yes'] += 1
            params = bc_def['params']
            for cat in cats:
                if cat in params and 'unpublished' not in params[cat]:
                    if cat == 'paper_reference':
                        name = params[cat][:-2]
                    else:
                        name = params[cat].capitalize()
                elif cat in params and params[cat] == 'Unclassified':
                    name = 'Normal'
                else:
                    if cat == 'paper_reference':
                        name = 'Unpublished'
                    else:
                        name = 'None'
                pie[cat][name] += 1
        else:
            pie['has_bc']['no'] += 1

    # make plots
    fig, axs = plt.subplots(2, 2, dpi=300, figsize=(25, 15))

    selection = ['deliverable_category', 'vascular_state', 'treatment', 'paper_reference']
    for cat, ax in zip(selection, axs.ravel()):
        labels = np.array([re.sub(r'\([^)]*\)', '', c) for c in pie[cat].keys()])
        sizes = np.array(list(pie[cat].values()))
        order = np.argsort(sizes)

        print('num', np.sum(sizes))
        abs_size = lambda p: '{:.0f}'.format(p * np.sum(sizes) / 100)

        theme = plt.get_cmap('Reds')
        ax.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])
        ax.pie(sizes[order], labels=labels[order], autopct=abs_size)
        ax.axis('equal')
        ax.set_title(names[cat], fontsize=40, pad=20)

    f_out = os.path.join(db.get_statistics_dir(), 'repo_statistics')#.pgf
    fig.savefig(f_out, bbox_inches='tight')
    plt.close(fig)


def plot_collage(db, geos):
    nx = 10
    ny = 8
    assert nx * ny >= len(geos), 'choose larger image grid: ' + str(len(geos))
    fig , ax = plt.subplots(nx, ny, figsize=(ny * 2, nx * 2.5), dpi=100)
    ig = 0
    for i in range(nx):
        for j in range(ny):
            ax[i, j].axis('off')
            if ig >= len(geos):
                continue
            geo = geos[ig]
            impath = db.get_png(geo)
            if not os.path.exists(impath):
                continue
            im = plt.imread(impath)
            ax[i, j].imshow(im)
            ax[i, j].set_title(geo.replace('_', '\_'), fontsize=18)
            ig += 1
    f_out = os.path.join(db.get_statistics_dir(), 'repo_models')#.pgf
    #fig.tight_layout(pad=3.0)
    fig.savefig(f_out, bbox_inches='tight')
    plt.close(fig)

def main():
    db = Database('1spb_length_stenosis')
    # geometries_paper = db.get_geometries_select('paper')

    # only geometries where a 0d AND a 3d_rerun solution exisis
    geometries_paper = []
    for geo in sorted(list(get_dict(db.get_log_file_0d()).keys())):
        if os.path.exists('/home/pfaller/work/osmsc/studies/ini_1d_quad/3d_flow/' + geo + '.vtp'):
            geometries_paper += [geo]
        else:
            if os.path.exists('/home/pfaller/work/osmsc/studies/ini_zero/3d_flow/' + geo + '.vtp'):
                geometries_paper += [geo]

    print(geometries_paper)
    plot_collage(db, geometries_paper)
    plot_model_statistics(db, geometries_paper)
    plot_area()


if __name__ == '__main__':
    descr = 'Make plots for 3D-1D-0D paper'
    main()
