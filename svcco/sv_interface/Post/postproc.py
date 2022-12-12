#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt

from get_database import input_args, Database, Post
from get_mean_flow_3d import extract_results
from simulation_io import get_dict, get_caps_db, collect_results_db_3d_3d


def transfer_array(node_trg, node_src, name):
    """
    Transfer point data from volume mesh to surface mesh using GlobalNodeID
    Args:
        node_trg: target mesh
        node_src: source mesh
        name: array name to be transfered
    """
    # get global node ids in both meshes
    nd_id_trg = v2n(node_trg.GetPointData().GetArray('GlobalNodeID')).astype(int)
    nd_id_src = v2n(node_src.GetPointData().GetArray('GlobalNodeID')).astype(int)

    # map source mesh to target mesh
    mask = map_meshes(nd_id_src, nd_id_trg)

    # transfer array from volume mesh to surface mesh
    assert node_src.GetPointData().HasArray(name), 'Source mesh has no array ' + name
    res = v2n(node_src.GetPointData().GetArray(name))

    # create array in target mesh
    out_array = n2v(res[mask])
    out_array.SetName(name)
    node_trg.GetPointData().AddArray(out_array)


def plot_bc(db, geo):
    # get post-processing constants
    post = Post()

    # get results
    res, time = collect_results_db_3d_3d(db, geo)

    # get caps
    caps = get_caps_db(db, geo)

    for i, f in enumerate(['flow', 'pressure']):
        names = []
        values = []
        for c, br in caps.items():
            values += [res[br][f]['3d_rerun_all'].tolist()]
            names += [c]

        fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
        ax.plot(time['3d_rerun_all'], np.array(values).T * post.convert[f])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
        ax.legend(names)
        ax.grid(True)

        fpath = os.path.join(db.fpath_study, 'simulation', geo + '_' + f + '.png')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close(fig)

def main(db, geometries):
    """
    Loop all geometries in database
    """
    for geo in geometries:
        fpath_1d = db.get_centerline_path(geo)
        fpath_vol = db.get_res_3d_vol_rerun(geo)
        fpath_out = db.get_3d_flow_rerun(geo)
        if not os.path.exists(fpath_vol):
            continue

        print('Processing ' + geo)
        extract_results(fpath_1d, fpath_vol, fpath_out, only_caps=True)
        plot_bc(db, geo)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
