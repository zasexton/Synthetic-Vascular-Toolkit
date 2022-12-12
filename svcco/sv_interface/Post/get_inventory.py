#!/usr/bin/env python

import pdb
import os
import csv
import numpy as np
import json
from collections import defaultdict, OrderedDict

from get_database import Database, Post
from get_sv_project import coronary_sv_to_oned
from common import get_dict
from get_bcs import get_bcs, get_params

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_json():
    db = Database()
    geometries = db.get_geometries()
    # geometries = ['0075_1001']
    for geo in geometries:
        print(geo)

        # get all meta-data
        par = db.get_bcs(geo)
        if par is None:
            continue
        par_tcl = db.get_bcs_local(geo)

        # add parameters from tcl that are not in database
        for k, v in par_tcl['params'].items():
            if k not in par['params']:
                # exclude local tcl stuff
                if k not in ['auto_path', 'tk_library', 'tcl_library', 'errorCode', 'tcl_pkgPath']:
                    par['params'][k] = v

        # add inflow
        par['bc']['inflow']['flow'] = db.get_inflow_smooth(geo)

        with open(db.get_json(geo), 'w') as file:
            json.dump(par, file, indent=4, sort_keys=True, cls=NumpyEncoder)


def main():
    db = Database()
    geometries = db.get_geometries()
    geometries_paper = db.get_geometries_select('paper')

    properties = defaultdict(OrderedDict)
    params_read = ['age', 'gender', 'deliverable_category', 'vascular_state', 'treatment', 'other_disease',
                   'sim_physio_state', 'image_data_modality', 'paper_reference',
                   'model_source', 'simulation_source', 'sim_steps_per_cycle']

    # extract simulation parameters
    for geo in geometries:
        print('Running geometry ' + geo)
        bc_def = db.get_bcs(geo)

        for pm in params_read:
            if bc_def['params'] is None or pm not in bc_def['params']:
                val = 'unknown'
            else:
                val = bc_def['params'][pm]
            properties[geo][pm] = val

    # check if simulation is to be published
    for geo in geometries:
        if geo in geometries_paper:
            publish = 'yes'
        else:
            publish = 'no'
        properties[geo]['publish'] = publish

    # get inflow type
    for geo in geometries:
        bc_def = db.get_bcs(geo)
        if bc_def is None:
            inflow = 'none'
        else:
            inflow = bc_def['bc']['inflow']['type']
        properties[geo]['inflow_type'] = inflow

    with open('osmsc2.csv', 'w', newline='') as csvfile:
        reader = csv.writer(csvfile, delimiter=',')

        # write header
        reader.writerow(['model_id'] + list(properties[geometries[0]].keys()))

        # write rows
        for geo in geometries:
            reader.writerow([geo] + [v for v in properties[geo].values()])


def time_constant():
    db = Database('1spb_length')
    post = Post()
    geometries = db.get_geometries()
    # geometries = db.get_geometries_select('paper')

    # get numerical time constants
    res_num = get_dict(db.get_convergence_path())

    fig1, ax1 = plt.subplots(dpi=400, figsize=(10, 3))

    geos = []
    i = 0
    taus = {}
    alphas = {}
    for geo in geometries:
        params = db.get_bcs(geo)

        time, _ = db.get_inflow(geo)

        # collect all time constants
        const = db.get_time_constants(geo)

        # skip geometries without RCR
        if const is None:
            continue

        tau_bc = list(const.values())
        while 0.0 in tau_bc:
            tau_bc.remove(0.0)

        if len(tau_bc) == 0:
            continue

        # skip non-RCR models
        # types = np.unique(list(params['bc_type'].values()))
        # if not len(types) == 1 or not types[0] == 'rcr':
        #     continue

        col = post.colors[params['params']['deliverable_category']]
        if geo in res_num:
            tau_num = res_num[geo]['tau']
            for j in range(len(tau_num['flow'])):
                # ax1.plot(i, tau_num['flow'][j], marker='o', color=col)
                ax1.plot(i, tau_num['pressure'][j], marker='x', color=col)
            taus[geo] = np.mean(tau_num['pressure'])
            alphas[geo] = 1 / (np.exp(1 / taus[geo]) - 1)

            print(geo + '\ttau_num = ' + '{:2.1f}'.format(taus[geo])+ '\talpha_num = ' + '{:2.1f}'.format(alphas[geo]))
        # skip models without 0d solution
        else:
            continue

        # ax1.boxplot(tau_bc, positions=[i])
        ax1.plot([i, i], [np.min(tau_bc), np.max(tau_bc)], '-', color=col)
        ax1.plot(i, np.min(tau_bc), '_', color=col)
        ax1.plot(i, np.max(tau_bc), '_', color=col)

        geos += [geo]
        i += 1

    # plt.yscale('log')
    ax1.xaxis.grid('minor')
    ax1.yaxis.grid(True)
    ax1.set_ylim(0, 10)
    ax1.set_xlim(-1, len(geos))
    plt.xticks(np.arange(len(geos)), geos, rotation='vertical')
    plt.ylabel(r'Time constants [-]')
    # fname = os.path.join(db.fpath_gen, 'time_constants_paper_rcr_with_0d.png')
    # fname = os.path.join(db.fpath_gen, 'time_constants_paper')
    fname = os.path.join(db.fpath_gen, 'time_constants')
    fig1.savefig(fname, bbox_inches='tight')
    pdb.set_trace()

if __name__ == '__main__':
    # main()
    # time_constant()
    get_json()
