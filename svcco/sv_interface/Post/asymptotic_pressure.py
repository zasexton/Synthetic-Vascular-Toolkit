#!/usr/bin/env python

import os
import sys
import re
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections import defaultdict
from scipy.interpolate import interp1d

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('..')
from get_database import Database, Post, input_args
from vtk_functions import read_geo, write_geo, get_all_arrays
from simulation_io import get_dict, get_caps_db, collect_results_db_3d_3d, collect_results_db_1d_3d, collect_results_db_0d
from get_sv_project import coronary_sv_to_oned
from get_statistics_bc import collect_errors

fsize = 12
plt.rcParams.update({'font.size': fsize})
# plt.style.use('dark_background')


def plot_correlation(db):
    post = Post()

    # get numerical time constants
    res_num = get_dict(db.get_convergence_path())

    # only plot subset of geometries
    geometries_paper = db.get_geometries_select('paper')

    # convergence criterion
    tol = 0.01

    fig1, ax1 = plt.subplots(1, 2, dpi=300, figsize=(12, 4), sharex='row')

    for pos, f in enumerate(['pressure', 'flow']):

        i_conv_all = []
        tau_all = []
        for geo in res_num:
            if geo not in geometries_paper:
                continue
            params = db.get_bcs(geo)

            # skip non-RCR models
            types = np.unique(list(params['bc_type'].values()))
            if not len(types) == 1 or not types[0] == 'rcr':
                continue

            i_conv = res_num[geo]['i_conv'][f]
            tau = np.mean(res_num[geo]['tau'][f])
            col = post.colors[params['params']['deliverable_category']]
            ax1[pos].plot(tau, i_conv, marker='o', color=col)
            i_conv_all += [i_conv]
            tau_all += [tau]

        # fit linear equation to data
        if db.study == '1spb_length' and f == 'pressure':
            # coef = np.polyfit(tau_all, i_conv_all, 1)
            # poly = np.poly1d(coef)
            # ax1.plot(tau_all, poly(tau_all), 'k-')

            t = np.linspace(0, 10, num=10000)
            fx = - np.log(tol) * t
            ax1[pos].plot(t, fx.astype(int) + 1, 'k-')

        # plt.yscale('log')
        ax1[pos].set_title(f.capitalize())
        ax1[pos].xaxis.grid('minor')
        ax1[pos].yaxis.grid(True)
        ax1[pos].set_xlim(0, 10)
        ax1[pos].set_ylim(0, 50)
        ax1[pos].set_xlabel(r'Model time constant $\bar{\tau}/T$ [-]')
        if pos == 0:
            ax1[pos].set_ylabel(r'Number of cardiac cycles [-]')

    fname = os.path.join(db.get_statistics_dir(), 'correlation.png')
    fig1.savefig(fname, bbox_inches='tight')
    plt.close(fig1)


def plot_convergence(db):
    post = Post()

    # get numerical time constants
    res_num = get_dict(db.get_convergence_path())

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))

    markers = {'flow': 'o', 'pressure': 'x'}

    geos = []
    i = 0
    for geo in res_num:
        params = db.get_bcs(geo)
        i_conv = res_num[geo]['i_conv']
        col = post.colors[params['params']['deliverable_category']]
        for f, m in markers.items():
            ax1.plot(i, i_conv[f], marker=m, color=col)

        geos += [geo]
        i += 1

    # plt.yscale('log')
    ax1.xaxis.grid('minor')
    ax1.yaxis.grid(True)
    # ax1.set_ylim(0, 13)
    plt.xticks(np.arange(len(geos)), geos, rotation='vertical')
    plt.ylabel('Number of cardiac cycles [-]')
    fname = os.path.join(db.get_statistics_dir(), 'convergence.png')
    fig1.savefig(fname, bbox_inches='tight')


def make_err_plot(db, geo, ax, pos, m, f, p, res_m, errors, time, title_study=''):
    post = Post()
    t = 0

    # study names
    studies = {'ini_zero': 'Zero',
               'ini_steady': 'Steady',
               'ini_irene': 'Steady (start from mean)',
               'ini_1d_quad': '1D'}

    # get caps
    caps = get_caps_db(db, geo)
    del caps['inflow']

    # get boundary conditions
    bcs = db.get_bcs(geo)['bc']
    bct = db.get_bcs(geo)['bc_type']

    # get time
    n_cycle = time[m + '_n_cycle']

    # plot times
    times_plot = []
    times_all = []
    for k in range(1, n_cycle + 1):
        times_all += [np.where(time[m + '_i_cycle_' + str(k)])[0]]
        times_plot += [np.where(time[m + '_i_cycle_' + str(k)])[0][t]]
    times_all = np.array(times_all)

    # add last time step
    if t == 0:
        times_plot += [np.where(time[m + '_i_cycle_' + str(n_cycle)])[0][-1]]

    cycles = np.arange(len(times_plot))
    if t != 0:
        cycles += 1

    # error threshold for a converged solution
    thresh = 1.0e-2
    e_thresh = 'asymptotic'

    # get numerical time constant
    res_num = get_dict(db.get_convergence_path())
    if geo in res_num:
        tau = np.mean(res_num[geo]['tau']['pressure'])
        alpha = 1 / (np.exp(1 / tau) - 1)
        thresh_cyclic = thresh / alpha
    else:
        thresh_cyclic = np.nan
    # thresh_cyclic = 0
    # pdb.set_trace()
    #
    # collect results
    res_m_all = []
    res_m_t = []
    res_m_m = []
    res_0d_t = []
    res_qm_t = []
    for c, br in caps.items():
        res_m_all += [res_m[br][f][m + '_all']]
        res_m_t += [res_m[br][f][m + '_all'][times_plot]]
        # res_0d_t += [interp1d(res_0d[br]['t'], res_0d[br]['p'], fill_value='extrapolate')(time[m][t]).tolist()]
        res_0d_t += [res_m_t[-1][-1]]
        res_m_m += [np.mean(res_m[br][f][m + '_all'][times_all], axis=1)]

        if bct[c] == 'resistance':
            resistance = bcs[c]['R']
        elif bct[c] == 'rcr':
            resistance = bcs[c]['Rd'] + bcs[c]['Rp']
        elif bct[c] == 'coronary':
            cor = coronary_sv_to_oned(bcs[c])
            resistance = cor['Ra1'] + cor['Ra2'] + cor['Rv1']
        res_qm_t += [resistance * np.mean(res_m[br]['flow'][m + '_cap'])]

    # make plot
    c_max = 20
    x_min = 1
    style = 'x-'
    if p == 'cycle':
        title = 'Solution'
        x_min = 0
        style = '-'
        x = time[m + '_all'] / time[m][-1]
        y = np.array(res_m_all).T * post.convert[f]
        xticks = [0]
        ylabel = f.capitalize() + ' [' + post.units[f] + ']'
    elif p == 'cycle_norm':
        title = 'Normalized solution'
        x_min = 0
        xticks = [0]
        style = '-'
        x = time[m + '_all'] / time[m][-1]
        y = np.array(res_m_all).T / np.array(res_m_m)[:, -1]
        ylabel = f.capitalize() + ' [-]'
    elif p == 'initial':
        title = 'Initial values'
        x = cycles
        y = np.array(res_m_t).T * post.convert[f]
        xticks = [1]
        ylabel = 'Initial ' + f + ' [' + post.units[f] + ']'
    elif p == 'mean':
        title = 'Mean cycle solution'
        x = cycles[1:]
        y = np.array(res_m_m).T * post.convert[f]
        y /= y[-1]
        xticks = [1]
        # ylabel = 'Mean ' + f + ' [' + post.units[f] + ']'
        ylabel = 'Mean ' + f + ' [-]'
    elif p in 'cyclic':
        title = 'Cyclic error $\epsilon_n$'
        x = cycles[1:-1] + 1
        y = errors['cyclic'][f]
        xticks = [2]
        ylabel = 'Cyclic ' + f + ' error [-]'
    elif p in 'asymptotic':
        title = 'Asymptotic error $\epsilon_\infty$'
        x = cycles[1:-1]
        y = errors['asymptotic'][f][:-1]
        xticks = [1]
        ylabel = 'Asymptotic ' + f + ' error [-]'
    else:
        title = ''
        x = np.nan
        y = np.nan
        xticks = []
    xticks += [c_max]

    # converged time step
    conv = np.where(np.all(errors[e_thresh][f] < thresh, axis=1))[0]
    if not conv.any():
        i_conv = -1
    else:
        i_conv = np.min(conv)
    if e_thresh == 'cyclic':
        i_conv += 2
    elif e_thresh == 'asymptotic':
        i_conv += 1
        if p in 'asymptotic':
            print(f[0] + ' ' + str(i_conv))
    xticks += [i_conv]

    # plot
    if p == 'initial' or p == 'mean':
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        # ax[pos].plot([cycles[0], cycles[-1]], np.vstack((y[-1], y[-1])), '--')
        ax[pos].plot([0, 999], [1, 1], 'k-')
        ax[pos].set_ylim([0, 1.2])
        # ax.plot([cycles[0], cycles[-1]], np.vstack((res_qm_t, res_qm_t)) * post.convert[f], '--')

    ax[pos].plot(x, y, style)
    ax[pos].axvline(x=i_conv, color='k')
    if not title_study or pos[1] == 0:
        ax[pos].set_ylabel(ylabel)
    x_eps = 0.5 * np.max(xticks) / 20
    ax[pos].set_xlim([x_min - x_eps, np.max(xticks) + x_eps])
    # ax[pos].set_xlim([0, np.max(x)])
    ax[pos].set_xticks(xticks)
    ax[pos].grid('both')

    if p in ['cyclic', 'asymptotic']:
        ax[pos].set_yscale('log')
        if f == 'pressure':
            ax[pos].set_ylim([1e-4, 1])
        elif f == 'flow':
            ax[pos].set_ylim([1e-5, 1e-1])
        # ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 2))
    if p == 'cyclic' and e_thresh == 'asymptotic':
        ax[pos].plot([0, 999], [thresh_cyclic, thresh_cyclic], 'k-')
    if p == e_thresh:
        ax[pos].plot([0, 999], [thresh, thresh], 'k-')
    if pos[0] == 1:
        ax[pos].set_xlabel('Cardiac cycle [-]')
    if pos[0] == 0:
        if title_study:
            ax[pos].set_title(studies[title_study])
        else:
            ax[pos].set_title(title)
    ax[pos].set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    return y, i_conv


def get_time_constants(db, geo, c_res):
    const = defaultdict(dict)

    # get cap names
    tau_ana_dic = db.get_time_constants(geo)
    caps = get_caps_db(db, geo)
    del caps['inflow']
    n_out = len(caps)

    # time constants (analytical)
    const['tau']['ana'] = np.array([tau_ana_dic[c] for c in caps])

    # factor between asymptotic and cyclic error (analytical)
    const['alpha']['ana'] = 1 / (np.exp(1 / const['tau']['ana']) - 1)

    const['tau']['num'] = {}
    const['alpha']['num'] = {}

    # tolerance for the linear slope in a log-plot
    tol = {'pressure': 1e-10,
           'flow': 1e-6}

    for f in c_res.keys():
        # # chose range of cardiac cycles to evaluate time constants (within given tolerance)
        i_min = -1
        i_good = np.sum(np.abs(np.diff(-np.diff(np.log(c_res[f]['cyclic']), axis=0), axis=0)) < tol[f], axis=1) == n_out
        for i, g in enumerate(i_good):
            if i_min == -1 and g:
                i_min = i
                continue
            if i_min > -1 and not g:
                i_max = i - 1
                break
        else:
            i_min = 0
            i_max = np.min([10, c_res[f]['cyclic'].shape[0] - 1])
        i_calc = np.arange(i_min, i_max)

        # time constants (numerical)
        const['tau']['num'][f] = 1 / np.mean(-np.diff(np.log(c_res[f]['cyclic'][i_calc]), axis=0), axis=0)

        # factor between asymptotic and cyclic error (numerical)
        const['alpha']['num'][f] = np.mean(c_res[f]['asymptotic'][i_calc + 1] / c_res[f]['cyclic'][i_calc], axis=0)

    return const


def plot_pressure(study, geo):
    # plot settings
    m = '3d_rerun'
    # m = '1d'
    # m = '0d'
    fields = ['pressure', 'flow']
    # fields = ['pressure']
    # comparisons = ['cycle', 'initial', 'cyclic', 'asymptotic']
    # comparisons = ['cycle', 'mean', 'asymptotic']
    # comparisons = ['cycle', 'mean', 'cyclic', 'asymptotic']
    comparisons = ['cycle', 'mean', 'cyclic', 'asymptotic']
    # comparisons = ['cycle_norm', 'mean', 'cyclic', 'asymptotic']

    # get database
    db = Database(study)

    # res_m, time = collect_results_db_1d_3d(db, geo)
    if m == '0d':
        res_m, time = collect_results_db_0d(db, geo)
    elif m == '3d_rerun':
        res_m, time = collect_results_db_3d_3d(db, geo)
    else:
        return

    if res_m is None:
        return

    if os.path.exists(db.get_bc_0D_path(geo, m)):
        res_0d = np.load(db.get_bc_0D_path(geo, m), allow_pickle=True).item()
    else:
        return

    print(geo)
    errors = collect_errors(res_m, res_0d, time, m)

    fig, ax = plt.subplots(len(fields), len(comparisons), figsize=(5 * len(comparisons), 5 * len(fields)), dpi=300)

    c_res = defaultdict(dict)
    i_conv = {}
    for j, f in enumerate(fields):
        # for i, p in zip([0, 1, 2, 2], comparisons):
        for i, p in enumerate(comparisons):
            if len(fields) == 1:
                pos = i
            else:
                pos = (j, i)

            c_res[f][p], i_conv[f] = make_err_plot(db, geo, ax, pos, m, f, p, res_m, errors, time)

    f_out = db.get_statistics_dir()
    fname = 'convergence_' + study + '_' + geo
    if len(fields) == 1:
        fname += '_' + f
    fpath = os.path.join(f_out, fname + '.png')
    # plt.subplots_adjust(right=0.8)
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)

    # plot time constants
    const = get_time_constants(db, geo, c_res)
    for f in fields:
        for c, v in const.items():
            fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
            ax.plot(const[c]['ana'], 'o')
            ax.plot(const[c]['num'][f], 'x')
            ax.grid('both')
            ax.legend(['analytical', 'numerical'])
            ax.set_xlabel('Outlet')
            ax.set_ylabel(c)
            os.makedirs(os.path.join(f_out, c), exist_ok=True)
            fpath = os.path.join(f_out, c, c + '_' + study + '_' + geo + '_' + f + '.png')
            fig.savefig(fpath, bbox_inches='tight')
            plt.close(fig)

    return {'tau': const['tau']['num'], 'i_conv': i_conv}


def plot_pressure_studies(geo):
    # plot settings
    m = '3d_rerun'
    fields = ['pressure', 'flow']
    # studies = ['ini_zero', 'ini_steady', 'ini_1d_quad']
    # studies = ['ini_zero', 'ini_steady', 'ini_irene', 'ini_1d_quad']
    studies = ['ini_zero', 'ini_irene', 'ini_1d_quad']
    # studies = ['ini_1d_quad', 'ini_asymp_pres_1d_velo', 'ini_1d_pres_asymp_velo']
    comparison = 'asymptotic'
    # comparison = 'mean'

    print(geo)
    fig, ax = plt.subplots(len(fields), len(studies), figsize=(6 * len(studies), 5 * len(fields)), dpi=300, sharey='row')

    for j, f in enumerate(fields):
        for i, p in enumerate(studies):
            # get database
            db = Database(p)

            if os.path.exists(db.get_bc_0D_path(geo, m)):
                res_0d = np.load(db.get_bc_0D_path(geo, m), allow_pickle=True).item()
            else:
                return

            res_m, time = collect_results_db_3d_3d(db, geo)
            errors = collect_errors(res_m, res_0d, time, m)

            if len(fields) == 1:
                pos = i
            else:
                pos = (j, i)

            make_err_plot(db, geo, ax, pos, m, f, comparison, res_m, errors, time, title_study=p)

    f_out = '/home/pfaller/work/paper/asymptotic'
    fname = 'comparison_' + geo
    if len(fields) == 1:
        fname += '_' + f
    fpath = os.path.join(f_out, fname + '.png')
    # plt.subplots_adjust(right=0.8)
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def main(db, geo):
    for g in geo:
        # plot_pressure_studies(g)
        res = plot_pressure(db.study, g)
        if res is not None:
            db.add_convergence(g, res)


if __name__ == '__main__':
    descr = 'Make plots for 3D-1D-0D paper'
    d, g, _ = input_args(descr)
    # main(d, g)
    # plot_convergence(d)
    plot_correlation(d)
