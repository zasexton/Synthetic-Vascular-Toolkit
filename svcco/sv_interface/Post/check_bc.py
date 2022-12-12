#!/usr/bin/env python

import numpy as np
import os
import sys
import pdb
import csv
from tqdm import tqdm
import seaborn as sns

from collections import defaultdict
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import input_args, Database, Post, SimVascular
from vtk_functions import read_geo, write_geo
from get_bc_integrals import integrate_surfaces, integrate_bcs
from simulation_io import get_caps_db, collect_results, collect_results_db, get_dict
from compare_1d import add_image
from get_sv_project import coronary_sv_to_oned
from bc_0d import run_rcr, run_coronary

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
# matplotlib.use('Agg')

def get_cycle(f, n_cycle):
    return np.hstack((f, np.tile(f[1:], n_cycle - 1)))


def cont_func(time, value, n_cycle):
    # repeat time for all cycles
    time_cycle = time
    for i in range(n_cycle):
        time_cycle = np.hstack((time_cycle, time[1:] + (i + 1) * time[-1]))

    # repeat value for all cycles
    value_cycle = np.hstack((value, np.tile(value[1:], n_cycle)))

    # create continues function
    fun = lambda t: interp1d(time_cycle, value_cycle)(t)

    # check if function matches origional values
    assert np.sum((fun(time) - value) ** 2) < 1e-16, 'function does not interpolate data'

    return fun, time_cycle, value_cycle


def run_0d_cycles(flow, time, p, distal_pressure, n_step=100, n_rcr=40, check=True):
    # number of cycles from rcr time constant
    if 'Rd' in p:
        t_rcr = p['C'] * p['Rd']
    elif 'R1' in p:
        t_rcr = p['C2'] * p['R2']
    else:
        raise ValueError('Unknown boundary conditions')
    n_cycle = np.max([int((t_rcr * n_rcr) // time[-1]), n_rcr])

    # 0d time
    t0 = 0.0  # best to set this as zero
    tf = time[-1] * n_cycle
    number_of_time_steps = n_step * n_cycle + 1
    time_0d = np.linspace(t0, tf, number_of_time_steps)

    # continuous inflow function
    Qfunc, _, _ = cont_func(time, flow, n_cycle)

    # output last cycle
    t_last = time[-1] * (n_cycle - 1)
    i_last = np.arange(n_step * (n_cycle - 1), n_step * n_cycle + 1, 1)
    i_prev = np.arange(n_step * (n_cycle - 2), n_step * (n_cycle - 1) + 1, 1)
    t_out = time_0d[i_last] - t_last

    # run 0d simulation
    if 'Rd' in p:
        p_0d = run_rcr(Qfunc, time_0d, p, distal_pressure)
    elif 'R1' in p:
        _, p_v_time, p_v_pres = cont_func(distal_pressure[:, 0], distal_pressure[:, 1], 3)
        p_0d = run_coronary(Qfunc, time_0d, p, p_v_time, p_v_pres, time[-1], 0.0)
    else:
        raise ValueError('Unknown boundary conditions')

    # get last cycle
    p_out = p_0d[i_last]

    # check if solution is periodic
    # delta_p = np.abs(np.mean(p_0d[i_last - 1] - p_0d[i_prev - 1]) / np.mean(p_0d[i_last - 1]))
    if check:
        delta_p = np.abs(p_out[-1] - p_out[0]) / (np.max(p_out) - np.min(p_out))
        assert delta_p < 1.0e-9, 'solution not periodic. diff=' + str(delta_p)

    return t_out, p_out


def compare_0d(db, geo, res, time, m):
    # get boundary conditions
    bc_def = db.get_bcs(geo)
    if bc_def is None:
        return None

    inlet_time = time[m]

    # get outlets
    caps = get_caps_db(db, geo)
    outlets = {}
    for cp, br in caps.items():
        if 'inflow' not in cp:
            outlets[cp] = br

    # initialize output dict
    res_bc = defaultdict(dict)

    # loop all outlets
    for j, (cp, br) in enumerate(outlets.items()):
        print('outlet ' + str(j+1) + '/' + str(len(outlets)))
        # cap bcs
        bc = bc_def['bc'][cp]
        t = bc_def['bc_type'][cp]

        # loop all cardiac cycles
        # for cycle in tqdm(range(1, time[m + '_n_cycle'] + 1)):
        for cycle in [time[m + '_n_cycle']]:
            # bc inlet flow
            inlet_flow = res[br]['flow'][m + '_all'][time[m + '_i_cycle_' + str(cycle)]]

            # output name for pressure in this cycle
            out = 'p_' + str(cycle)

            # select boundary condition
            p = {}
            if t == 'rcr':
                res_bc[br]['t'], res_bc[br][out] = run_0d_cycles(inlet_flow, inlet_time, bc, bc['Po'])
            elif t == 'resistance':
                res_bc[br]['t'] = inlet_time
                res_bc[br][out] = bc['Po'] + bc['R'] * inlet_flow
            elif t == 'coronary':
                if not bc_def['coronary']:
                    continue

                # convert coronary parameters
                cor = coronary_sv_to_oned(bc)
                p['R1'], p['R2'], p['R3'], p['C1'], p['C2'] = (cor['Ra1'], cor['Ra2'], cor['Rv1'], cor['Ca'], cor['Cc'])
                p_v_t = bc_def['coronary'][bc['Pim']][:, 0]
                p_v_p = bc_def['coronary'][bc['Pim']][:, 1]

                res_bc[br]['t'], res_bc[br][out] = run_0d_cycles(inlet_flow, inlet_time, p, np.vstack((p_v_t, p_v_p)).T)

        # copy last cycle
        res_bc[br]['p'] = res_bc[br][out]

    return res_bc


def check_bc(db, geo):
    # collect results
    # if plot_rerun and not os.path.exists(db.get_3d_flow_rerun(geo)):
    #     return

    m = '3d_rerun'
    # m = '0d'

    # get 3d results
    res, time = collect_results_db(db, geo, m)
    if m not in time:
        return

    print('Plotting ' + geo)
    # get 0d results
    if not os.path.exists(db.get_bc_0D_path(geo, m)):
        res_0d = compare_0d(db, geo, res, time, m)
        np.save(db.get_bc_0D_path(geo, m), res_0d)
    else:
        res_0d = np.load(db.get_bc_0D_path(geo, m), allow_pickle=True).item()

    inlet_time = time[m]

    # if res_0d is None or 'p_1' not in res_0d[list(res_0d.keys())[0]]:
    #     return

    # get outlets
    caps = get_caps_db(db, geo)
    outlets = {}
    for cp, br in caps.items():
        if 'inflow' not in cp:
            outlets[cp] = br

    # bounbdary condition types
    bc_def = db.get_bcs(geo)

    # get cap names
    names = db.get_cap_names(geo)

    # get numerical boundary condition time constants
    # res_num = get_dict(db.get_convergence_path())[geo]

    dpi = 300
    if len(outlets) > 50:
        dpi //= 4

    # fields = ['Flow 3D', 'Pressure 3D', 'Pressure 0D']
    # fields = ['Flow 3D', 'Pressure 3D', 'Pressure 0D', 'Pressure Paper']
    # fields = ['Pressure 0D', 'Pressure Paper']
    fields = ['Flow 3D', 'Pressure 3D']
    # fields = ['Pressure 3D', 'Pressure 0D']
    # fields = ['Pressure 3D', 'Pressure 0D', 'Pressure Q const']
    fig, ax = plt.subplots(len(fields), len(outlets), figsize=(len(outlets) * 2 + 2, 4), dpi=dpi, sharex=True, sharey='row')

    # get post-processing constants
    post = Post()

    n_max = 21
    c_max = np.min([time[m + '_n_cycle'], n_max])

    errors = []
    for j, (cp, br) in enumerate(outlets.items()):
        if br not in res_0d:
            continue
        for i, field in enumerate(fields):
            f = field.split()[0].lower()

            # plot settings
            if len(outlets) == 1:
                pos = i
            elif len(fields) == 1:
                pos = j
            else:
                pos = (i, j)
            ax[pos].grid(True)
            if j == 0:
                ax[pos].set_ylabel(field + '\n[' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)
            if i == 0:
                ax[pos].set_title(names[cp])# + ' (' + bc_def['bc_type'][cp].upper() + ')')
            if i == len(fields) - 1:
                ax[pos].set_xlabel('Time [s]')
                ax[pos].set_xlim(0, inlet_time[-1])
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)

            # plot bcs
            for cycle in range(1, c_max + 1):
                ids = field.split()[1].lower()
                if ids == '3d':
                    x = inlet_time
                    y = res[br][f][m + '_all'][time[m + '_i_cycle_' + str(cycle)]]
                elif ids == 'q':
                    inlet_flow = res[br]['flow'][m + '_all'][time[m + '_i_cycle_' + str(cycle)]]
                    if bc_def['bc_type'][cp] == 'rcr':
                        resistance = bc_def['bc'][cp]['Rp'] + bc_def['bc'][cp]['Rd']
                    elif bc_def['bc_type'][cp] == 'coronary':
                        cor = coronary_sv_to_oned(bc_def['bc'][cp])
                        resistance = cor['Ra1'] + cor['Ra2'] + cor['Rv1']
                    x = inlet_time
                    y = np.ones(len(x)) * np.mean(inlet_flow) * resistance
                elif ids == 'paper':
                    if cycle == time[m + '_n_cycle']:
                        x = np.nan
                        y = np.nan
                    else:
                        tau = np.mean(res_num['tau'][f])
                        alpha = 1 / (np.exp(1 / tau) - 1)
                        p0 = res_0d[br]['p_' + str(cycle)]
                        p1 = res_0d[br]['p_' + str(cycle + 1)]

                        x = res_0d[br]['t']
                        y = alpha * p0 + (1 - alpha) * p1
                else:
                    raise RuntimeError('Unknown selection ' + ids)

                i_c = 1 - (cycle - 1) / (c_max - 1)
                ax[pos].plot(x, y * post.convert[f], color=plt.get_cmap('coolwarm_r')(i_c))
            if field == 'Pressure 3D':
                x = res_0d[br]['t']
                y = res_0d[br]['p']
                ax[pos].plot(x, y * post.convert[f], 'k--')

        # calculate error
        diff = interp1d(res_0d[br]['t'], res_0d[br]['p'], fill_value='extrapolate')(inlet_time) - res[br]['pressure'][m + '_cap_last']
        norm = np.max(res[br]['pressure'][m + '_cap']) - np.min(res[br]['pressure'][m + '_cap_last'])
        err = np.mean(np.abs(diff)) / norm
        errors += [err]

    max_err = np.max(errors) * 100
    max_outlet = db.get_cap_names(geo)[list(outlets.keys())[np.argmax(errors)]]

    out_str = geo + ' err=' + '{:05.2f}'.format(max_err) + '% at outlet ' + max_outlet + '\n'
    print(out_str)
    plt.gcf().suptitle(out_str)

    # save figure
    # add_image(db, geo, fig)
    f_out = db.get_bc_comparison_path(geo, m)
    fig.savefig(f_out, bbox_inches='tight')
    plt.close(fig)

    # add error to log
    db.add_bc_err(geo, m, max_err)


def check_first_last(db, geo):
    # collect results
    res, time = collect_results_db_3d_3d(db, geo)
    if res is None:
        return

    return


def plot(db, geometries):
    # read all errors
    errors = get_dict(db.get_bc_err_file('3d'))
    errors_re = get_dict(db.get_bc_err_file('3d_rerun'))

    # color by category
    colors = {'Cerebrovascular': 'k',
              'Coronary': 'r',
              'Aortofemoral': 'm',
              'Pulmonary': 'c',
              'Congenital Heart Disease': 'y',
              'Aorta': 'b',
              'Animal and Misc': '0.75'}

    err = []
    geo = []
    col = []
    for g in geometries:
        if g in errors_re:
            err += [errors_re[g]]
        elif g in errors:
            err += [errors[g]]
        else:
            continue
            # err += [np.nan]
        geo += [g]
        col += [colors[db.get_bcs(g)['params']['deliverable_category']]]

    # sort according to error
    order = np.argsort(err)
    order = np.argsort(geo)

    geo = np.array(geo)[order]
    err = np.array(err)[order]
    col = np.array(col)[order]

    geo_str = '['
    for i in np.where(err > 0.1)[0]:
        geo_str += '\'' + geo[i] + '\', '
    geo_str = geo_str[:-2] + ']'
    # print(geo_str)

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))
    plt.cla()
    plt.yscale('log')
    ax1.bar(np.arange(len(err)), err, color=col)
    ax1.yaxis.grid(True)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
    ax1.set_ylim(0.01, 100)
    plt.xticks(np.arange(len(err)), geo, rotation='vertical')
    plt.ylabel('Max. outlet pressure error 3D vs. 0D BC')
    fname = os.path.join(db.fpath_gen, 'bc_err.png')
    # plt.legend(list(colors.keys()))
    fig1.savefig(fname, bbox_inches='tight')

    # write to csv
    with open(os.path.join(db.fpath_gen, 'bc_err.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for g, e in zip(geo, err):
            writer.writerow([g, str(e)])


def compare_rcr():
    params = {'Rp': 100, 'Rd': 1000, 'C': 0.001}
    distal_pressure = 0

    q_mean = 1
    nt = 1000
    nc = 10
    T = 1
    time = np.linspace(0, T*nc, nt*nc+1)
    q_var = lambda x: q_mean + np.cos(x * 2 * np.pi / T)
    q_const = lambda x: q_mean
    flows = {'var': q_var, 'const': q_const}

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))

    out = defaultdict(list)
    for m, q in flows.items():
        pressure = run_rcr(q, time, params, distal_pressure)
        for i in range(nc):
            out[m] += [np.mean(pressure[np.arange(i * nt, (i + 1) * nt + 1)])]
        # pdb.set_trace()

        ax1.plot(np.arange(nc), out[m], 'o--')
    plt.show()
    pdb.set_trace()

def main(db, geometries):
    for geo in geometries:
        check_bc(db, geo)


if __name__ == '__main__':
    descr = 'Check RCR boundary condition of 3d simulation'
    d, g, _ = input_args(descr)
    main(d, g)
    # plot(d, g)
    # compare_rcr()
