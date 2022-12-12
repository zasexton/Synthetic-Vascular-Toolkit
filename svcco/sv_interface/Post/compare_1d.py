#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse
import pdb

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# for styles, see https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
# plt.style.use('ggplot')

import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.font_manager
from collections import OrderedDict, defaultdict
from scipy.interpolate import interp1d

from common import rec_dict, get_dict
from get_database import Database, Post, input_args
from simulation_io import get_caps_db, collect_results_db, collect_results_db_3d_3d, \
    collect_results_db_3d_3d_spatial

sys.path.append('/home/pfaller/work/repos/SimVascular_fork/Python/site-packages/')

import sv_rom_simulation as oned

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})

def add_image(db, geo, fig):
    if not os.path.exists(db.get_png(geo)):
        return
    im = plt.imread(db.get_png(geo))
    newax = fig.add_axes([-0.42, 0.1, 0.5, 0.5], anchor='E')#, zorder=-1
    newax.imshow(im)
    newax.axis('off')


def plot_1d_3d_all(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get models
    models = [k[:-4] for k in time.keys() if '_all' in k]
    if '3d' in models and '3d_rerun' in models and opt['exclude_old']:
        models.remove('3d')

    # get 1d/3d map
    caps = get_caps_db(db, geo)

    fig, ax = plt.subplots(len(post.fields), len(models), figsize=(20, 10), dpi=opt['dpi'], sharex='col', sharey='row')

    for i, f in enumerate(post.fields):
        for j, m in enumerate(models):
            if len(models) > 1:
                pos = (i, j)
            else:
                pos = i
            ax[pos].grid(True)

            if opt['legend_row'] or i == 0:
                ax[pos].set_title((geo + ' ' + m + ' ' + f).replace('_', '\_'))
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[pos].set_xlabel('Time [s]')
            if opt['legend_col'] or j == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            for c in caps.values():
                ax[pos].plot(time[m + '_all'], res[c][f][m + '_all'] * post.convert[f])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'all'))
    plt.close(fig)


def plot_1d_3d_cyclic(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c.replace('_', '\_'))
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            res_split = np.split(res[c][f]['1d_all'][:time['step_cycle'] * time['n_cycle']], time['n_cycle'])
            for r in res_split:
                ax[i, j].plot(time['1d'], r * post.convert[f], post.styles['1d'])

            ax[i, j].plot(time['3d'], res[c][f]['3d'] * post.convert[f], post.styles['3d'])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'cyclic'))
    plt.close(fig)


def plot_1d_3d_caps(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get models
    models = [k[:-4] for k in time.keys() if '_all' in k]
    if '3d' in models and '3d_rerun' in models and opt['exclude_old']:
        models.remove('3d')

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    
    # get cap names
    names = db.get_cap_names(geo)

    # get 0d bcs
    bc_0d = get_dict(db.get_bc_0D_path(geo, '3d_rerun'))
    
    if len(caps) > 50:
        dpi = opt['dpi'] // 2
        sharey = opt['sharey']
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fields = post.fields
    if 'area' in fields:
        fields.remove('area')

    fig, ax = plt.subplots(len(fields), len(caps), figsize=(opt['w'], opt['h']), dpi=dpi, sharex=opt['sharex'], sharey=sharey)

    for i, f in enumerate(fields):
        for j, (c, br) in enumerate(caps.items()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(names[c].replace('_', '\_'))
            if opt['legend_row'] or i == len(fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in models:
                if m == '3d':
                    i0 = 1
                else:
                    i0 = 0
                ax[i, j].plot(time[m][i0:], res[br][f][m + '_cap_last'][i0:] * post.convert[f], post.styles[m], color=post.color[m])
                lg += [post.model_names[m]]

            # if f == 'pressure' and bc_0d:
            #     if br in bc_0d:
            #         ax[i, j].plot(bc_0d[br]['t'], bc_0d[br]['p'] * post.convert[f], 'k--')
            #         lg += ['0D BC']
            #     pres = res[br][f][m + '_cap']
            #     delta_p = np.abs(pres[-1] - pres[0]) / (np.max(pres) - np.min(pres))
            #     print(names[c], '{:.2e}'.format(delta_p))
            if geo == '0069_0001' and f == 'pressure':
                ax[i, j].set_ylim(55, 105)

    ax[0, 0].legend(lg, bbox_to_anchor=(-0.5, 1), loc='upper right')

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'caps'), bbox_inches='tight')
    plt.close(fig)


def plot_1d_3d_interior(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get models
    models = [k[:-4] for k in time.keys() if '_all' in k]
    if '3d' in models and '3d_rerun' in models and opt['exclude_old']:
        models.remove('3d')

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    cap_br = list(caps.values())
    cap_names = list(caps.keys())

    # get cap names
    names = db.get_cap_names(geo)

    if len(res) > 50:
        dpi = opt['dpi'] // 2
        sharey = opt['sharey']
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fig, ax = plt.subplots(len(post.fields), len(res), figsize=(opt['w'], opt['h']), dpi=dpi, sharex='col', sharey=sharey)

    # pick reference time step with highest inflow
    for m in models:
        if '3d' in m:
            m_ref = m
            t_max = {m_ref: np.argmax(res[0]['flow'][m_ref + '_cap_last'])}

    # pick closest time step for other models
    for m in models:
        if m != m_ref:
            t_max[m] = np.argmin(np.abs(time[m_ref][t_max[m_ref]] - time[m]))

    area_diff = []
    for i, f in enumerate(post.fields):
        for j, br in enumerate(res.keys()):
            if len(res.keys()) > 1:
                pos = (i, j)
            else:
                pos = i
            ax[pos].grid(True)

            if opt['legend_row'] or i == 0:
                if br in cap_br:
                    name = names[cap_names[cap_br.index(br)]]
                    if not name.isupper():
                        name = name.capitalize()
                else:
                    name = 'Branch ' + str(br)
                ax[pos].set_title(name.replace('_', '\_'))
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[pos].set_xlabel('Vessel path [-]')
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in models:
                path = res[br][m + '_path']
                ax[pos].plot(path / path[-1], res[br][f][m + '_int_last'][:, t_max[m]] * post.convert[f], post.styles[m], color=post.color[m])
                lg.append(post.model_names[m])

                # maximum area change in branch over time
                if m == '1d' and f == 'area':
                    area = res[br][f][m + '_int_last'] * post.convert[f]
                    area_diff += [np.max((np.max(area, axis=1) - np.min(area, axis=1))/np.mean(area, axis=1))]

            ax[pos].set_xlim(0, 1)
            if geo == '0074_0001' and f == 'pressure':
                ax[pos].set_ylim(64, 72)
            if i == 0 and j == 0:
                ax[pos].legend(lg, bbox_to_anchor=(-0.5, 1), loc='upper right')

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'interior'), bbox_inches='tight')
    plt.close(fig)

    if '1d' in models and 'area' in post.fields:
        print('  max. area change: ' + '{:.2f}'.format(np.max(area_diff) * 100) + '%')


def plot_1d_3d_paper(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    # 0075_1001
    # names = {'inflow': 'Inflow', 'aorta_outflow': 'Aorta', 'Lt-carotid_segs_final': 'Left Carotid',
    #          'rt-carotid_segs_final': 'Right Carotid', 'btrunk_segs_final': 'Right Subclavian',
    #          'subclavian_segs_final': 'Left Subclavian'}
    # myorder = ['inflow', 'aorta_outflow', 'Lt-carotid_segs_final', 'rt-carotid_segs_final', 'subclavian_segs_final', 'btrunk_segs_final']
    # myorder = ['aorta_outflow', 'rt-carotid_segs_final', 'btrunk_segs_final']

    names = {'inflow': 'Aorta', 'R_int_iliac': 'Right Internal Iliac', 'R_ext_iliac': 'Right External Iliac',
             'SMA': 'Superior Mesenteric', 'IMA': 'Inferior Mesenteric'}
    myorder = ['IMA', 'R_int_iliac', 'R_ext_iliac']

    # layout
    nx = 2
    ny = 3

    # plot caps
    fig, ax = plt.subplots(ny, nx, figsize=(7, 6), dpi=opt['dpi'], sharex=True, sharey='col')
    for i, f in enumerate(post.fields):
        for j, c in enumerate(myorder):
            # pos = np.unravel_index(j, (ny, nx))
            pos = (j, i)
            ax[pos].grid(True)
            ax[pos].set_title(names[c])

            if opt['legend_row'] or pos[0] == ny - 1:
                ax[pos].set_xlabel('Time [s]')
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or pos[1] == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)

            for m in post.models:
                ax[pos].plot(time['3d'], res[c][f][m] * post.convert[f], post.styles[m], color=post.colors[m])
            plt.sca(ax[pos])
            plt.xlim([0, plt.xlim()[1]])

    fig.savefig(db.get_post_path(geo, 'paper_caps'), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

    # pick time step
    t_max = np.argmax(res['flow']['inflow']['3d'])

    for i, f in enumerate(post.fields):
        fig, ax = plt.subplots(ny, nx, figsize=(9, 8), dpi=opt['dpi'], sharex=True, sharey=True)
        for j, c in enumerate(myorder):
            pos = np.unravel_index(j, (ny, nx))
            ax[pos].grid(True)
            ax[pos].set_title(names[c])
            ax[pos].set_ylim([0, 110])

            if opt['legend_row'] or pos[0] == ny - 1:
                ax[pos].set_xlabel('Vessel path [1]')
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or pos[1] == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)

            for m in post.models:
                ax[pos].plot(res['path'][c][m], res[c][f][m + '_int'][t_max] * post.convert[f], post.styles[m], color=post.colors[m])

        fig.savefig(db.get_post_path(geo, 'paper_interior_' + f), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def calc_error_all(db, geo, res, time):
    # reduced model to compare
    model_order = ['0d', '1d', '3d']
    for m_rom in model_order:
        if m_rom not in time:
            continue
        if m_rom == '3d' and '3d_rerun' not in time:
            continue

        # get reference solution
        models = [k[:-4] for k in time.keys() if '_all' in k]
        if '3d_rerun' in models:
            m_ref = '3d_rerun'
        else:
            # m_ref = '3d'
            # raise RuntimeError('No 3d_rerun results for ' + geo)
            return

        err = calc_error(db, geo, res, time, m_rom, m_ref)

        if m_rom == '0d':
            db.add_0d_3d_comparison(geo, err)
        elif m_rom == '1d':
            db.add_1d_3d_comparison(geo, err)
        elif m_rom == '3d':
            db.add_3d_3d_comparison(geo, err)

    if '0d' in time and '1d' in time:
        err = calc_error(db, geo, res, time, '0d', '1d')
        db.add_0d_1d_comparison(geo, err)


def calc_error(db, geo, res, time, m_rom, m_ref):
    # set post-processing fields
    fields = ['flow', 'pressure']

    # get 1d/3d map
    caps = get_caps_db(db, geo)

    # interpolate 1d to 3d in space and time (allow extrapolation due to round-off errors at bounds)
    interp = lambda x_1d, y_1d, x_3d: interp1d(x_1d, y_1d.T, fill_value='extrapolate')(x_3d)

    # relative difference between two arrays
    rel_diff = lambda a, b: np.abs((a - b) / b)

    # branches (interior and caps)
    branches = {'int': {}, 'cap': {}}
    for f in fields:
        branches['int'][f] = list(res.keys())
        branches['cap'][f] = list(caps.values())

    # skip inlet branch for flow (flow is prescribed)
    for d in branches.keys():
        if len(branches[d]['flow']) > 1:
            branches[d]['flow'].remove(0)

    # pick systole and diastole time points
    inflow = res[0]['flow'][m_ref + '_cap_last']
    times = {'sys': np.argmax(inflow), 'dia': np.argmin(inflow)}

    # get spatial errors
    err = rec_dict()
    for f in fields:
        for br in branches['int'][f]:
            # retrieve 3d results
            res_3d = res[br][f][m_ref + '_int_last']

            # map paths to interval [0, 1]
            path_1d = res[br][m_rom + '_path'] / res[br][m_rom + '_path'][-1]
            path_3d = res[br][m_ref + '_path'] / res[br][m_ref + '_path'][-1]

            # interpolate in space and time
            res_1d = interp(path_1d, res[br][f][m_rom + '_int_last'], path_3d)
            res_1d = interp(time[m_rom], res_1d, time[m_ref])

            # calculate spatial error (eliminate time dimension)
            if f == 'flow' or (f == 'area' and m_rom == '1d'):
                norm = np.max(res_3d, axis=1) - np.min(res_3d, axis=1)
            else:
                norm = np.mean(res_3d, axis=1)

            # difference between ref and rom
            diff = np.abs(res_1d - res_3d).T

            # loop through different error metrics
            for n in ['abs', 'rel']:
                if n == 'rel':
                    delta = diff / norm
                else:
                    delta = diff
                for tn, ti in times.items():
                    err[f]['spatial'][tn][n][br] = delta[ti]
                err[f]['spatial']['avg'][n][br] = np.mean(delta, axis=0)
                err[f]['spatial']['max'][n][br] = np.max(delta, axis=0)

    # get mean errors
    for f in fields:
        for m in err[f]['spatial'].keys():
            for n in err[f]['spatial'][m].keys():
                # get interior error
                for br in branches['int'][f]:
                    err[f]['int'][m][n][br] = np.mean(err[f]['spatial'][m][n][br])

                # get cap error
                for br in branches['cap'][f]:
                    if len(branches['cap'][f]) > 1 and br == 0:
                        # inlet
                        i_cap = 0
                    else:
                        # outlet
                        i_cap = -1
                    err[f]['cap'][m][n][br] = err[f]['spatial'][m][n][br][i_cap]

                # get error over all branches
                err[f]['int'][m][n]['all'] = np.mean([err[f]['int'][m][n][br] for br in branches['int'][f]])
                err[f]['cap'][m][n]['all'] = np.mean([err[f]['cap'][m][n][br] for br in branches['cap'][f]])
    return err


def main(db, geometries):
    # get post-processing constants
    post = Post()

    for geo in geometries:
        # generate plots
        print(geo)

        # read results
        deform = db.study == 'deformable'
        res, time = collect_results_db(db, geo, post.models, deformable=deform)

        # plot options
        opt = {'legend_col': False,
               'legend_row': False,
               'sharex': True,
               'sharey': 'row',
               'dpi': 200,
               'w': 0.5 * (len(db.get_surface_names(geo)) * 3 + 4),
               'h': 1 * (len(Post().fields) * 1 + 2),
               'exclude_old': True}

        # calculate error
        if ('0d' in time and '1d' in time) and '3d_rerun' in time:
            print('  claculating error')
            calc_error_all(db, geo, res, time)
    
        if '3d_rerun' in time and '1d' in time:# or '3d' in time:
            print('  plotting')
            plot_1d_3d_all(db, opt, geo, res, time)
            plot_1d_3d_caps(db, opt, geo, res, time)
            plot_1d_3d_interior(db, opt, geo, res, time)

        # plot_1d_3d_paper(db, opt, geo, res, time)
        # plot_1d_3d_cyclic(db, opt, geo, res, time)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
