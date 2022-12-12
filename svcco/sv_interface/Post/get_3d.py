#!/usr/bin/env python

import os
import sys
import shutil
import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from get_sv_project import Project
from get_sv_meshes import get_meshes
from get_database import input_args, SimVascular, run_command
from inflow import overwrite_inflow, check_inflow

sys.path.append(os.path.join(SimVascular().perigee, '..'))
# from inflow_fourier_fit import fit_fourier_series


def generate_3d_svsolver(db, geo):
    # simvascular object
    sv = SimVascular()

    mode = ''
    if db.study in ['steady', 'steady0', 'irene']:
        mode = db.study

    try:
        # get meshes
        get_meshes(db, geo)

        # create sim-vascular project
        pj = Project(db, geo, mode)
        pj.create_sv_project()
        
        # run pre-processor
        sv.run_pre(db.get_solve_dir_3d(geo), db.get_svpre_file(geo, 'svsolver'))

        # # write new high-fidelity inflow (don't use what svpre is writing)
        # overwrite_inflow(db, geo, n_sample_real)

        # check inflow
        check_inflow(db, geo)

        # copy to study folder
        src = db.get_solve_dir_3d(geo)
        dst = os.path.join(db.fpath_solve, geo + '_' + db.study)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except Exception as e:
        return e

    return False


def write_inflow_perigee(db, geo):
    # perigee inflow file
    f_out_txt = os.path.join(db.get_solve_dir_3d_perigee(geo), 'inflow_fourier_series.txt')
    f_out_png = os.path.join(db.get_solve_dir_3d_perigee(geo), 'inflow_fourier_series.png')

    # read inflow conditions
    time, inflow = db.get_inflow_smooth(geo)

    y = np.fft.rfft(inflow)

    an = y[:-1].real
    bn = y[:-1].imag

    n_mode = an.shape[0] - 1
    period = time[-1]
    omega = y[-1].real

    # Write out 'inflow_fourier_series.txt'
    with open(f_out_txt, 'w') as f:
        f.write('# num_fourier_modes / fundamental frequency (w) / period\n')
        f.write(str(n_mode) + ' ' + str(omega) + ' ' + str(period) + '\n')

        f.write('\n# Coefficients a_n with length = num_fourier_modes + 1\n')
        f.write(' '.join(map(str, an)) + '\n')

        f.write('\n# Coefficients b_n with length = num_fourier_modes + 1\n')
        f.write(' '.join(map(str, bn)) + '\n')

    # fun = an[0] + np.dot(an[1:], np.cos(np.outer(np.arange(n_mode) + 1, omega * time))) \
    #             + np.dot(bn[1:], np.sin(np.outer(np.arange(n_mode) + 1, omega * time)))
    #
    # plt.plot(time, inflow, 'g-')
    # plt.plot(time, fun, 'r--')
    # plt.ylabel('Inflow [mL/s]')
    # plt.xlabel('Time [s]')
    # plt.savefig(f_out_png)
    # # this produced a bunch of nans
    # # f_in = db.get_sv_flow_path(geo, '3d')
    # # fit_fourier_series(f_in, f_out_png, f_out_txt, 10)
    # pdb.set_trace()


def perigee_path(db, geo, cmd, opt):
    for o, n in opt.items():
        if isinstance(n, str):
            cmd += ['-' + o, os.path.join(db.get_solve_dir_3d_perigee(geo), n)]
        else:
            cmd += ['-' + o, str(n)]


def generate_3d_perigee(db, geo):
    # simvascular object
    sv = SimVascular()

    # prepare steps for svsolver
    # svproj.create_sv_project(db, geo)

    write_inflow_perigee(db, geo)

    # sv_conv = os.path.join(sv.perigee, 'sv_converter')
    svpre = os.path.join(sv.perigee, 'svpre_converter')
    svrcr = os.path.join(sv.perigee, 'svrcr_converter')

    svpre_opt = {'vol_geo_name': 'whole_vol.vtu',
                 'wal_geo_name': 'wall_vol.vtp',
                 'inl_geo_name': 'inflow_vol.vtp',
                 'out_geo_base': 'outflow_vol_'}

    svpre_cmd = [svpre, '-svpre_file', db.get_svpre_file(geo, 'perigee')]
    perigee_path(db, geo, svpre_cmd, svpre_opt)

    svrcr_opt = {'num_outlet': len(db.get_outlet_names(geo))}
    # 'out_rcr_file': 'lpn_rcr_input.txt'

    svrcr_cmd = [svrcr, '-sv_rcr_file', 'rcrt.dat']
    perigee_path(db, geo, svrcr_cmd, svrcr_opt)

    run_folder = db.get_solve_dir_3d(geo)

    # run_command(run_folder, svpre_cmd)
    run_command(run_folder, svrcr_cmd)

    shutil.move(os.path.join(db.get_solve_dir_3d(geo), 'lpn_rcr_input.txt'),
                os.path.join(db.get_solve_dir_3d_perigee(geo), 'lpn_rcr_input.txt'))


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        generate_3d_svsolver(db, geo)
        # generate_3d_perigee(db, geo)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
