#!/usr/bin/env python

import numpy as np
import os
import pdb

from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import input_args, Database, Post, SimVascular
from vtk_functions import read_geo, write_geo
from get_bc_integrals import integrate_surfaces, integrate_bcs

import matplotlib.pyplot as plt


def fourier(x, n_sample_freq=128):
    """
    Inverse fourier transformation from frequencies in x (real, imaginary)
    """
    assert x.shape[0] % 2 == 0, 'odd number of parameters'
    n_mode = x.shape[0] // 2

    x_complex = x[:n_mode] + 1j * x[n_mode:]
    inflow_fft = np.zeros(n_sample_freq + 1, dtype=complex)
    inflow_fft[:n_mode] = x_complex

    return np.fft.irfft(inflow_fft)


def error(time, inflow, time_smooth, inflow_smooth):
    """
    Get error between input inflow and smooth inflow
    """
    # repeat last value at the start
    time_smooth = np.insert(time_smooth, 0, 0)
    inflow_smooth = np.insert(inflow_smooth, 0, inflow_smooth[-1])

    # interpolate to coarse time
    inflow_interp = interp1d(time_smooth, inflow_smooth)(time)

    return np.sqrt(np.sum((inflow - inflow_interp) ** 2))


def optimize_inflow(time, inflow, n_mode=10, n_sample_real=256, debug=False):
    """
    Optimize fourier-smoothed inflow to interpolate input inflow
    """
    # define fourier smoothing
    assert n_sample_real % 2 == 0, 'odd number of samples'
    n_sample_freq = n_sample_real // 2

    # insert last 3d time step as 1d initial condition (periodic solution)
    time = np.insert(time, 0, 0)
    inflow = np.insert(inflow, 0, inflow[-1])

    # linearly interpolate at fine time points
    time_smooth = np.linspace(0, time[-1], n_sample_real + 1)[1:]
    inflow_interp_lin = interp1d(time, inflow)(time_smooth)

    # get starting value from fft
    inflow_fft = np.fft.rfft(inflow_interp_lin)
    x0 = inflow_fft[:n_mode]
    x0_split = np.array(np.hstack((np.real(x0), np.imag(x0))))

    # setup otimization problem
    run = lambda x: error(time, inflow, time_smooth, fourier(x, n_sample_freq))

    # optimize frequencies to match inflow profile
    res = minimize(run, x0_split, tol=1.0e-8, options={'disp': debug})
    inflow_smooth = fourier(res.x, n_sample_freq)

    # add time step zero
    time_smooth = np.insert(time_smooth, 0, 0)
    inflow_smooth = np.insert(inflow_smooth, 0, inflow_smooth[-1])

    # re-sample to n_sample_real
    time_out = np.linspace(0, time_smooth[-1], n_sample_real)
    inflow_out = interp1d(time_smooth, inflow_smooth)(time_out)

    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
        ax.plot(time_smooth[1:], fourier(x0_split), 'b-')
        ax.plot(time_smooth, inflow_smooth, 'r-')
        ax.plot(time, inflow, 'kx')
        plt.grid()
        plt.show()

    return time_out, inflow_out


def error_phase(t1, f1, t2, f2, dt):
    # period
    p = t2[-1]

    # shift and replicate
    t2_cyc = np.hstack((t2[:-1] - p, t2, t2[1:] + p)) + dt
    f2_cyc = np.hstack((f2[:-1], f2, f2[1:]))

    # interpolate to coarse time
    f2_interp = interp1d(t2_cyc, f2_cyc)(t1)

    return np.sum((f1 - f2_interp) ** 2)


def phase_shift(t, f, dt):
    # period
    p = t[-1]

    # shift and replicate
    t_cyc = np.hstack((t[:-1] - p, t, t[1:] + p)) + dt
    f_cyc = np.hstack((f[:-1], f, f[1:]))

    # interpolate to original time
    return interp1d(t_cyc, f_cyc)(t)


def optimize_phase(t1, f1, t2, f2, debug=False):
    # setup otimization problem
    run = lambda x: error_phase(t1, f1, t2, f2, x)

    # find time shift to match inflow profile
    res = minimize(run, [0.15], tol=1.0e-8, options={'disp': debug}, bounds=((-t2[-1]/4, t2[-1]/4), ))

    return res.x[0]


def read_velocity(f_dat):
    """
    Read velocity, time steps, and node ids from bct.dat
    """
    # read text file
    with open(f_dat) as f:
        lines = f.readlines()

    # get number of points and time steps
    n_p, n_t = (int(l) for l in lines[0].strip().split())

    # read points
    vel = []
    time = []
    points = []
    coords = []
    for i in range(n_p):
        # line of point header
        split = lines[1 + i * (n_t + 1)].strip().split()

        # point coordinates
        coords += [[float(split[i]) for i in range(3)]]

        # point id
        points += [int(split[-1])]

        # read time steps
        vel_p = []
        for j in range(n_t):
            # line of time step
            split = lines[2 + i + i * n_t + j].split()

            # velocity vector
            vel_p += [[float(split[i]) for i in range(3)]]

            # time
            if i == 0:
                time += [float(split[-1])]

        vel += [vel_p]
    return np.array(vel), np.array(time), np.array(points), np.array(coords)


def write_velocity(f_dat, vel, time, points, coords):
    """
    Write bct.dat file
    """
    # get dimensions
    n_p, n_t, dim = vel.shape

    assert n_p == points.shape[0], 'number of points mismatch'
    assert n_p == coords.shape[0], 'number of coordinates mismatch'
    assert n_t == time.shape[0], 'number of time steps mismatch'
    assert dim == 3, 'number of dimensions mismatch'
    assert coords.shape[1] == 3, 'number of dimensions mismatch'

    with open(f_dat, 'w+') as f:
        # write header
        f.write(str(n_p) + ' ' + str(n_t) + '\n')

        # write points
        for i in range(n_p):
            # write point
            for j in range(3):
                f.write("{:.6e}".format(coords[i, j]) + ' ')
            f.write(str(n_t) + ' ' + str(points[i]) + '\n')

            # write time steps
            np.savetxt(f, np.vstack((vel[i].T, time)).T, fmt='%1.6e')


def add_velocity(inlet, vel, time, points):
    """
    Add velocity vectors to inlet geometry bct.vtp
    """
    # get unique point ids
    ids = v2n(inlet.GetPointData().GetArray('GlobalNodeID'))

    # remove all point arrays except GLobalNodeId
    names = [inlet.GetPointData().GetArrayName(i) for i in range(inlet.GetPointData().GetNumberOfArrays())]
    for n in names:
        if n != 'GlobalNodeID':
            inlet.GetPointData().RemoveArray(n)

    # add velocity vectors to nodes
    for i, t in enumerate(time):
        # create new array for time step
        array = vtk.vtkDoubleArray()
        array.SetNumberOfComponents(3)
        array.SetNumberOfTuples(vel.shape[0] * 3)
        array.SetName('velocity_' + str(t))
        inlet.GetPointData().AddArray(array)

        # fill array
        for j, p in enumerate(points):
            k = np.where(ids == p)[0][0]
            v = vel[j, i]
            array.SetTuple3(k, v[0], v[1], v[2])


def integrate_inlet(f_in):
    """
    Get inlet flow from bct.dat and bct.vtp
    """
    # read inlet geometry from bct.vtp
    inlet = read_geo(f_in + '.vtp').GetOutput()

    # integrate over inlet
    return integrate_surfaces(inlet, inlet.GetCellData(), 'velocity', face_array='ModelFaceID')


def overwrite_inflow(db, geo, n_sample_real=256):
    """
    Overwrite bct.dat and bct.vtp from svpre with own high-fidelity inflow
    """
    # define project paths
    f_in = os.path.join(db.get_solve_dir_3d(geo), 'bct')

    # read inflow from file
    time, inflow = db.get_inflow_osmsc(geo)

    # fit inflow using fourier smoothing
    time, inflow = optimize_inflow(time, inflow, n_sample_real)

    # read constant inflow
    vel_dat, time_dat, points, coords = read_velocity(f_in + '.dat')

    # integrate inflow from from bct.dat and bct.vtp
    surf_int = integrate_inlet(f_in)

    # scale velocity
    pdb.set_trace()
    vel_scaled = vel_dat / surf_int['velocity'] * np.expand_dims(inflow, axis=1)

    # overwrite bct.dat
    write_velocity(f_in + '.dat', vel_scaled, time, points, coords)

    # overwrite bct.vtp
    inlet = read_geo(f_in + '.vtp').GetOutput()
    add_velocity(inlet, vel_scaled, time, points)
    write_geo(f_in + '.vtp', inlet)


def check_inflow(db, geo):
    post = Post()

    # create output folder
    check_dir = os.path.join(db.get_solve_dir_3d(geo), 'check')
    os.makedirs(os.path.join(check_dir), exist_ok=True)

    # define project paths
    f_in = os.path.join(db.get_solve_dir_3d(geo), 'bct')
    f_out_fig = os.path.join(check_dir, geo + '_inflow')
    f_out_vtp = os.path.join(check_dir, 'initial.vtp')

    # read inflow from file
    time, inflow = db.get_inflow(geo)
    time_smooth, inflow_smooth = db.get_inflow_smooth(geo)

    # get model inlet from bct.dat and bct.vtp
    surf_int = integrate_inlet(f_in)

    # postproc initial conditions
    sv = SimVascular()
    sv.run_post(db.get_solve_dir_3d(geo), ['-start', '0', '-stop', '0', '-incr', '1', '-vtkcombo', '-vtp', 'check/initial.vtp'])

    # get initial conditions
    fpath_surf = os.path.join(db.get_solve_dir_3d(geo), 'mesh-complete', 'mesh-surfaces', 'inflow.vtp')
    ini = integrate_bcs(fpath_surf, f_out_vtp, ['pressure', 'velocity'], face_array='ModelFaceID')

    # plot comparison
    fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
    plt.plot(time_smooth, inflow_smooth * post.convert['flow'], 'g-')
    plt.plot(surf_int['time'], surf_int['velocity'][:, -1] * post.convert['flow'], 'r--')
    plt.plot(0, ini['velocity'][0][-1] * post.convert['flow'], 'bo', fillstyle='none')
    plt.plot(time, inflow * post.convert['flow'], 'kx')
    plt.xlabel('Time [s]')
    plt.ylabel('Flow [l/h]')
    plt.title('Initial pressure ' + '{:2.1f}'.format(ini['pressure'][0][-1] * post.convert['pressure']) + ' mmHg')
    plt.grid()
    ax.legend(['Optimized for rerun', 'SimVascular', 'Initial condition for rerun', 'OSMSC'])
    fig.savefig(f_out_fig, bbox_inches='tight')
    plt.cla()


def fix_inflows(db, geo):
    plot = False

    t1, f1 = db.get_inflow(geo)
    if t1 is None:
        return

    if geo == '0174_0000':
        geo_in = '0176_0000'
    elif geo == '0176_0000':
        geo_in = '0174_0000'
    else:
        geo_in = geo
    t2, f2 = db.get_inflow_osmsc(geo_in)

    if len(t2) - 1 == len(t1):
        # optimize an inflow profile
        print('  only coarse inflow available')
        t2, f2 = optimize_inflow(t1, f1)
        plot = True
    else:
        # minimize difference between inflows by shifting in time
        dt = optimize_phase(t1, f1, t2, f2)
        f2 = phase_shift(t2, f2, dt)

    # interpolate to coarse time steps
    f12 = interp1d(t2, f2)(t1)

    # fix scaling
    if geo == '0174_0000' or geo == '0176_0000':
        a = np.min(f1) / np.min(f12)
        f12 *= a
        f2 *= a

    # error between flow profiles
    err = np.max(np.abs(f1 - f12)) / np.abs(np.mean(f1))

    print(geo, "{:.2e}".format(err))

    # save
    if err < 0.1:
        np.savetxt(db.get_inflow_smooth_path(geo), np.vstack((t2, f2)).T)

    if plot:
        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
        post = Post()
        plt.plot(t1, f1 * post.convert['flow'], 'k-')
        plt.plot(t2, f2 * post.convert['flow'], 'r--')
        plt.xlabel('Time [s]')
        plt.ylabel('Flow [l/h]')
        ax.legend(['Inflow from results', 'Inflow from extras'])
        ax.grid(True)
        f_out = os.path.join('/home/pfaller/work/osmsc/data_generated/check_inflows', geo)
        fig.savefig(f_out, bbox_inches='tight')
        plt.close(fig)

    return False


def main(db, geometries):
    for geo in geometries:
        # print('Checking geometry ' + geo)
        # check_inflow(db, geo)
        fix_inflows(db, geo)

if __name__ == '__main__':
    descr = 'Check inlet flow of 3d simulation'
    d, g, _ = input_args(descr)
    main(d, g)
