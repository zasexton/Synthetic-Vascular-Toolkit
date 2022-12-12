#!/usr/bin/env python

import argparse
import os

import numpy as np
from collections import defaultdict

def get_dict(fpath):
    """
    Read .npy dictionary saved with numpy.save if path is defined and exists
    Args:
        fpath: path to .npy file

    Returns:
        dictionary
    """
    if fpath is not None and os.path.exists(fpath):
        return np.load(fpath, allow_pickle=True).item()
    else:
        return {}


def coronary_sv_to_oned(bc):
    """
    Convert format of coronary boundary condition parameters from svSimVascular to svOneDSolver
    """
    # unpack constants
    p1, p2, q0, q1, q2, b1 = (bc['p1'], bc['p2'], bc['q0'], bc['q1'], bc['q2'], bc['b1'])
    Rv_micro = 0.0

    # build system of equations (obtained from analytically solving for constants as defined in paper)
    # see H. J. Kim et al. "Patient-Specific Modeling of Blood Flow and Pressure in Human Coronary Arteries", p. 3198
    Ra = q2 / p2
    Ra_micro = (p1 ** 2 * q2 ** 2 - 2 * p1 * p2 * q1 * q2 + p2 ** 2 * q1 ** 2) / (
            p2 * (- q2 * p1 ** 2 + q1 * p1 * p2 - q0 * p2 ** 2 + q2 * p2))
    Ca = -p2 ** 2 / (p1 * q2 - p2 * q1)
    Cim = (- q2 * p1 ** 2 + q1 * p1 * p2 - q0 * p2 ** 2 + q2 * p2) ** 2 / ((p1 * q2 - p2 * q1) * (
            p1 ** 2 * q0 * q2 - p1 * p2 * q0 * q1 - p1 * q1 * q2 + p2 ** 2 * q0 ** 2 - 2 * p2 * q0 * q2 + p2 * q1 ** 2 + q2 ** 2))
    Rv = -(
            p1 ** 2 * q0 * q2 - p1 * p2 * q0 * q1 - p1 * q1 * q2 + p2 ** 2 * q0 ** 2 - 2 * p2 * q0 * q2 + p2 * q1 ** 2 + q2 ** 2) / (
                 - q2 * p1 ** 2 + q1 * p1 * p2 - q0 * p2 ** 2 + q2 * p2)

    # check equation residuals
    res = [p1 - (Ra_micro * Ca + (Rv + Rv_micro) * (Ca + Cim)),
           p2 - (Ca * Cim * Ra_micro * (Rv + Rv_micro)),
           q0 - (Ra + Ra_micro + Rv + Rv_micro),
           q1 - (Ra * Ca * (Ra_micro + Rv + Rv_micro) + Cim * (Ra + Ra_micro) * (Rv + Rv_micro)),
           q2 - (Ca * Cim * Ra * Ra_micro * (Rv + Rv_micro)),
           b1 - (Cim * (Rv + Rv_micro))]
    assert np.max(np.abs(res)) < 1e-5, 'SV coronary constants inconsistent'

    # export constants
    return {'Ra1': Ra, 'Ra2': Ra_micro, 'Ca': Ca, 'Cc': Cim, 'Rv1': Rv, 'P_v': 0.0}


def rec_dict():
    """
    Recursive defaultdict
    """
    return defaultdict(rec_dict)
