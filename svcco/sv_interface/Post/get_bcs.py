#!/usr/bin/env python

import re
import tkinter
import pdb
from collections import defaultdict
import numpy as np

def get_dic(st):
    """
    assemble string into dict
    :param st: string
    :return: dictionary
    """
    assert len(st) % 2 == 0, 'array order is not right'
    dc = {}
    for i in range(len(st) // 2):
        a = st[i * 2]
        b = st[i * 2 + 1]

        # convert dict value to float if possible
        try:
            val = float(b)
        except ValueError:
            val = b
        dc[a] = val

    return dc


def read_array(r, name, subdic=True):
    """
    read tcl array
    :param r: tkinter object
    :param name: array name
    :param subdic: group array in sub-dicts if possible?
    :return: dictionary
    """
    exe = r.tk.eval('array get ' + name)

    # extract dict components
    st = list(filter(None, re.split(' |({|})', exe)))

    # replace empty dicts by -1
    st_clean = []
    i = 0
    while i < len(st) - 1:
        if st[i] == '{' and st[i + 1] == '}':
            st_clean.append('-1')
            i += 1
        else:
            st_clean.append(st[i])
        i += 1
    if i == len(st) - 1:
        st_clean.append(st[-1])

    # check for sub-dicts
    limits = [i for i, s in enumerate(st_clean) if s == '{' or s == '}']
    assert len(limits) % 2 == 0, 'array order is not right'

    # one dictionary
    if len(limits) == 0:
        dc = get_dic(st_clean)

    # several sub-dictionaries
    else:
        dc = {}
        for a, b in zip(limits[::2], limits[1::2]):
            if not subdic:
                dc[st_clean[a - 1]] = st_clean[a + 1:b]
            else:
                dc[st_clean[a - 1]] = get_dic(st_clean[a + 1:b])
    return dc


def get_in_model_units(s_units, symbol, val):
    # equal units, no conversion
    if s_units == 'cm':
        return val

    # convert units
    else:
        sign = +1.0
        # if s_units == 'mm' and m_units == 'cm':
        #     sign = +1.0
        # elif s_units == 'cm' and m_units == 'mm':
        #     sign = -1.0
        # else:
        #     raise ValueError('Unknown unit combination ' + s_units + ' and ' + m_units)

        if symbol == 'R' or symbol == 'q':
            return val * np.power(10.0, sign * 4)
        elif symbol == 'C':
            return val * np.power(10.0, - sign * 4)
        elif symbol == 'P':
            return val * np.power(10.0, sign * 1)
        elif 'density' in symbol:
            return val * np.power(10.0, sign * 3)
        elif 'viscosity' in symbol:
            return val * np.power(10.0, sign * 1)
        else:
            raise ValueError('Unknown boundary condition symbol ' + symbol)


def get_flow_coronary(r):
    # read tcl array
    exe = r.tk.eval('array get sim_array')
    st = list(filter(None, re.split(' |({|})', exe)))
    st = [i for i in st if i != '{' and i != '}']

    # create new dict key for each pressure table
    pressures = defaultdict(list)
    name = ''
    for e in st:
        try:
            val = float(e)
            pressures[name] += [val]
        except ValueError:
            name = e

    # split arrays in time and pressure
    for k, v in pressures.items():
        pressures[k] = np.array(v).reshape(-1, 2)

    return pressures


def get_bcs(tcl, tcl_bc):
    # evaluate tcl-script to extract variables for boundary conditions
    r_bc = tkinter.Tk()
    r_bc.tk.eval('source ' + tcl_bc)

    # generate dictionaries from tcl output
    sim_bc = read_array(r_bc, 'sim_bc')
    sim_spid = read_array(r_bc, 'sim_spid')
    sim_preid = read_array(r_bc, 'sim_preid')
    sim_spname = read_array(r_bc, 'sim_spname', False)

    # get parameters
    params = get_params(tcl)

    # get boundary condition types
    sim_bc_type = get_bc_type(sim_bc)

    # convert boundary condition parameters to sim units
    for cp, bc in sim_bc.items():
        if is_outlet(cp):
            # add zero reference pressure if not defined
            if sim_bc_type[cp] in ('rcr', 'resistance') and 'Po' not in bc:
                sim_bc[cp]['Po'] = 0.0
            for p, v in bc.items():
                if p[0] in ('R', 'C'):
                    sim_bc[cp][p] = get_in_model_units(params['sim_units'], p[0], v)

    # extract pressure over time in case of coronary boundary conditions
    coronary = {}
    for bc in sim_bc.values():
        if 'COR' in bc.keys():
            coronary = get_flow_coronary(r_bc)
            for cp, v in coronary.items():
                coronary[cp][:, 1] = get_in_model_units(params['sim_units'], 'P', v[:, 1])

    # convert model parameters to sim units
    for p, v in params.items():
        if p in ('sim_density', 'sim_viscosity'):
            params[p] = get_in_model_units(params['sim_units'], p, float(v))

    bcs = {'bc': sim_bc, 'spid': sim_spid, 'preid': sim_preid, 'spname': sim_spname, 'coronary': coronary,
           'params': params, 'bc_type': sim_bc_type}

    # close window
    r_bc.destroy()

    return bcs


def get_bc_type(bc_def):
    bc_type = {}
    for s, bc in bc_def.items():
        if not is_outlet(s):
            continue
        if 'Rp' in bc and 'C' in bc and 'Rd' in bc:
            bc_type[s] = 'rcr'
        elif 'R' in bc and 'Po' in bc:
            bc_type[s] = 'resistance'
        elif 'COR' in bc:
            bc_type[s] = 'coronary'
        else:
            raise RuntimeError('boundary conditions not implemented')

    return bc_type


def is_outlet(s):
    if 'wall' in s or 'inflow' in s or 'stent' in s:
        return False
    else:
        return True


def get_params(tcl):
    # evaluate tcl-script to extract variables for general simulation parameters
    r = tkinter.Tk()
    r.tk.eval('source ' + tcl)
    params = {}
    for v in r.tk.eval('info globals').split():
        try:
            var = r.tk.getvar(v)

            # don't add tcl stuff to dict
            if isinstance(var, str) or isinstance(var, int) or isinstance(var, float):
                params[v] = var
            elif isinstance(var, tuple):
                params[v] = list(var)
            else:
                try:
                    params[v] = var.string
                except:
                    pass
        except:
            pass

    # close window
    r.destroy()

    return params
