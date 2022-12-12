# Post Processing for 1D Simulations

import pyvista as pv
import pickle
import os
import csv
import glob
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import trange
import re

def read_results_1d(res_dir):
    """
    Read results from oneDSolver and store in dictionary
    Args:
        res_dir: directory containing 1D results
        params_file: optional, path to dictionary of oneDSolver input parameters

    Returns:
    Dictionary sorted as [result field][segment id][time step]
    """
    # requested output fields
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']
    # read 1D simulation results
    results_1d = {}
    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(res_dir, '*branch*seg*_' + field + '.dat'))
        # loop segments
        results_1d[field] = defaultdict(dict)
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')
                # loop nodes
                results_1d_f = []
                for line in reader:
                    results_1d_f.append([float(l) for l in line if l][1:])
            # store results and GroupId
            seg = int(re.findall(r'\d+', f_res)[-1])
            branch = int(re.findall(r'\d+', f_res)[-2])
            results_1d[field][branch][seg] = np.array(results_1d_f)
    # read params file from pickled object in folder
    param_file = open(res_dir+os.sep+'params.pkl','rb')
    params = pickle.load(param_file)
    param_attributes = params.__dict__
    # read simulation parameters and add to result dict
    results_1d['params'] = param_attributes
    return results_1d

def make_points(x,y,z):
    """Helper to make XYZ points"""
    return np.column_stack((x, y, z))

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

def build_blank_result_model(model):
    min_seg_id = 0
    max_seg_id = int(max(model.cell_data['BranchId']))
    base = pv.PolyData()
    for id in range(min_seg_id,max_seg_id+1):
        pts     = model.cell_points(id)
        cell_id = np.argwhere(model.cell_data['BranchId']==id).flatten()
        num_ele = model.cell_data['1d_seg'][cell_id]
        t       = np.linspace(0,1,num=int(num_ele+1))
        branch_id   = np.ones(num_ele+1,dtype=int)*id
        line_points = pts[1,:].reshape(1,-1)*t.reshape(-1,1)+pts[0,:].reshape(1,-1)*(1-t.reshape(-1,1))
        line_points = make_points(line_points[:,0],line_points[:,1],line_points[:,2])
        line = lines_from_points(line_points)
        line.point_data['BranchId'] = branch_id
        base = base.merge(line)
    return base


def build_timepoint(model,results,timepoint):
    result_keys = list(results.keys())
    for res in result_keys:
        if res == 'params':
            continue
        branches = list(results[res].keys())
        blank    = np.zeros(model.n_points)
        model.point_data[res] = blank
        for branch in branches:
            segments = list(results[res][branch].keys())
            for seg in segments:
                pt_ids = np.argwhere(model.point_data['BranchId']==branch).flatten()
                model.point_data[res][pt_ids] = results[res][branch][seg][:,timepoint]
    return model

def build_result_vtp(res_dir):
    # Read results from folder into a dictionary object
    results = read_results_1d(res_dir)
    # Get name of mesh output file to append results to
    model_name = results['params']['mesh_output_file']
    num_time_steps = results['params']['num_time_steps']
    # Load model
    model = pv.read(res_dir+os.sep+model_name)
    # Build Blank Model
    blank = build_blank_result_model(model)
    # Make object folder
    outdir = res_dir+os.sep+'results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for t in trange(num_time_steps):
        tmp_result = build_timepoint(blank,results,t)
        tmp_result.save(outdir+os.sep+'results_{}.vtp'.format(t))
