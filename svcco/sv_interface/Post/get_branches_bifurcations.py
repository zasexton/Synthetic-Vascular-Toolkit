#!/usr/bin/env python

import pdb
import vtk

import scipy
import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, input_args
from vtk_functions import read_geo, write_geo, collect_arrays, threshold, geo, clean, region_grow, region_grow_simultaneous, ClosestPoints

from collections import defaultdict


def color_clip(poly, name_this, name_other):
    # label array of this name
    color_this = vtk.vtkDoubleArray()
    color_this.SetNumberOfValues(poly.GetNumberOfCells())
    color_this.SetName(name_this)

    # get cell labels from connected points (assume they all have the same label)
    pids = vtk.vtkIdList()
    for i in range(poly.GetNumberOfCells()):
        poly.GetCellPoints(i, pids)
        color_this.SetValue(i, poly.GetPointData().GetArray(name_this).GetValue(pids.GetId(0)))

    # label array of other name (constant -1)
    color_other = vtk.vtkDoubleArray()
    color_other.SetNumberOfValues(poly.GetNumberOfCells())
    color_other.SetName(name_other)
    color_other.Fill(-1)

    poly.GetCellData().AddArray(color_this)
    poly.GetCellData().AddArray(color_other)


def get_bifurcations(cent):
    # read all point arrays
    arr_cent = collect_arrays(cent.GetPointData())

    # all bifurcation ids
    bifurcation_ids = np.unique(arr_cent['BifurcationId']).tolist()
    bifurcation_ids.remove(-1)

    # collect point and cell ids
    pids = vtk.vtkIdList()
    cids = vtk.vtkIdList()

    # collect points surrounding each bifurcation
    bifurcations = {}
    for bf in bifurcation_ids:
        bifurcations[bf] = defaultdict(list)
        for i in range(cent.GetNumberOfPoints()):
            if arr_cent['BifurcationId'][i] == bf:
                cent.GetPointCells(i, cids)
                for j in range(cids.GetNumberOfIds()):
                    cent.GetCellPoints(cids.GetId(j), pids)
                    for k in range(pids.GetNumberOfIds()):
                        br_id = arr_cent['BranchId'][pids.GetId(k)]
                        if br_id != -1:
                            bifurcations[bf]['branches'] += [br_id]
                            bifurcations[bf]['points'] += [pids.GetId(k)]
    return bifurcations


def split_geo(fpath_surf, fpath_cent, fpath_sect, fpath_vol):
    surf = read_geo(fpath_surf).GetOutput()
    cent = read_geo(fpath_cent).GetOutput()
    sect = read_geo(fpath_sect).GetOutput()
    vol = read_geo(fpath_vol).GetOutput()

    arr_surf = collect_arrays(surf.GetPointData())
    arr_cent = collect_arrays(cent.GetPointData())

    bifurcation_ids = np.unique(arr_cent['BifurcationId']).tolist()
    bifurcation_ids.remove(-1)
    branch_ids = np.unique(arr_cent['BranchId']).tolist()
    branch_ids.remove(-1)

    pids = vtk.vtkIdList()
    cids = vtk.vtkIdList()

    # get centerline connectivity
    bifurcations = get_bifurcations(cent)

    branch_ids = arr_surf['BranchId']

    cp = ClosestPoints(surf)

    cut_name = 'distance_bifurcation'

    # distance array used for clipping
    distance = -1 * np.ones(surf.GetNumberOfPoints())

    rings = defaultdict(dict)
    # loop bifurcations
    for bf, bifurcation in bifurcations.items():
        
        # loop attached branches
        # seed_bf = []
        # ring_bf = []
        for i, (p, br) in enumerate(zip(bifurcation['points_global'], bifurcation['branches'])):
            # print(bf, br)

            # pick slice separating bifurcation and branch
            sliced = threshold(sect, p, 'GlobalNodeId').GetOutput()

            # signed distance from slice
            dist = vtk.vtkDistancePolyDataFilter()
            dist.SetInputData(0, surf)
            dist.SetInputData(1, geo(sliced))
            dist.Update()
            dist = v2n(dist.GetOutput().GetPointData().GetArray('Distance'))

            # reverse inlet
            if i > 0:
                dist *= -1

            # pick slice separating bifurcation and branch
            sliced = threshold(sect, p, 'GlobalNodeId').GetOutput()

            # mark bifurcation cuts
            ring = cp.search(v2n(sliced.GetPoints().GetData()))
            # ring_bf += ring

            rings[bf][br] = ring

            # ring_grow_1 = region_grow(surf, ring, distance, 2)
            # ring_grow_2 = region_grow(surf, ring, distance, 3)
            #
            # indicator_1 = np.intersect1d(ring_grow_1, np.where(branch_ids == br))
            # indicator_2 = np.intersect1d(ring_grow_2, np.where(branch_ids == br))

            # bifurcation_id[indicator_1] = 1

            # ring_diff = list(set(indicator_2) - set(indicator_1))
            # seed_bf += np.array(ring_diff)[dist[ring_diff] > 0].tolist()

            # assert np.unique(distance[indicator]).shape[0] == 1, 'overwriting branches'

            # branch_dist[indicator_2] = dist[indicator_2]

        # mark region inside bifurcation
        # bf_ids = region_grow(surf, seed_bf, bifurcation_id)
    bifurcation_id = -1 * np.ones(surf.GetNumberOfPoints())
    branch_dist = np.zeros(surf.GetNumberOfPoints())
    region_grow_simultaneous(surf, rings, bifurcation_id, branch_dist)

    # distance[bf_ids] = 1
    # indicator = branch_dist != -1
    # distance[indicator] = branch_dist[indicator]




    # assemble cutting array
    out = vtk.vtkDoubleArray()
    out.SetNumberOfValues(surf.GetNumberOfPoints())
    out.SetName('branch_dist')
    for i in range(surf.GetNumberOfPoints()):
        out.SetValue(i, branch_dist[i])
    surf.GetPointData().AddArray(out)
    out = vtk.vtkDoubleArray()
    out.SetNumberOfValues(surf.GetNumberOfPoints())
    out.SetName('BifurcationIdTmp')
    for i in range(surf.GetNumberOfPoints()):
        out.SetValue(i, bifurcation_id[i])
    surf.GetPointData().AddArray(out)

    append = vtk.vtkAppendFilter()
    append.AddInputData(surf)
    append.MergePointsOn()
    append.Update()
    return append.GetOutput()

    # assemble cutting array
    out = vtk.vtkDoubleArray()
    out.SetNumberOfValues(surf.GetNumberOfPoints())
    out.SetName(cut_name)
    for i in range(surf.GetNumberOfPoints()):
        out.SetValue(i, distance[i])

    surf.GetPointData().AddArray(out)
    surf.GetPointData().SetActiveScalars(cut_name)

    # clip into branches and bifurcations
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(surf)
    clip.SetValue(0.0)
    clip.GenerateClippedOutputOn()
    clip.GenerateClipScalarsOff()
    clip.Update()

    # color branches and bifurcations
    poly_bf = clip.GetOutput(0)
    poly_br = clip.GetOutput(1)
    color_clip(poly_bf, 'BifurcationId', 'BranchId')
    color_clip(poly_br, 'BranchId', 'BifurcationId')

    # add back together
    append = vtk.vtkAppendFilter()
    append.AddInputData(poly_br)
    append.AddInputData(poly_bf)
    append.MergePointsOn()
    append.Update()

    # remove outdated point arrays
    unstruc_out = append.GetOutput()
    # unstruc_out.GetPointData().RemoveArray(cut_name)
    # unstruc_out.GetPointData().RemoveArray('BranchId')
    # unstruc_out.GetPointData().RemoveArray('BifurcationId')

    unstruc_out.GetCellData().SetActiveScalars('BranchId')

    return unstruc_out


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)
        fpath_cent = db.get_centerline_path(geo)
        fpath_surf = db.get_surfaces_grouped_path(geo)
        fpath_sect = db.get_section_path(geo)
        fpath_vol = db.get_volume(geo)

        # try:
        surf_cut = split_geo(fpath_surf, fpath_cent, fpath_sect, fpath_vol)
        # except Exception as e:
        #     print(e)
        #     continue

        write_geo(db.get_surfaces_cut_path(geo), surf_cut)


if __name__ == '__main__':
    descr = 'Split geometry in branches and bifurcation_ids'
    d, g, _ = input_args(descr)
    main(d, g)
