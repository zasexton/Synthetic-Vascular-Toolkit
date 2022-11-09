from tqdm import tqdm
from .weingarten_map import *
import numpy as np

def sampling(points,angle=0,max_local_size=20,min_local_size=10,normals=None,l=0.5,quiet=True):
    """
    """
    # want to sample points that have high absolute
    # gaussian curvature because the local behavior
    # deviates from a parabolic shape thus more
    # patches will be needed to better approximate.
    curvature,idx,KDT = estimate_gaussian_curvature(points,quiet=quiet)
    abs_curvature = abs(curvature)
    sorted_abs_curvature_idx = abs_curvature.flatten().argsort().tolist()
    patches = []
    patch_idx = []
    centers = []
    total_length = len(sorted_abs_curvature_idx)
    dist, idx = KDT.query(points,k=max_local_size+1)
    if quiet:
        while len(sorted_abs_curvature_idx) > 0:
            center = sorted_abs_curvature_idx[0]
            count = 0
            patch_idxs = []
            for k in idx[center,:]:
                #if abs(abs_curvature[center] - abs_curvature[k]) > 0.1:
                #    pass
                if np.isclose(np.dot(normals[center],normals[k]),1) and np.all(points[center]==points[k]):
                    patch_idxs.append(count)
                elif np.dot(normals[center],normals[k]) <= angle or abs(abs_curvature[center] - abs_curvature[k]) > 0.2:
                    if len(patch_idxs) > min_local_size:
                        break
                else:
                    patch_idxs.append(count)
                #if np.dot(normals[center],normals[k]) <= angle or abs(abs_curvature[center] - abs_curvature[k]) > 0.2:#np.all(points[center]==points[k]):
                #    if len(patch_idxs) > min_local_size:
                #        break
                #else:
                #    patch_idxs.append(count)
                #    #pass
                count += 1
            #limit = count//3
            if len(patch_idxs) < round(min_local_size*0.5):
                limit = len(patch_idxs)
            else:
                limit = round(len(patch_idxs)*l)
            #if len(patch_idxs) >= min_local_size:
            #    #if limit == 0:
            #    #       limit += 1
            #    #if limit >= 1:
            #    patches.append(points[idx[center,patch_idxs],:])
            #    patch_idx.append(idx[center,patch_idxs])
            #    centers.append(center)
            #    for i in idx[center,patch_idxs]:
            #        if i in sorted_abs_curvature_idx:
            #            sorted_abs_curvature_idx.remove(i)
            #else:
            #    #if idx[center,patch_idxs[0]] in sorted_abs_curvature_idx:
            #    #    sorted_abs_curvature_idx.remove(idx[center,patch_idxs[0]])
            #    sorted_abs_curvature_idx.remove(center)
            patches.append(points[idx[center,patch_idxs],:])
            patch_idx.append(idx[center,patch_idxs])
            centers.append(center)
            #print(points[idx[center,patch_idxs],:])
            for i in idx[center,patch_idxs[:limit]]:

                if i in sorted_abs_curvature_idx:
                    sorted_abs_curvature_idx.remove(i)
            #if limit-1 > 0:
            #    pbar.update(limit-1)
        return patches,patch_idx,KDT,centers
    with tqdm(total=total_length,desc='Determining Curvature  ') as pbar:
        while len(sorted_abs_curvature_idx) > 0:
            center = sorted_abs_curvature_idx[0]
            count = 0
            patch_idxs = []
            for k in idx[center,:]:
                #if abs(abs_curvature[center] - abs_curvature[k]) > 0.1:
                if np.dot(normals[center],normals[k]) <= angle:
                    if count > min_local_size:
                        break
                else:
                    patch_idxs.append(count)
                    #pass
                count += 1
            #limit = count//3
            limit = round(len(patch_idxs)*l)
            if len(patch_idxs) >= min_local_size:
                #if limit == 0:
                #	limit += 1
                #if limit >= 1:
                patches.append(points[idx[center,patch_idxs],:])
                patch_idx.append(idx[center,patch_idxs])
                centers.append(center)
                for i in idx[center,patch_idxs]:
                    if i in sorted_abs_curvature_idx:
                        sorted_abs_curvature_idx.remove(i)
                        pbar.update(1)
            else:
                #if idx[center,patch_idxs[0]] in sorted_abs_curvature_idx:
                #    sorted_abs_curvature_idx.remove(idx[center,patch_idxs[0]])
                sorted_abs_curvature_idx.remove(center)
                pbar.update(1)
            #if limit-1 > 0:
            #    pbar.update(limit-1)
    return patches,patch_idx,KDT,centers
