# Mesh searching versus kd-VVU patch searching
import svcco
import pyvista as pv
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import curve_fit
###########################################
# Code for Building Surface
###########################################

cube = pv.Cube().triangulate().subdivide(3)

s = svcco.surface()
s.set_data(10*cube.points,cube.point_normals)
s.solve()
s.build()

###########################################
#
###########################################

def test(surf,point_range=(1,100),iter=25):
    mesh = surf.tet.grid
    mesh_search = surf.cell_lookup
    kd   = surf.patch_KDTree
    times_mesh = []
    times_kd   = []
    xb = mesh.bounds[1] - mesh.bounds[0]
    yb = mesh.bounds[3] - mesh.bounds[2]
    zb = mesh.bounds[5] - mesh.bounds[4]
    mesh_search.query([0,0,0])
    kd.query([0,0,0])
    for i in range(iter):
        timeline_mesh = []
        timeline_kd   = []
        for j in trange(point_range[0],point_range[1]):
            total_time_mesh = 0
            total_time_kd   = 0
            #for k in range(j):
            xp = (mesh.bounds[0] + np.random.random(j)*xb)
            yp = (mesh.bounds[2] + np.random.random(j)*yb)
            zp = (mesh.bounds[4] + np.random.random(j)*zb)
            point = np.array([xp,yp,zp]).T
            start = perf_counter()
            mesh_search.query(point)
            total_time_mesh += perf_counter()-start
            start = perf_counter()
            kd.query(point)
            total_time_kd += perf_counter()-start
            timeline_mesh.append(total_time_mesh)
            timeline_kd.append(total_time_kd)
        times_mesh.append(timeline_mesh)
        times_kd.append(timeline_kd)
    times_mesh = np.array(times_mesh)
    times_kd   = np.array(times_kd)
    time_mesh_avg = np.mean(times_mesh,axis=0)
    time_mesh_std = np.std(times_mesh,axis=0)
    time_kd_avg = np.mean(times_kd,axis=0)
    time_kd_std = np.std(times_kd,axis=0)
    return time_mesh_avg,time_mesh_std,time_kd_avg,time_kd_std
#test(s)

def gen_figure(surf_object,point_range=(1,100),iter=25):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    TM,TM_STD,TP,TP_STD = test(surf_object,point_range=point_range,iter=iter)
    PTS = list(range(point_range[0],point_range[1]))
    ax.plot(PTS,TM,label="Mesh",color='red')
    ax.fill_between(PTS,TM-2*TM_STD,TM+TM_STD,alpha=0.5,color='red')
    ax.plot(PTS,TP,label="VVU",color='blue')
    ax.fill_between(PTS,TP-TP_STD,TP+TP_STD,alpha=0.5,color='blue')
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Time (seconds)")
    #ax.set_ylim(bottom=0)
    ax.set_xlim(left=0,right=point_range[1])
    ax.set_yscale('log')
    plt.legend()
    name = "mesh_vs_patches_point-range_{}-{}_iter_{}.svg".format(point_range[0],point_range[1],iter)
    fig.savefig(name,format="svg")
    plt.show()
    return TM,TM_STD,TP,TP_STD

def gen_figure_curve_fit(surf_object,point_range=(1,100),iter=25,func=None):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    TM,TM_STD,TP,TP_STD = test(surf_object,point_range=point_range,iter=iter)
    if func is None:
        def func(x,a,b,c,d,e):
            return  a*(np.log(b*x+c)**d)+e
    PTS = list(range(point_range[0],point_range[1]))
    ax.plot(PTS,TM,label="Mesh",color='red',linestyle='dashed')
    ax.fill_between(PTS,TM-2*TM_STD,TM+TM_STD,alpha=0.5,color='red')
    ax.plot(PTS,TP,label="VVU",color='blue',linestyle='dashed')
    ax.fill_between(PTS,TP-TP_STD,TP+TP_STD,alpha=0.5,color='blue')
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Time (seconds)")
    #ax.set_ylim(bottom=0)
    ax.set_xlim(left=0,right=point_range[1])
    ax.set_yscale('log')
    # curve fitting
    PTS = np.array(PTS)
    popt_mesh,pcov_mesh = curve_fit(func,PTS,TM,p0=[1,1,1,1,1],maxfev=50000)
    popt_pat,pcov_pat = curve_fit(func,PTS,TP,p0=[1,1,1,1,1],maxfev=50000)
    ax.plot(PTS,func(PTS,popt_mesh[0],popt_mesh[1],popt_mesh[2],popt_mesh[3],popt_mesh[4]),color='red',linestyle='solid')
    ax.plot(PTS,func(PTS,popt_pat[0],popt_pat[1],popt_pat[2],popt_pat[3],popt_pat[4]),color='blue',linestyle='solid')
    plt.legend()
    name = "mesh_vs_patches_point-range_{}-{}_iter_{}_with_curve.svg".format(point_range[0],point_range[1],iter)
    fig.savefig(name,format="svg")
    plt.show()
    return TM,TM_STD,TP,TP_STD,popt_mesh,pcov_mesh,popt_pat,pcov_pat
"""
def results(start=3,end=6):
    POINTS = []
    MESH_AVG   = []
    MESH_STD   = []
    KD_AVG     = []
    KD_STD     = []
    for i in range(start,end):
        print("{}/{}".format(i,end-1))
        cube = pv.Cube().triangulate().subdivide(i)
        s = svcco.surface()
        s.set_data(10*cube.points,normals=cube.point_normals)
        s.solve()
        s.build()
        POINTS.append(s.points.shape[0])
        m_avg,m_std,kd_avg,kd_std = test(s)
        MESH_AVG.append(m_avg)
        MESH_STD.append(m_std)
        KD_AVG.append(kd_avg)
        KD_STD.append(kd_std)
    MESH_AVG = np.array(MESH_AVG)
    MESH_STD = np.array(MESH_STD)
    KD_AVG   = np.array(KD_AVG)
    KD_STD   = np.array(KD_STD)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(POINTS,MESH_AVG,label="Mesh")
    ax.fill_between(POINTS,MESH_AVG-MESH_STD,MESH_AVG+MESH_STD,alpha=0.5)
    ax.plot(POINTS,KD_AVG,label="kd-VVU")
    ax.fill_between(POINTS,KD_AVG-KD_STD,KD_AVG+KD_STD,alpha=0.5)
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Time (seconds)")
    ax.set_ylim(bottom=0)
    plt.legend()
    plt.show()
"""
