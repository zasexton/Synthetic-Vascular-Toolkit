import svcco
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

cube = pv.Cube(x_length=3.72,y_length=3.72,z_length=3.72).triangulate().subdivide(3)
sphere = pv.Sphere(radius=2.3).triangulate().subdivide(1)

s = svcco.surface()
s.set_data(cube.points,cube.point_normals)
s.solve()
s.build()

t = svcco.tree()
t.set_boundary(s)
t.convex = True
t.set_root()
t.n_add(300)


heart = svcco.surface()
heart_points = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_points_unique.csv',delimiter=',')
heart_normals = np.genfromtxt('D:\\svcco\\svcco\\implicit\\tests\\heart_normals_unique.csv',delimiter=',')
heart.set_data(heart_points,heart_normals)
heart.solve()
heart.build(q=6,resolution=120,k=2,buffer=5)


left_gyrus   = "D:\\Tree\\Tree_8-0\\brain_testing\\FJ3801_BP58201_FMA72658_Left inferior frontal gyrus.obj"

gyrus = svcco.surface()
gyrus.load(left_gyrus)

gyrus.solve()
gyrus.build(q=4,resolution=40)

sph = svcco.surface()
sph.set_data(sphere.points,sphere.point_normals)
sph.solve()
sph.build(q=2,resolution=20)


###########################################
# Test function
###########################################

def test_cube(size=1000):
    t = svcco.tree()
    t.set_boundary(s)
    t.convex = True
    t.set_root()
    t.n_add(size)

    w = svcco.utils.fluid_analysis.wss.wss(t)
    r = t.data[:,21]*10 #should be microns
    l = t.data[:,20]*10

    a = []
    for i in range(t.data.shape[0]):
        if t.data[i,15] > 0:
            dau_1 = int(t.data[i,15].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_1,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_1,12:15]))
            if dot > 1:
                dot = 1
            ang_1 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_1)
        if t.data[i,16] > 0:
            dau_2 = int(t.data[i,16].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_2,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_2,12:15]))
            if dot > 1:
                dot = 1
            ang_2 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_2)
    a = np.array(a)

    branches = svcco.sv_interface.get_sv_data.get_branches(t.data)
    torts = []
    for branch in branches:
        tort = 0
        for idx in range(len(branch)-1):
            dot = np.dot(t.data[branch[idx],12:15],t.data[branch[idx+1],12:15].T)
            dot = dot/(np.linalg.norm(t.data[branch[idx],12:15])*np.linalg.norm(t.data[branch[idx+1],12:15]))
            if dot > 1:
                dot = 1
            ang = 180*(np.arccos(dot)/np.pi)
            tort += ang
        torts.append(tort)
    torts = np.array(torts)
    return w,r,l,a,torts

def test_heart(size=1000):
    t = svcco.tree()
    t.set_boundary(heart)
    t.set_root()
    t.n_add(size)

    w = svcco.utils.fluid_analysis.wss.wss(t)
    r = t.data[:,21]*10 #should be microns
    l = t.data[:,20]*10

    a = []
    for i in range(t.data.shape[0]):
        if t.data[i,15] > 0:
            dau_1 = int(t.data[i,15].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_1,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_1,12:15]))
            if dot > 1:
                dot = 1
            ang_1 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_1)
        if t.data[i,16] > 0:
            dau_2 = int(t.data[i,16].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_2,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_2,12:15]))
            if dot > 1:
                dot = 1
            ang_2 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_2)
    a = np.array(a)

    branches = svcco.sv_interface.get_sv_data.get_branches(t.data)
    torts = []
    for branch in branches:
        tort = 0
        for idx in range(len(branch)-1):
            dot = np.dot(t.data[branch[idx],12:15],t.data[branch[idx+1],12:15].T)
            dot = dot/(np.linalg.norm(t.data[branch[idx],12:15])*np.linalg.norm(t.data[branch[idx+1],12:15]))
            if dot > 1:
                dot = 1
            ang = 180*(np.arccos(dot)/np.pi)
            tort += ang
        torts.append(tort)
    torts = np.array(torts)
    return w,r,l,a,torts

def test_sphere(size=1000):
    t = svcco.tree()
    t.set_boundary(sph)
    t.convex = True
    t.set_root()
    t.n_add(size)

    w = svcco.utils.fluid_analysis.wss.wss(t)
    r = t.data[:,21]*10 #should be microns
    l = t.data[:,20]*10

    a = []
    for i in range(t.data.shape[0]):
        if t.data[i,15] > 0:
            dau_1 = int(t.data[i,15].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_1,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_1,12:15]))
            if dot > 1:
                dot = 1
            ang_1 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_1)
        if t.data[i,16] > 0:
            dau_2 = int(t.data[i,16].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_2,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_2,12:15]))
            if dot > 1:
                dot = 1
            ang_2 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_2)
    a = np.array(a)

    branches = svcco.sv_interface.get_sv_data.get_branches(t.data)
    torts = []
    for branch in branches:
        tort = 0
        for idx in range(len(branch)-1):
            dot = np.dot(t.data[branch[idx],12:15],t.data[branch[idx+1],12:15].T)
            dot = dot/(np.linalg.norm(t.data[branch[idx],12:15])*np.linalg.norm(t.data[branch[idx+1],12:15]))
            if dot > 1:
                dot = 1
            ang = 180*(np.arccos(dot)/np.pi)
            tort += ang
        torts.append(tort)
    torts = np.array(torts)
    return w,r,l,a,torts

def test_gyrus(size=1000):
    t = svcco.tree()
    t.set_boundary(gyrus)
    t.set_root()
    t.n_add(size)

    w = svcco.utils.fluid_analysis.wss.wss(t)
    r = t.data[:,21]*10 #should be microns
    l = t.data[:,20]*10

    a = []
    for i in range(t.data.shape[0]):
        if t.data[i,15] > 0:
            dau_1 = int(t.data[i,15].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_1,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_1,12:15]))
            if dot > 1:
                dot = 1
            ang_1 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_1)
        if t.data[i,16] > 0:
            dau_2 = int(t.data[i,16].item())
            dot   = np.dot(t.data[i,12:15],t.data[dau_2,12:15].T)
            dot   = dot/(np.linalg.norm(t.data[i,12:15])*np.linalg.norm(t.data[dau_2,12:15]))
            if dot > 1:
                dot = 1
            ang_2 = 180*(np.arccos(dot)/np.pi)
            a.append(ang_2)
    a = np.array(a)

    branches = svcco.sv_interface.get_sv_data.get_branches(t.data)
    torts = []
    for branch in branches:
        tort = 0
        for idx in range(len(branch)-1):
            dot = np.dot(t.data[branch[idx],12:15],t.data[branch[idx+1],12:15].T)
            dot = dot/(np.linalg.norm(t.data[branch[idx],12:15])*np.linalg.norm(t.data[branch[idx+1],12:15]))
            if dot > 1:
                dot = 1
            ang = 180*(np.arccos(dot)/np.pi)
            tort += ang
        torts.append(tort)
    torts = np.array(torts)
    return w,r,l,a,torts


def results(test_list,size=1000,iter=10,bins=50,range_w=(0,1500),range_r=(0,1),range_l=(0,3.5),range_a = (0,90),range_t=(0,4*180)):
    WSS_ALL = []
    RAD_ALL = []
    LEN_ALL = []
    ANG_ALL = []
    TORT_ALL = []
    WSS_ALL_MEAN = []
    RAD_ALL_MEAN = []
    LEN_ALL_MEAN = []
    ANG_ALL_MEAN = []
    TORT_ALL_MEAN = []
    WSS_ALL_STD  = []
    RAD_ALL_STD  = []
    LEN_ALL_STD  = []
    ANG_ALL_STD  = []
    TORT_ALL_STD = []
    WSS_WIDTH_ALL= []
    RAD_WIDTH_ALL= []
    LEN_WIDTH_ALL= []
    ANG_WIDTH_ALL= []
    TORT_WIDTH_ALL=[]
    WSS_CENTER_ALL = []
    RAD_CENTER_ALL = []
    LEN_CENTER_ALL = []
    ANG_CENTER_ALL = []
    TORT_CENTER_ALL = []
    for i in range(len(test_list)):
        WSS = []
        RAD = []
        LEN = []
        ANG = []
        TORT = []
        for j in range(iter):
            w,r,l,a,torts = test_list[i](size=size)
            freq_w,edges_w = np.histogram(w,bins=bins,range=range_w)
            freq_r,edges_r = np.histogram(r,bins=bins,range=range_r)
            freq_l,edges_l = np.histogram(l,bins=bins,range=range_l)
            freq_a,edges_a = np.histogram(a,bins=bins,range=range_a)
            freq_t,edges_t = np.histogram(torts,bins=bins,range=range_t)
            WSS.append(freq_w.tolist())
            RAD.append(freq_r.tolist())
            LEN.append(freq_l.tolist())
            ANG.append(freq_a.tolist())
            TORT.append(freq_t.tolist())
        WSS_ALL.append(WSS)
        RAD_ALL.append(RAD)
        LEN_ALL.append(LEN)
        ANG_ALL.append(ANG)
        TORT_ALL.append(TORT)
        WSS = np.array(WSS)
        RAD = np.array(RAD)
        LEN = np.array(LEN)
        ANG = np.array(ANG)
        #ANG = ANG[np.isfinite(ANG)]
        TORT = np.array(TORT)
        #TORT = TORT[np.isfinite(TORT)]
        WSS_mean = np.mean(WSS,axis=0)
        WSS_ALL_MEAN.append(WSS_mean)
        WSS_std  = np.std(WSS,axis=0)
        WSS_ALL_STD.append(WSS_std)
        RAD_mean = np.mean(RAD,axis=0)
        RAD_ALL_MEAN.append(RAD_mean)
        RAD_std  = np.std(RAD,axis=0)
        RAD_ALL_STD.append(RAD_std)
        LEN_mean = np.mean(LEN,axis=0)
        LEN_ALL_MEAN.append(LEN_mean)
        LEN_std  = np.std(LEN,axis=0)
        LEN_ALL_STD.append(LEN_std)
        ANG_mean = np.mean(ANG,axis=0)
        ANG_ALL_MEAN.append(ANG_mean)
        ANG_std  = np.std(ANG,axis=0)
        ANG_ALL_STD.append(ANG_std)
        TORT_mean = np.mean(TORT,axis=0)
        TORT_ALL_MEAN.append(TORT_mean)
        TORT_std  = np.std(TORT,axis=0)
        TORT_ALL_STD.append(TORT_std)

        width_w = (edges_w[1]-edges_w[0])
        WSS_WIDTH_ALL.append(width_w)
        width_r = (edges_r[1]-edges_r[0])
        RAD_WIDTH_ALL.append(width_r)
        width_l = (edges_l[1]-edges_l[0])
        LEN_WIDTH_ALL.append(width_l)
        width_a = (edges_a[1]-edges_a[0])
        ANG_WIDTH_ALL.append(width_a)
        width_t = (edges_t[1]-edges_t[0])
        TORT_WIDTH_ALL.append(width_t)
        centers_w = 0.5*(edges_w[1:]+edges_w[:-1])
        WSS_CENTER_ALL.append(centers_w)
        centers_r = 0.5*(edges_r[1:]+edges_r[:-1])
        RAD_CENTER_ALL.append(centers_r)
        centers_l = 0.5*(edges_l[1:]+edges_l[:-1])
        LEN_CENTER_ALL.append(centers_l)
        centers_a = 0.5*(edges_a[1:]+edges_a[:-1])
        ANG_CENTER_ALL.append(centers_a)
        centers_t = 0.5*(edges_t[1:]+edges_t[:-1])
        TORT_CENTER_ALL.append(centers_t)
    fig,ax = plt.subplots(4,len(test_list))
    for i in range(len(test_list)):
        centers_w = WSS_CENTER_ALL[i]
        WSS_mean  = WSS_ALL_MEAN[i]
        width_w   = WSS_WIDTH_ALL[i]
        WSS_std   = WSS_ALL_STD[i]

        centers_r = RAD_CENTER_ALL[i]
        RAD_mean  = RAD_ALL_MEAN[i]
        width_r   = RAD_WIDTH_ALL[i]
        RAD_std   = RAD_ALL_STD[i]

        centers_l = LEN_CENTER_ALL[i]
        LEN_mean  = LEN_ALL_MEAN[i]
        width_l   = LEN_WIDTH_ALL[i]
        LEN_std   = LEN_ALL_STD[i]

        centers_a = ANG_CENTER_ALL[i]
        ANG_mean  = ANG_ALL_MEAN[i]
        width_a   = ANG_WIDTH_ALL[i]
        ANG_std   = ANG_ALL_STD[i]

        centers_t = TORT_CENTER_ALL[i]
        TORT_mean = TORT_ALL_MEAN[i]
        width_t   = TORT_WIDTH_ALL[i]
        TORT_std  = TORT_ALL_STD[i]
        #ax[0][0].bar(centers_w,WSS_mean,width=width_w,yerr=WSS_std)
        if len(test_list) > 1:
            ax[0][i].bar(centers_r,RAD_mean,width=width_r,yerr=RAD_std)
            ax[1][i].bar(centers_l,LEN_mean,width=width_l,yerr=LEN_std)
            ax[2][i].bar(centers_a,ANG_mean,width=width_a,yerr=ANG_std)
            ax[3][i].bar(centers_t,TORT_mean,width=width_t,yerr=TORT_std)
            #for pos,y,err in zip(centers_w,WSS_mean,WSS_std):
            #    ax[0][0].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_r,RAD_mean,RAD_std):
                ax[0][i].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_l,LEN_mean,LEN_std):
                ax[1][i].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_a,ANG_mean,ANG_std):
                ax[2][i].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_t,TORT_mean,TORT_std):
                ax[3][i].errorbar(pos,y,err,capsize=2,color='black')
            ax[0][i].set_xlim([0,None])
            ax[0][i].set_ylim([0,None])

            ax[1][i].set_xlim([0,None])
            ax[1][i].set_ylim([0,None])

            ax[2][i].set_xlim([0,None])
            ax[2][i].set_ylim([0,None])

            ax[3][i].set_xlim([0,None])
            ax[3][i].set_ylim([0,None])
        else:
            ax[0].bar(centers_r,RAD_mean,width=width_r,yerr=RAD_std)
            ax[1].bar(centers_l,LEN_mean,width=width_l,yerr=LEN_std)
            ax[2].bar(centers_a,ANG_mean,width=width_a,yerr=ANG_std)
            ax[3].bar(centers_t,TORT_mean,width=width_t,yerr=TORT_std)
            #for pos,y,err in zip(centers_w,WSS_mean,WSS_std):
            #    ax[0][0].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_r,RAD_mean,RAD_std):
                ax[0].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_l,LEN_mean,LEN_std):
                ax[1].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_a,ANG_mean,ANG_std):
                ax[2].errorbar(pos,y,err,capsize=2,color='black')
            for pos,y,err in zip(centers_t,TORT_mean,TORT_std):
                ax[3].errorbar(pos,y,err,capsize=2,color='black')
            ax[0].set_xlim([0,None])
            ax[0].set_ylim([0,None])

            ax[1].set_xlim([0,None])
            ax[1].set_ylim([0,None])

            ax[2].set_xlim([0,None])
            ax[2].set_ylim([0,None])

            ax[3].set_xlim([0,None])
            ax[3].set_ylim([0,None])
    if len(test_list) > 1:
        ax[0][0].set_ylabel('Number of Vessels')
        ax[1][0].set_ylabel('Number of Vessels')
        ax[2][0].set_ylabel('Number of Vessels')
        ax[3][0].set_ylabel('Number of Vessels')
        for i in range(len(test_list)):
            ax[0][i].set_xlabel('Vessel Radius (mm)')
            ax[1][i].set_xlabel('Vessel Length (mm)')
            ax[2][i].set_xlabel('Parent-daughter Angles')
            ax[3][i].set_xlabel('Tortuosity')
    else:
        ax[0].set_ylabel('Number of Vessels')
        ax[1].set_ylabel('Number of Vessels')
        ax[2].set_ylabel('Number of Vessels')
        ax[3].set_ylabel('Number of Vessels')
        ax[0].set_xlabel('Vessel Radius (mm)')
        ax[1].set_xlabel('Vessel Length (mm)')
        ax[2].set_xlabel('Parent-daughter Angles')
        ax[3].set_xlabel('Tortuosity')
    plt.show()
