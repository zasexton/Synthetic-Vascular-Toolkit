import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from tqdm import tqdm

def generate_bases(points,quiet=True):
    n_points = points.shape[0]
    dim = points.shape[1]
    int_dim = dim - 1
    tangent_basis = np.zeros((dim,int_dim,n_points))
    normal_basis = np.zeros((dim,dim-int_dim,n_points))
    K = int(max(20, dim*np.log10(n_points)))
    KDT = spatial.KDTree(points)
    dist, idx = KDT.query(points,k=K+1)
    if quiet:
        for i in range(n_points):
            tmp = points[idx[i,:],:]
            tmp = tmp - np.ones((K+1,1))@tmp[0,:].reshape(1,-1)
            tmp = tmp[1:,:]
            U, S, H = np.linalg.svd(tmp)
            H = H.T
            diag_idx = S.argsort()[::-1]
            sort_diag = np.sort(S)[::-1]
            H = H[:,diag_idx]
            tangent_basis[:,:,i] = H[:,:int_dim]
            normal_basis[:,:,i] = H[:,dim-1].reshape(-1,1)
    else:
        for i in tqdm(range(n_points),desc='Generating bases       '):
    	    tmp = points[idx[i,:],:]
    	    tmp = tmp - np.ones((K+1,1))@tmp[0,:].reshape(1,-1)
    	    tmp = tmp[1:,:]
    	    U, S, H = np.linalg.svd(tmp)
    	    H = H.T
    	    diag_idx = S.argsort()[::-1]
    	    sort_diag = np.sort(S)[::-1]
    	    H = H[:,diag_idx]
    	    tangent_basis[:,:,i] = H[:,:int_dim]
    	    normal_basis[:,:,i] = H[:,dim-1].reshape(-1,1)
    return tangent_basis,normal_basis,idx,KDT

def estimate_weingarten_map(points,quiet=True):
    n_points = points.shape[0]
    dim = points.shape[1]
    tangent_basis, normal_basis, idx, KDT = generate_bases(points,quiet=quiet)
    tangent_dim = tangent_basis.shape[1]
    normal_dim = normal_basis.shape[1]
    weingarten_map = np.zeros((tangent_dim,tangent_dim,
                               normal_dim,n_points))
    if quiet:
        for i in range(n_points):
            for j in range(normal_dim):
                tmp_point = points[i,:]
                tmp_tangent_basis = tangent_basis[:,:,i]
                tmp_normal_basis = normal_basis[:,j,i]
                tmp_neighborhood = idx[i,:]
                tmp_neighborhood_size = len(tmp_neighborhood)
                tmp_local_normals = normal_basis[:,:,tmp_neighborhood]
                tmp_normal_extension = np.zeros((tmp_neighborhood_size,dim))
                for k in range(tmp_neighborhood_size):
                    projection = tmp_local_normals[:,:,k]@\
                                 tmp_local_normals[:,:,k].T@\
                                 tmp_normal_basis.reshape(-1,1)
                    tmp_normal_extension[k,:] = projection.T
                tmp_diff_normal = np.zeros((tmp_neighborhood_size-1,dim))
                tmp_diff_position = np.zeros((tmp_neighborhood_size-1,dim))
                for k in range(tmp_neighborhood_size-1):
                    tmp_diff_normal[k,:] = tmp_normal_extension[k+1,:] -\
                                           tmp_normal_basis.reshape(1,-1)
                    tmp_diff_position[k,:] = points[tmp_neighborhood[k+1],:] -\
                                             tmp_point
                tmp_normal_projection = tmp_diff_normal@tmp_tangent_basis
                tmp_position_projection = tmp_diff_position@tmp_tangent_basis
                A = -(np.linalg.inv(tmp_position_projection.T@\
                                    tmp_position_projection))@\
                                    tmp_position_projection.T@\
                                    tmp_normal_projection
                weingarten_map[:,:,j,i] = (1/2) * (A+A.T)
    else:
        for i in tqdm(range(n_points),desc='Building Weingarten Map'):
            for j in range(normal_dim):
                tmp_point = points[i,:]
                tmp_tangent_basis = tangent_basis[:,:,i]
                tmp_normal_basis = normal_basis[:,j,i]
                tmp_neighborhood = idx[i,:]
                tmp_neighborhood_size = len(tmp_neighborhood)
                tmp_local_normals = normal_basis[:,:,tmp_neighborhood]
                tmp_normal_extension = np.zeros((tmp_neighborhood_size,dim))
                for k in range(tmp_neighborhood_size):
                    projection = tmp_local_normals[:,:,k]@\
                                 tmp_local_normals[:,:,k].T@\
                                 tmp_normal_basis.reshape(-1,1)
                    tmp_normal_extension[k,:] = projection.T
                tmp_diff_normal = np.zeros((tmp_neighborhood_size-1,dim))
                tmp_diff_position = np.zeros((tmp_neighborhood_size-1,dim))
                for k in range(tmp_neighborhood_size-1):
                    tmp_diff_normal[k,:] = tmp_normal_extension[k+1,:] -\
                                           tmp_normal_basis.reshape(1,-1)
                    tmp_diff_position[k,:] = points[tmp_neighborhood[k+1],:] -\
                                              tmp_point
                tmp_normal_projection = tmp_diff_normal@tmp_tangent_basis
                tmp_position_projection = tmp_diff_position@tmp_tangent_basis
                A = -(np.linalg.inv(tmp_position_projection.T@\
                                    tmp_position_projection))@\
                                    tmp_position_projection.T@\
                                    tmp_normal_projection
                weingarten_map[:,:,j,i] = (1/2) * (A+A.T)
    return weingarten_map, idx, KDT

def estimate_gaussian_curvature(points,quiet=True):
    n_points = points.shape[0]
    guassian_curvature = np.zeros((n_points,1))
    w_map, idx, KDT = estimate_weingarten_map(points,quiet=quiet)
    for i in range(n_points):
        guassian_curvature[i] = np.linalg.det(w_map[:,:,0,i])
    return guassian_curvature, idx, KDT

def show_gaussian_curvature(points):
    guassian_curvature, idx, KDT = estimate_gaussian_curvature(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    coolwarm = cm.get_cmap('coolwarm',1000)
    gc_min = min(guassian_curvature)
    gc_max = max(guassian_curvature)
    colors = coolwarm(guassian_curvature)
    cax = ax.scatter3D(points[:,0],points[:,1],points[:,2],cmap=coolwarm,c=guassian_curvature)
    cbar = fig.colorbar(cax,ax=ax,ticks=[gc_min,gc_max])
    return fig,ax
"""
def bumpy_sphere(samples=10,scale=5,a=3,b=1):
    data = np.zeros((samples**2,3))
    theta = np.linspace(0,2*np.pi,num=samples)
    phi = np.linspace(0,np.pi,num=samples)
    count = 0
    for t in theta:
        for p in phi:          
            r = scale + np.cos(a*t)*np.sin(b*p)
            data[count,0] = r*np.cos(t)*np.sin(p)
            data[count,1] = r*np.sin(t)*np.sin(p)
            data[count,2] = r*np.cos(p)
            count += 1
    final_data = []
    for i in range(data.shape[0]):
        if i == 0:
            final_data.append(data[i,:])
            continue
        add = True
        for j in range(i):
            if np.all(np.isclose(data[i,:],data[j,:])):
                add = False
        if add:
            final_data.append(data[i,:])
    return np.array(final_data)
"""
