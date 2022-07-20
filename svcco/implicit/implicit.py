from __future__ import print_function
import numba as nb
import vtk
from vtk.util import numpy_support
from scipy import optimize
from .visualize.visualize import *
from .sampling import *
from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import as_completed
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from threading import RLock as TRLock
from .derivative import *
from tqdm.auto import tqdm, trange
from functools import partial
from .solver.solver import solver
from ..branch_addition.basis import tangent_basis
from pickle import dumps,loads
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import splprep,splev
#from gekko import GEKKO
import marshal, types

import pyvista as pv
import tetgen
import time

import imp
import sys

from .core.m_matrix import M
from .core.n_matrix import N
from .core.a_matrix import A
from .core.h_matrix import H
from .load import load3d
from .visualize.visualize import show_mesh

class patch:
    def __init__(self,points,normals):
        self.ndim = points.shape[0]
        self.ddim = points.shape[1]
        self.points = points
        self.normals = normals
        self.A_inv,self.K00,self.K01,self.K11 = A(points)
        self.H_0 = H(self.K00,self.K01,self.K11,0)

    def solve(self,local_method='L-BFGS-B',regularize=False,
              local_verbosity=False,variational=False,
              solver_method='Bounded',solver_verbosity=False):
        if regularize:
            s = solver(self.points,self.normals)
            a,b = s.solve(local_verbosity=local_verbosity,local_method=local_method,
                          variational=variational,solver_method=solver_method,
                          solver_verbosity=solver_verbosity)
            self.a = a
            self.b = b
        else:
            g = np.zeros(self.ndim*3)
            M_inv = self.A_inv[:self.ndim*4,:self.ndim*4]
            N_inv = self.A_inv[:self.ndim*4,self.ndim*4:]
            for i in range(self.ndim):
                g[i] = self.normals[i,0]
                g[i+self.ndim] = self.normals[i,1]
                g[i+2*self.ndim] = self.normals[i,2]
            self.g = g
            s = np.zeros(self.ndim)
            l_side = np.zeros(M_inv.shape[1])
            l_side[:len(s)] = s
            l_side[len(s):(len(s)+len(g))] = g
            a = np.matmul(M_inv,l_side)
            b = np.matmul(N_inv.T,l_side)
            self.a = a
            self.b = b

class surface:
    def __init__(self):
        self.subtracted_volumes = []
        self.added_volumes = []
        self.polydata = None
        pass
    def load(self,filename,subdivisions=0):
        points,normals,polydata = load3d(filename,subdivisions)
        self.set_data(points,normals=normals)
        self.polydata = polydata
    def set_data(self,points,normals=None,workers=1,local_min=10,local_max=20,l=0.5,PCA_samplesize=25):
        self.points = points
        self.ddim = points.shape[1]
        self.dim_range = [0]*self.ddim*2
        for i in range(self.ddim):
        	self.dim_range[i*2]   = min(points[:,i])
        	self.dim_range[i*2+1] = max(points[:,i])
        self.local_min = local_min
        self.local_max = local_max
        self.l = l
        self.workers = workers
        if np.any(normals) is None:
            point_cloud_vtk_array  = numpy_support.numpy_to_vtk(points,deep=True)
            point_cloud_vtk_points = vtk.vtkPoints()
            point_cloud_vtk_points.SetData(point_cloud_vtk_array)
            point_cloud_polydata   = vtk.vtkPolyData()
            point_cloud_polydata.SetPoints(point_cloud_vtk_points)
            PCA_normals            = vtk.vtkPCANormalEstimation()
            PCA_normals.SetInputData(point_cloud_polydata)
            PCA_normals.SetSampleSize(PCA_samplesize)
            PCA_normals.SetNormalOrientationToGraphTraversal()
            PCA_normals.Update()
            PCA_polydata = PCA_normals.GetOutput()
            PCA_vtk_pts  = PCA_polydata.GetPointData()
            vtk_normals  = PCA_vtk_pts.GetNormals()
            self.normals = numpy_support.vtk_to_numpy(vtk_normals)
        else:
            self.normals = normals

    @staticmethod
    def patch_constructor_progress(tuple_data, auto_position=False,
                                   write_safe=False, blocking=True,
                                   progress=True,regularize=False,
                                   local_verbosity=False,local_method='L-BFGS-B',
                                   variational=False,solver_method='Bounded',
                                   solver_verbosity=True):
        points = tuple_data[0]
        normals = tuple_data[1]
        chunk = tuple_data[2]
        patches = []
        total = len(points)
        text = "Processing Chunk {0}  : ".format(chunk)
        for i in trange(total, desc=text, leave=False, disable=not progress,
                        lock_args=None if blocking else (False,),
                        position=None if auto_position else chunk):
            p_tmp = patch(points[i],normals[i])
            p_tmp.solve(regularize=regularize,local_verbosity=local_verbosity,
                        local_method=local_method,variational=varitaional,
                        solver_method=solver_method,solver_verbosity=solver_verbosity)
            patches.append(p_tmp)
        return patches

    @staticmethod
    def patch_constructor_no_progress(tuple_data, auto_position=False,
                                      write_safe=False, blocking=True,
                                      progress=True,regularize=False,
                                      local_verbosity=False,local_method='L-BFGS-B',
                                      variational=False,solver_method='Bounded',
                                      solver_verbosity=True):
        points = tuple_data[0]
        normals = tuple_data[1]
        chunk = tuple_data[2]
        patches = []
        total = len(points)
        for i in range(total):
            p_tmp = patch(points[i],normals[i])
            p_tmp.solve(regularize=regularize,local_verbosity=local_verbosity,
                        local_method=local_method,variational=variational,
                        solver_method=solver_method,solver_verbosity=solver_verbosity)
            patches.append(p_tmp)
        return patches

    def parallel_build(self,chunks,show_individual=False,PU=True):
        patches = []
        if show_individual and PU:
            freeze_support()
            tqdm.set_lock(RLock())
            executor = PPE(max_workers=len(chunks),initializer=tqdm.set_lock,initargs=(tqdm.get_lock(),))
            patch_function = partial(self.patch_constructor_no_progress,regularize=self.__regularize__,
                                     local_verbosity=self.__local_verbosity__,local_method=self.__local_method__,
                                     variational=self.__variational__,solver_method=self.__solver_method__,
                                     solver_verbosity=self.__solver_verbosity__)
            futures = {executor.submit(patch_function,chunk): chunk for chunk in chunks}
        elif self.workers == 1 and PU:
            patch_function = partial(self.patch_constructor_no_progress,regularize=self.__regularize__,
                                     local_verbosity=self.__local_verbosity__,local_method=self.__local_method__,
                                     variational=self.__variational__,solver_method=self.__solver_method__,
                                     solver_verbosity=self.__solver_verbosity__)
            futures = patch_function(chunks[0])
        elif PU:
            executor = PPE(max_workers=len(chunks))
            patch_function = partial(self.patch_constructor_no_progress,regularize=self.__regularize__,
                                     local_verbosity=self.__local_verbosity__,local_method=self.__local_method__,
                                     variational=self.__variational__,solver_method=self.__solver_method__,
                                     solver_verbosity=self.__solver_verbosity__)
            futures = {executor.submit(patch_function,chunk): chunk for chunk in chunks}
        else:
            global_patch = patch(self.points,self.normals)
            global_patch.solve(local_method=self.__local_method__,regularize=self.__regularize__,
                               local_verbosity=self.__local_verbosity__,variational=self.__variational__,
                               solver_method=self.__solver_method__,solver_verbosity=self.__solver_verbosity__)
        total_patches = []
        if show_individual and PU:
            for future in as_completed(futures):
                 total_patches.extend(future.result())
        elif self.workers == 1 and PU:
            total_patches = futures
        elif PU:
            for future in as_completed(futures):
                total_patches.extend(future.result())
        else:
            total_patches.append(global_patch)
        if self.workers != 1:
            executor.shutdown()
        return total_patches

    def solve(self,angle=0,PU=True,show_individual=False,regularize=False,
              local_verbosity=False,local_method='L-BFGS-B',variational=False,
              solver_method='Bounded',solver_verbosity=False,quiet=True):
        self.__regularize__ = regularize
        self.__local_verbosity__ = local_verbosity
        self.__local_method__ = local_method
        self.__variational__ = variational
        self.__solver_method__ = solver_method
        self.__solver_verbosity__= solver_verbosity
        if PU:
            patches,idxs,KDTree,centers = sampling(self.points,
                                           normals=self.normals,
                                           min_local_size=self.local_min,
                                           max_local_size=self.local_max,l=self.l,
                                           angle=angle,quiet=quiet)
            patches = []
            patch_centers = []
            patch_center_idx = []
            self.max_patch_size = 0
            chunksize = len(idxs)//self.workers
            number_of_chunks = len(idxs)//chunksize
            chunks = []
            for i in range(number_of_chunks):
                tmp_points = []
                tmp_normals = []
                if i < number_of_chunks-1:
                    if quiet:
                        for j in range(i*chunksize,i*chunksize+chunksize):
                            tmp_points.append(self.points[idxs[j]])
                            tmp_normals.append(self.normals[idxs[j]])
                            if len(idxs[j]) > self.max_patch_size:
                                self.max_patch_size = len(idxs[j])
                    else:
                        for j in tqdm(range(i*chunksize,i*chunksize+chunksize),desc='Building Chunk {}       '.format(i)):
                            tmp_points.append(self.points[idxs[j]])
                            tmp_normals.append(self.normals[idxs[j]])
                            if len(idxs[j]) > self.max_patch_size:
                                self.max_patch_size = len(idxs[j])
                    chunk_chunk = i
                    chunks.append([tmp_points,tmp_normals,chunk_chunk])
                else:
                    if i*chunksize >= len(idxs):
                        break
                    else:
                        if quiet:
                            for j in range(i*chunksize,len(idxs)):
                                tmp_points.append(self.points[idxs[j]])
                                tmp_normals.append(self.normals[idxs[j]])
                                if len(idxs[j]) > self.max_patch_size:
                                    self.max_patch_size = len(idxs[j])
                        else:
                            for j in tqdm(range(i*chunksize,len(idxs)),desc='Building Chunk {}       '.format(i)):
                                tmp_points.append(self.points[idxs[j]])
                                tmp_normals.append(self.normals[idxs[j]])
                                if len(idxs[j]) > self.max_patch_size:
                                    self.max_patch_size = len(idxs[j])
                        chunk_chunk = i
                        chunks.append([tmp_points,tmp_normals,chunk_chunk])
            _ = patch(chunks[0][0][0],chunks[0][1][0])
            self.patches = self.parallel_build(chunks,show_individual=show_individual)
        else:
            self.max_patch_size = self.points.shape[0]
            self.patches = self.parallel_build(None,show_individual=show_individual,PU=PU)

    def build(self,**kwargs):
        h          = kwargs.get('h',0)
        q          = kwargs.get('q',1)
        d_num      = kwargs.get('d_num',2)
        resolution = kwargs.get('resolution',20)
        k          = kwargs.get('k',2)
        level      = kwargs.get('level',0)
        visualize  = kwargs.get('visualize',False)
        buffer     = kwargs.get('buffer',1)
        self.x_range = [min(self.points[:,0]),max(self.points[:,0])]
        self.y_range = [min(self.points[:,1]),max(self.points[:,1])]
        self.z_range = [min(self.points[:,2]),max(self.points[:,2])]
        solution_matrix = []
        a_coef          = []
        b_coef          = []
        b_sol           = []
        n_points        = []
        patch_points    = []
        patch_x         = []
        patch_y         = []
        patch_z         = []
        patch_points_m  = []
        for patch in self.patches:
            solution_matrix_tmp = np.empty((self.max_patch_size,len(patch.b)))
            b_sol_tmp = np.zeros((self.max_patch_size,len(patch.b)))
            points = np.empty((self.max_patch_size,len(patch.b)-1))
            patch_points_m.append(patch.points)
            solution_matrix_tmp.fill(np.nan)
            points.fill(np.nan)
            points[:patch.ndim,:] = patch.points
            a_coef.append(patch.a)
            b_coef.append(patch.b)
            n_points.append(patch.ndim)
            patch_points.append(points)
            patch_x.append(patch.points[0,0])
            patch_y.append(patch.points[0,1])
            patch_z.append(patch.points[0,2])
            solution_matrix_tmp[:patch.ndim,0] = patch.a[:patch.ndim]
            b_sol_tmp[0,:] = patch.b
            for point in range(patch.ndim):
                solution_matrix_tmp[point,1] = patch.a[patch.ndim+point]
                solution_matrix_tmp[point,2] = patch.a[patch.ndim*2+point]
                solution_matrix_tmp[point,3] = patch.a[patch.ndim*3+point]
            solution_matrix.append(solution_matrix_tmp)
            b_sol.append(b_sol_tmp)
        self.solution_matrix = solution_matrix
        self.a_coef          = a_coef
        self.b_coef          = b_coef
        self.b_sol           = b_sol
        self.n_points        = n_points
        self.patch_points    = patch_points
        self.patch_x         = patch_x
        self.patch_y         = patch_y
        self.patch_z         = patch_z
        self.patch_points_m  = patch_points_m
        if len(self.patches) > 1:
            functions = []
            KDTree = spatial.KDTree(np.array([patch_x,patch_y,patch_z]).T)
            self.patch_KDTree = KDTree
            preassembled_functions,function_strings = construct(d_num)
            foo = imp.new_module("foo")
            sys.modules["foo"] = foo
            pickled_DD = []
            DD = []
            exec("import numpy as np",foo.__dict__)
            for fss in function_strings:
                exec(fss,foo.__dict__)
            function_names = list(filter(lambda item: '__' not in item,dir(foo)))
            for fn in function_names:
                if fn == 'np':
                    continue
                exec("pickled_DD.append(dumps(partial(foo.{},KDTree=KDTree,patch_points".format(fn)+\
                     "=np.array(patch_points),b_coef=np.array(b_sol),h=h,q=q,sol_mat=np"+\
                     ".array(solution_matrix),patch_1=np.array(patch_x),patch_2=np.arra"+\
                     "y(patch_y),patch_3=np.array(patch_z))))")
                exec("DD.append(partial(foo.{},KDTree=KDTree,patch_points".format(fn)+\
                     "=np.array(patch_points),b_coef=np.array(b_sol),h=h,q=q,sol_mat=np"+\
                     ".array(solution_matrix),patch_1=np.array(patch_x),patch_2=np.arra"+\
                     "y(patch_y),patch_3=np.array(patch_z)))")
            func_marching = partial(function_marching,patch_points=patch_points_m,
                                    a_coef=a_coef,b_coef=b_coef,h=h,q=q,
                                    patch_x=np.array(patch_x),patch_y=np.array(patch_y),
                                    patch_z=np.array(patch_z))
            self.function_marching = func_marching
        else:
            functions = []
            preassembled_functions,function_strings = construct_global(d_num)
            foo = imp.new_module("foo")
            sys.modules["foo"] = foo
            pickled_DD = []
            DD = []
            exec("import numpy as np",foo.__dict__)
            for fss in function_strings:
                exec(fss,foo.__dict__)
            function_names = list(filter(lambda item: '__' not in item,dir(foo)))
            for fn in function_names:
                if fn == 'np':
                    continue
                exec("pickled_DD.append(dumps(partial(foo.{},patch_points=np.array(patch_points),b_coef=np.array(b_sol),sol_mat=np.array(solution_matrix))))".format(fn))
                exec("DD.append(partial(foo.{},patch_points=np.array(patch_points),b_coef=np.array(b_sol),sol_mat=np.array(solution_matrix)))".format(fn))
        self.pickled_DD = pickled_DD
        self.DD = DD
        #print('get properties')
        self._get_properties_(resolution=resolution,k=k,level=level,visualize=visualize,buffer=buffer)
        """
        DD = []
        CDD = []
        for f in functions:
            if len(self.patches) > 1:
                DD.append(partial(f,KDTree=KDTree,patch_points=np.array(patch_points),
                                  b_coef=np.array(b_sol),h=h,q=q,sol_mat=np.array(solution_matrix),
                                  patch_1=np.array(patch_x),patch_2=np.array(patch_y),
                                  patch_3=np.array(patch_z)))
            else:
                tmp_DD = partial(foo,patch_points=np.array(patch_points),b_coef=np.array(b_sol),
                                 sol_mat=np.array(solution_matrix))
                CDD.append(dumps(tmp_DD))
                DD.append(tmp_DD)
        if len(self.patches) > 1:
            func = partial(function,KDTree=KDTree,patch_points=np.array(patch_points),
                           b_coef=np.array(b_coef),h=h,q=q,sol_mat=np.array(solution_matrix),
                           patch_x=np.array(patch_x),patch_y=np.array(patch_y),
                           patch_z=np.array(patch_z))
            func_marching = partial(function_marching,patch_points=patch_points_m,
                                    a_coef=a_coef,b_coef=b_coef,h=h,q=q,
                                    patch_x=np.array(patch_x),patch_y=np.array(patch_y),
                                    patch_z=np.array(patch_z))
            grad = partial(gradient,KDTree=KDTree,patch_points=np.array(patch_points),
                           b_coef=np.array(b_coef),h=h,q=q,sol_mat=np.array(solution_matrix),
                           patch_x=np.array(patch_x),patch_y=np.array(patch_y),
                           patch_z=np.array(patch_z))
            self.function = func
            self.function_marching = func_marching
            self.gradient = grad
        self.DD = DD
        self.CDD = CDD
        """
    def build_individual(self,idx,h=0,q=1):
        func_marching = partial(function_marching,patch_points=[self.patch_points_m[idx]],
                                a_coef=[self.a_coef[idx]],b_coef=[self.b_coef[idx]],h=h,q=q,
                                patch_x=np.array([self.patch_x[idx]]),patch_y=np.array([self.patch_y[idx]]),
                                patch_z=np.array([self.patch_z[idx]]))
        return func_marching

    def within(self,x,y,z,k):
        return self.DD[0]([x,y,z,k]) < 0

    def within_mesh(self,x,y,z,k):
        point = pv.PolyData(np.array([x,y,z]))
        select = point.select_enclosed_points(self.pv_polydata)
        return bool(select['SelectedPoints'][0])

    def pick_in_cell(self,cell_idx):
        simplex = self.tet_pts[self.tet_verts[cell_idx,:],:]
        point = np.array(generate(simplex,1))
        return point,None

    def pick(self,**kwargs):
        homogeneous = kwargs.get('homogeneous',False)
        replacement = kwargs.get('replacement',True)
        size = kwargs.get('size',1)
        vert_applied = kwargs.get('vert',False)
        verts = kwargs.get('verts',[])
        if homogeneous:
            point = None
            if not vert_applied:
                rdx = np.random.Generator(np.random.PCG64()).choice(list(range(self.tet_verts.shape[0])),size,p=self.norm_ele_vol,replace=replacement)
            else:
                rdx = verts
            simplex = self.tet_pts[self.tet_verts[rdx,:],:]
            point = np.array(generate(simplex,size))
            return point,None
        id = np.random.choice(self.pv_polydata_surf.n_faces)
        faces = self.pv_polydata_surf.faces.reshape(-1,self.pv_polydata_surf.faces[0]+1)
        pts = self.pv_polydata_surf.points[faces[id,1:],:]
        a1 = np.random.random(1)
        a2 = np.random.random(1)
        if (a1+a2) > 1:
            a1 = 1 - a1
            a2 = 1 - a2
        A = 1 - a1 - a2
        B = a1
        C = a2
        pts = pts*np.array([A,B,C])
        pts = np.sum(pts,axis=0)
        pts = np.array(pts)
        #x = np.random.uniform(self.x_range[0],self.x_range[1])
        #y = np.random.uniform(self.y_range[0],self.y_range[1])
        #z = np.random.uniform(self.z_range[0],self.z_range[1])
        #k     = kwargs.get('k',len(self.patches))
        #layer = kwargs.get('layer',0)
        #x0    = kwargs.get('x0',np.array([x,y,z]))
        #f     = lambda x: (self.DD[0]([x[0],x[1],x[2],k])-layer)**2
        #df    = lambda x: 2*(self.DD[0]([x[0],x[1],x[2],k])-layer)*self.DD[1]([x[0],x[1],x[2],k])
        #ddf   = lambda x,y,z: 2*np.dot(self.DD[1]([x,y,z,k],self.DD[1]([x,y,z,k]))+2*self.DD[2]([x,y,z,k])*(self.DD[0]([x,y,z,k])-layer)
        #res   = optimize.minimize(f,x0,method='L-BFGS-B')
        return pts,None

    def path(self,start,distance,steps,dive=0.01,theta_steps=40):
        start_layer = self.DD[0]([start[0],start[1],start[2],len(self.patches)])
        theta = np.linspace(0,2*np.pi,theta_steps)
        path = [start]
        path_d = [0]
        path_layer = [start_layer]
        point = start
        while np.sum(path_d) < distance:
            normal = self.DD[1]([point[0],point[1],point[2],max(round(len(self.patches)),2)])
            t1,t2,unit_normal = tangent_basis(normal,point)
            next_steps = path[-1] + (distance/steps)*(t1*np.cos(theta).reshape(-1,1)+t2*np.sin(theta).reshape(-1,1)) #-(dive/steps)*(unit_normal)
            distances  = np.linalg.norm(next_steps - path[-1],axis=1)
            distances1 = np.linalg.norm(next_steps - path[0],axis=1)+np.linalg.norm(next_steps - path[-1],axis=1)
            next_step  = next_steps[np.argmax(distances1),:]
            next_step,next_layer = self.pick(layer=path_layer[-1]-(dive/steps),x0=next_step)
            path_layer.append(next_layer)
            path_d.append(distances[np.argmax(distances)])
            path.append(next_step)
            point = next_step
        return path,path_d,path_layer

    def mesh(self,path,distance,steps,dive=0.01,theta_steps=40,others=None):
        center = round(len(path)*0.5)
        start  = path[center]
        start_layer = self.DD[0]([start[0],start[1],start[2],len(self.patches)])
        ###################
        ###################
        theta = np.linspace(0,2*np.pi,theta_steps)
        center_path = [start]
        path_d = []
        path_layer = [start_layer]
        point = start
        direction = (path[center+1] - path[center-1])/np.linalg.norm(path[center+1] - path[center-1])
        while np.sum(path_d) < distance:
            normal = self.DD[1]([point[0],point[1],point[2],max(round(len(self.patches)*0.1),2)])
            t1,t2,unit_normal = tangent_basis(normal,point)
            next_steps = center_path[-1] + (distance/steps)*(t1*np.cos(theta).reshape(-1,1)+
                                                             t2*np.sin(theta).reshape(-1,1)) #-(dive/steps)*(unit_normal)
            distances  = np.linalg.norm(next_steps - center_path[-1],axis=1)
            if len(center_path) == 1:
                distances1 = np.linalg.norm(next_steps - center_path[0],axis=1)+np.linalg.norm(next_steps - path[center+1],axis=1)+np.linalg.norm(next_steps - path[center-1],axis=1)
            else:
                #distances1 = np.linalg.norm(next_steps - center_path[0],axis=1)+np.linalg.norm(next_steps - center_path[-1],axis=1)+\
                if others is None:
                    distances1 = (1 - np.sum(path_d)/distance)*np.linalg.norm(next_steps - center_path[0],axis=1)+np.sum(path_d)/distance*np.linalg.norm(next_steps - center_path[-1],axis=1)
                else:
                    other_dist,_ = others.query(next_steps,k=1)
                    distances1 = (1 - np.sum(path_d)/distance)*np.linalg.norm(next_steps - center_path[0],axis=1)+np.sum(path_d)/distance*np.linalg.norm(next_steps - center_path[-1],axis=1)+\
                                 (other_dist)
            next_step  = next_steps[np.argmax(distances1),:]
            next_step,next_layer = self.pick(layer=path_layer[-1]-(dive/steps),x0=next_step)
            path_layer.append(next_layer)
            path_d.append(distances[np.argmax(distances)])
            center_path.append(next_step)
            point = next_step
        past_up = []
        past_down = []
        mesh = []
        #mesh.extend(center_path)
        #mesh.extend(path)
        up_past = []
        down_past = []
        past_point_up = []
        past_point_down = []
        past_line = center_path
        #Upper triangle Mesh
        #print('Max Center Path: {}'.format(len(path[center+1:])))
        for i in range(len(path[center+1:-1])):
            ii = i + center +1
            point = path[ii]
            line = [point]
            percent = round(min((1-i/len(path[center+1:])),1)*len(center_path))
            #print(percent)
            for j in range(1,percent):
                normal = self.DD[1]([point[0],point[1],point[2],max(round(len(self.patches)*0.1),2)])
                m1 = (past_line[j] - past_line[j-1])/np.linalg.norm(past_line[j] - past_line[j-1])
                m2 = (point - past_line[j-1])/np.linalg.norm(line[-1] - past_line[j-1])
                predicted = past_line[j-1] + m1*(distance/steps)+m2*(distance/steps)
                t1,t2,unit_normal = tangent_basis(normal,point)
                next_steps = point + (distance/steps)*(t1*np.cos(theta).reshape(-1,1)+t2*np.sin(theta).reshape(-1,1))
                distances = np.linalg.norm(next_steps - predicted,axis=1)
                line.append(next_steps[np.argmin(distances),:])
                point = line[-1]
            past_line = line
            mesh.extend(line)
            #show_mesh(np.array(mesh),np.array(path),np.array(center_path))
        past_line = center_path
        #Lower Triangle Mesh
        for i in range(len(path[1:center-1])):
            ii = center -1-i
            point = path[ii]
            line = [point]
            percent = round(min((1-i/len(path[:center-1])),1)*len(center_path))
            for j in range(1,percent):
                normal = self.DD[1]([point[0],point[1],point[2],max(round(len(self.patches)*0.1),2)])
                m1 = (past_line[j] - past_line[j-1])/np.linalg.norm(past_line[j] - past_line[j-1])
                m2 = (point - past_line[j-1])/np.linalg.norm(line[-1] - past_line[j-1])
                predicted = past_line[j-1] + m1*(distance/steps)+m2*(distance/steps)
                t1,t2,unit_normal = tangent_basis(normal,point)
                next_steps = point + (distance/steps)*(t1*np.cos(theta).reshape(-1,1)+t2*np.sin(theta).reshape(-1,1))
                distances = np.linalg.norm(next_steps - predicted,axis=1)
                line.append(next_steps[np.argmin(distances),:])
                point = line[-1]
            past_line = line
            mesh.extend(line)
        #proximal = path[0]
        #distal = path[-1]
        #terminal = center_path[-1]
        #mesh.insert(0,terminal)
        #mesh.insert(0,distal)
        #mesh.insert(0,proximal)
        mesh = np.array(mesh)
        mesh_tree = cKDTree(mesh)
        graph = mesh_tree.sparse_distance_matrix(mesh_tree,((distance/steps)**2+(dive/steps)**2)*(1/2))
        return mesh,path,center_path,path_d,graph

    def _get_properties_(self,**kwargs):
        resolution = kwargs.get('resolution',20)
        k          = kwargs.get('k',2)
        level      = kwargs.get('level',0)
        visualize  = kwargs.get('visualize',False)
        buffer     = kwargs.get('buffer',1)
        self.polydata,self.tet_pts,self.tet_verts,self.ele_vol,self.tet = marching_cubes(self,resolution,k,level,visualize,buffer)
        self.cell_lookup = cKDTree(self.tet.grid.cell_centers().points)
        self.pv_polydata = pv.PolyData(var_inp=self.polydata)
        self.pv_polydata_surf = self.pv_polydata.extract_surface()
        nodes_check = set([])
        nodes = []
        idx = [[0,1],[1,2],[2,3],[3,1]]
        lengths = []
        for tet in self.tet_verts:
            for dx in idx:
                if tuple([tet[dx[0]],tet[dx[1]]]) in nodes_check:
                    continue
                nodes_check.add(tuple([tet[dx[0]],tet[dx[1]]]))
                nodes_check.add(tuple([tet[dx[1]],tet[dx[0]]]))
                nodes.append([tet[dx[0]],tet[dx[1]]])
                nodes.append([tet[dx[1]],tet[dx[0]]])
                l = np.linalg.norm(self.tet_pts[tet[dx[0]]] - self.tet_pts[tet[dx[1]]])
                lengths.append(l)
                lengths.append(l)
        M = np.array(nodes)
        graph = csr_matrix((lengths,(M[:,0],M[:,1])),shape=(max(M[:,0])+1,max(M[:,1])+1))
        self.graph = graph
        self.shortest_algorithm = lambda ind: shortest_path(csgraph=graph,directed=False,indices=ind,return_predecessors=True)
        def get_path(i,j):
            ds, Pr = shortest_path(csgraph=graph,directed=False,indices=i,return_predecessors=True)
            path = [j]
            lengths = []
            k = j
            while Pr[k] != -9999:
                path.append(Pr[k])
                lengths.append(ds[k])
                k = Pr[k]
            #lengths.append(ds[-9999])
            path = path[::-1]
            lengths = lengths[::-1]
            lines = []
            for jdx in range(len(path)-1):
                lines.append([path[jdx],path[jdx+1]])
            return path,lengths,lines
        def get_cells(path,grid=self.tet.grid):
            cells = []
            cell_sets = []
            for idx,i in enumerate(path):
                cells.append(grid.extract_points(i))
                cell_sets.append(set(cells[-1].cell_data['vtkOriginalCellIds']))
                if idx > 0:
                    cell_sets[-2] = cell_sets[-2].difference(cell_sets[-1])
                    cell_sets[-1] = cell_sets[-1].difference(cell_sets[-2])
            cell_sets = [list(cs) for cs in cell_sets]
            return cell_sets
        def get_cell_points(cells,verts=self.tet_verts,pts=self.tet_pts):
            return pts[verts[cells]]
        def point_function(s1,s2,s3,n,cell_pts):
            dn = 1/cell_pts.shape[0]
            N  = int(n // dn)
            if N > cell_pts.shape[0]-1:
                N =  cell_pts.shape[0]-1
            if N < 0:
                N = 0
            if s1 + s2 > 1:
                s1 = 1.0 - s1
                s2 = 1.0 - s2
            if s2 + s3 > 1:
                tmp = s3
                s3 = 1.0 - s1 - s2
                s2 = 1.0 - tmp
            elif s1 + s2 + s3 > 1:
                tmp = s3
                s3 = s1 + s2 + s3 - 1.0
                s1 = 1.0 - s2 - tmp
            s4 = 1 - s1 - s2 - s3
            #print(s4)
            #print(s1.VALUE)
            #print(s2.VALUE)
            #print(s3.VALUE)
            #print(s4.value)
            S  = np.array([s1,s2,s3,s4]).reshape(-1,1)
            #id = 0
            #for i in range(cell_pts.shape[0]):
            #    if n.value == i:
            #        id = i
            #        break
            #P1  = s1*cell_pts[id,0,0] + s2*cell_pts[id,1,0] + s3*cell_pts[id,2,0] + s4*cell_pts[id,3,0]
            #P2  = s1*cell_pts[id,0,1] + s2*cell_pts[id,1,1] + s3*cell_pts[id,2,1] + s4*cell_pts[id,3,1]
            #P3  = s1*cell_pts[id,0,2] + s2*cell_pts[id,1,2] + s3*cell_pts[id,2,2] + s4*cell_pts[id,3,2]
            #print(id)
            P  = cell_pts[N]*S
            return np.sum(P,axis=0)

        def get_point_functions(cell_sets):
            point_functions = []
            for i in range(len(cell_sets)):
                cell_pts = get_cell_points(cell_sets[i])
                f = partial(point_function,cell_pts=cell_pts)
                point_functions.append(f)
            return point_functions
        def path_function(x,start,end,point_functions):
            a = [start]
            b = []
            #l = []
            for i in range(len(point_functions)):
                x_tmp = x[i*4:i*4+4]
                pt = point_functions[i](x_tmp[0],x_tmp[1],x_tmp[2],x_tmp[3])
                #l.append(((pt[0] - a[-1][0])**2+(pt[1] - a[-1][1])**2+(pt[2] - a[-1][2])**2)**(1/2))
                a.append(pt)
                b.append(pt)
            b.append(end)
            #l.append(((pt[0] - end[0])**2+(pt[1] - end[1])**2+(pt[2] - end[2])**2)**(1/2))
            A = np.array(a)
            B = np.array(b)
            L = np.sum(np.linalg.norm(B-A,axis=1))
            #L = sum(l)
            return L
        def build_path_function(i,j,start,end):
            path,_,_ = get_path(i,j)
            cell_sets = get_cells(path)
            point_functions = get_point_functions(cell_sets)
            pf = partial(path_function,start=start,end=end,point_functions=point_functions)
            return pf,point_functions
        def optimize_path(pf,func_num,niter=200,disp=True):
            x0 = np.zeros(func_num*4)
            bounds = []
            for i in range(len(x0)):
                bounds.append(tuple([0,0.99]))
            kwargs = {'method':"L-BFGS-B",
                      'bounds':bounds}
            #m = GEKKO(remote=False)
            #x = []
            #for i in range(len(funcs)):
            #    x.append(m.Var(value=0,lb=0,ub=1))
            #    x.append(m.Var(value=0,lb=0,ub=1))
            #    x.append(m.Var(value=0,lb=0,ub=1))
            #    x.append(m.Var(value=0,lb=0,ub=len(funcs[i].keywords['cell_pts']),integer=True))
            #x = m.Array(m.Var,len(x0),lb=0,ub=1,integer=True)
            #m.Minimize(pf(x))
            #m.options.SOLVER=1
            #m.solver_options = ['minlp_maximum_iterations 500', \
            #        # minlp iterations with integer solution
            #        'minlp_max_iter_with_int_sol 10', \
            #        # treat minlp as nlp
            #        'minlp_as_nlp 0', \
            #        # nlp sub-problem max iterations
            #        'nlp_maximum_iterations 50', \
            #        # 1 = depth first, 2 = breadth first
            #        'minlp_branch_method 1', \
            #        # maximum deviation from whole number
            #        'minlp_integer_tol 0.05', \
            #        # covergence tolerance
            #        'minlp_gap_tol 0.01']
            #m.solve()

            res = optimize.basinhopping(pf,x0,niter=niter,minimizer_kwargs=kwargs,disp=disp)
            return res
        def optimized_points(res,point_functions,start,end):
            pts = [start]
            a = [start]
            b = []
            sol = res.x
            for i in range(len(point_functions)):
                x_tmp = sol[i*4:i*4+4]
                pts.append(point_functions[i](x_tmp[0],x_tmp[1],x_tmp[2],x_tmp[3]))
                a.append(pts[-1])
                b.append(pts[-1])
            pts.append(end)
            b.append(end)
            A = np.array(a)
            B = np.array(b)
            L = np.linalg.norm(B-A,axis=1)
            return pts,L
        def find_best_path(start,end,grid=self.tet.grid,niter=200,disp=True):
            i = grid.find_closest_point(start)
            j = grid.find_closest_point(end)
            pf,funcs = build_path_function(i,j,start,end)
            res = optimize_path(pf,len(funcs),niter=niter,disp=disp)
            pts,L = optimized_points(res,funcs,start,end)
            ## check for unique
            _,u_pts_idx = np.unique(pts,axis=0,return_index=True)
            u_pts_idx = np.flip(u_pts_idx)
            rm_idx = np.argwhere(np.diff(u_pts_idx)>1)
            return pts,L,res,pf,funcs,rm_idx
        def spline_resample(path_pts):
            if len(path_pts) == 2:
                path = splprep(path_pts.T,k=1,s=0)
            elif len(path_pts) == 3:
                path = splprep(path_pts.T,k=2,s=0)
            else:
                path = splprep(path_pts.T,s=0)
            return path
        def resample(path,niter=20,disp=True):
            def resample_optimizer(on_off,path_pts=path,within=self.pickled_DD[0]):
                on_off = on_off > 0.5
                last_pts = path_pts[0]
                end_pts = path_pts[-1]
                cost = 0
                within = loads(within)
                cost += np.sum(on_off)
                for idx,i in enumerate(on_off):
                    if i == 1:
                        subdivision = 4
                        for j in range(1,2*subdivision):
                            tmp = (last_pts + path_pts[idx+1])*(j/(2*subdivision))
                            if within((tmp[0],tmp[1],tmp[2],2)) > 0.15:
                                cost += 100
                        last_pts = path_pts[idx+1]
                subdivision  = 4
                for j in range(1,2*subdivision):
                    tmp = (last_pts + path_pts[-1])*(j/(2*subdivision))
                    if within((tmp[0],tmp[1],tmp[2],2)) > 0.15:
                        cost += 100
                return cost
            x0 = np.ones(len(path)-2)
            bounds = []
            for i in range(len(x0)):
                bounds.append(tuple([0,1]))
            kwargs = {'method':"L-BFGS-B",
                      'bounds':bounds}
            #print(len(path))
            #m = GEKKO(remote=False)
            #x = m.Array(m.Var,len(x0),lb=0,ub=1,integer=True)
            #m.Minimize(resample_optimizer(x))
            #m.options.SOLVER=1
            #m.solve()
            #on_off = x
            #print(on_off)
            res = optimize.basinhopping(resample_optimizer,x0,niter=100,minimizer_kwargs=kwargs,disp=disp)
            on_off = res.x > 0.5
            #print(res[1])
            new_path = [path[0]]
            new_lengths = []
            for idx,i in enumerate(on_off):
                if i:
                    new_path.append(path[idx+1])
                    new_lengths.append(np.linalg.norm(new_path[-1]-new_path[-2]))
            new_path.append(path[-1])
            new_lengths.append(np.linalg.norm(new_path[-1]-new_path[-2]))
            #print(len(new_path))
            return np.array(new_path), new_lengths
        self.get_shortest_path = get_path
        self.build_path_function = build_path_function
        self.optimize_path = optimize_path
        self.optimized_points = optimized_points
        self.find_best_path = find_best_path
        self.resample = resample
        self.total_ele_vol = sum(self.ele_vol)
        self.norm_ele_vol = self.ele_vol/self.total_ele_vol
        volume,surface_area = properties(self.polydata)
        self.volume= volume
        self.surface_area = surface_area

    def subtract(self,other_surface):
        self.subtracted_volumes.append(other_surface)
        def d_0(xyz,F=self.DD[0],S=other_surface.DD[0]):
            return F(xyz)-(S(xyz)<0)*(F(xyz)+S(xyz))
        self.DD[0] = d_0
        def d_1(xyz,F=self.DD[0],S=other_surface.DD[0],dF=self.DD[1],dS=other_surface.DD[1]):
            return dF(xyz)-(S(xyz)<0)*(1-S(xyz))*(dF(xyz)+dS(xyz))
        self.DD[1] = d_1
        def d_2(xyz,F=self.DD[0],S=other_surface.DD[0],ddF=self.DD[2],ddS=self.DD[2]):
            return ddF(xyz)-(S(xyz)<0)*(1-S(xyz))*(ddF(xyz)+ddS(xyz))
        self.DD[2] = d_2
        self.volume = self.volume - other_surface.volume
        self.surface_area = self.surface_area - other_surface.surface_area

    def add(self,other_surface):
        self.added_volumes.append(other_surface)
        def d_0(xyz,F=self.DD[0],S=other_surface.DD[0]):
            return F(xyz)+(S(xyz)<0)(F(xyz)>0)*(S(xyz))
        self.DD[0] = d_0
        def d_1(xyz,F=self.DD[0],S=other_surface.DD[0],dF=self.DD[1],dS=other_surface.DD[1]):
            return dF(xyz)+(S(xyz)<0)*(F(xyz)>0)*dS(xyz)
        self.DD[1] = d_1
        def d_2(xyz,F=self.DD[0],S=other_surface.DD[0],ddF=self.DD[2],ddS=self.DD[2]):
            return ddF(xyz)+(S(xyz)<0)*(F(xyz)>0)*ddS(xyz)
        self.DD[2] = d_2

    def export_polydata(self,file):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file)
        writer.SetInputDataObject(self.polydata)
        writer.Write()

def marching_cubes(surface_object,resolution=20,k=2,level=0,visualize=False,buffer=1):
    """
    Takes an interpolated volume and performs
    a descritization on a selected hyperplane.
    """
    surface_algorithm = vtk.vtkMarchingCubes()
    image_data        = vtk.vtkImageData()
    colors = vtk.vtkNamedColors()
    X,Y,Z = np.mgrid[surface_object.x_range[0]-buffer:surface_object.x_range[1]+buffer:resolution*1j,
                     surface_object.y_range[0]-buffer:surface_object.y_range[1]+buffer:resolution*1j,
                     surface_object.z_range[0]-buffer:surface_object.z_range[1]+buffer:resolution*1j]
    #print(X.shape[1],X.shape[0],X.shape[2])
    SHAPE = (resolution,resolution,resolution)
    Xs = ((-(surface_object.x_range[0]-buffer)+(surface_object.x_range[1]+buffer))/(resolution-1))
    Ys = ((-(surface_object.y_range[0]-buffer)+(surface_object.y_range[1]+buffer))/(resolution-1))
    Zs = ((-(surface_object.z_range[0]-buffer)+(surface_object.z_range[1]+buffer))/(resolution-1))
    SPACING = (Zs,Ys,Xs)
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    Kf = np.ones(Xf.shape,dtype=int)*k
    ORIGIN = (min(Zf),min(Yf),min(Xf))
    #Sample Implicit Function
    #print('Sampling Implicit Volume...')
    Vf = []
    for i in zip(Zf,Yf,Xf,Kf):
        Vf.append(surface_object.DD[0](i))
    Vf = np.array(Vf)
    Vf = Vf.reshape(X.shape)
    #grid = pv.UniformGrid(dims=SHAPE,spacing=SPACING,origin=ORIGIN)
    #out = grid.contour(1,scalars=Vf,rng=[0,0],method='marching_cubes')
    #out.plot(color='tan',smooth_shading=True)
    #print('Converting to VTK Image Data...')
    #Vf = np.array(Vf)
    #Vf = Vf.reshape(X.shape)
    pointsdata = vtk.vtkPoints()
    for i in range(surface_object.points.shape[0]):
        id = pointsdata.InsertNextPoint(surface_object.points[i,0],surface_object.points[i,1],surface_object.points[i,2])
    pt_poly = vtk.vtkPolyData()
    pt_poly.SetPoints(pointsdata)
    vertexfilter = vtk.vtkVertexGlyphFilter()
    vertexfilter.SetInputData(pt_poly)
    vertexfilter.Update()
    PTS = vtk.vtkPolyData()
    PTS.ShallowCopy(vertexfilter.GetOutput())
    pt_mapper = vtk.vtkPolyDataMapper()
    pt_mapper.SetInputData(PTS)
    pt_actor = vtk.vtkActor()
    pt_actor.SetMapper(pt_mapper)
    pt_actor.GetProperty().SetPointSize(10)
    pt_actor.GetProperty().SetColor(colors.GetColor3d('Red'))
    #Vf = np.flipud(Vf)
    #Vf = np.flip(Vf,1)
    data_vtk = numpy_support.numpy_to_vtk(Vf.ravel(),deep=1,array_type=vtk.VTK_FLOAT)
    image_data.SetDimensions(SHAPE)
    image_data.SetSpacing(SPACING)
    image_data.SetOrigin(ORIGIN)
    #image_data.SetOrigin(-np.max(Xf)/2,np.max(Yf)/2,-np.max(Zf)/2)
    image_data.GetPointData().SetScalars(data_vtk)
    image_data.Modified()
    #print('Marching...')
    surface_algorithm.SetInputData(image_data)
    surface_algorithm.ComputeNormalsOn()
    surface_algorithm.SetValue(0,level)

    if visualize:
        colors = vtk.vtkNamedColors()

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('White'))

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('Marching Cubes')

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface_algorithm.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))

        renderer.AddActor(actor)
        renderer.AddActor(pt_actor)
        render_window.Render()
        interactor.Start()
    else:
        pass
        #surface_algorithm.Update()
    if surface_object.polydata is not None:
        volume_polydata = surface_object.polydata
    else:
        surface_algorithm.Update()
        volume_polydata = surface_algorithm.GetOutput()
    transform = vtk.vtkTransform()
    pv_tmp = pv.PolyData(var_inp=volume_polydata)
    if not pv_tmp.is_all_triangles:
        #print('Triangulating mesh...')
        pv_tmp.triangulate(inplace=True)
        pv_tmp.plot(show_edges=True,opacity=0.5)
    tet = tetgen.TetGen(pv_tmp)
    try:
        nodes, verts = tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
    except:
        tet.make_manifold()
        nodes, verts = tet.tetrahedralize(order=1,mindihedral=20,minratio=1.5)
    cells = tet.grid.compute_cell_sizes()
    ele_vol = cells.cell_data["Volume"]
    return volume_polydata, nodes, verts, ele_vol, tet

def properties(polydata_object):
    MASS = vtk.vtkMassProperties()
    MASS.SetInputData(polydata_object)
    MASS.Update()
    return MASS.GetVolume(),MASS.GetSurfaceArea()

@nb.jit(nopython=True)
def generate(points,num):
    #r1,r2,r3 = np.random.random((3,num))
    s = np.zeros(num*4)
    for i in range(num):
        r1,r2,r3 = np.random.random(3)
        if r1 + r2 > 1:
            r1 = 1.0 - r1
            r2 = 1.0 - r2
        if r2 + r3 > 1:
            tmp = r3
            r3 = 1.0 - r1 - r2
            r2 = 1.0 - tmp
        elif r1 + r2 + r3 > 1:
            tmp = r3
            r3 = r1 + r2 + r3 - 1.0
            r1 = 1.0 - r2 - tmp
        a = 1.0 - r1 - r2 - r3
        idx_1 = i*4
        idx_2 = i*4+1
        idx_3 = i*4+2
        idx_4 = i*4+3
        s[idx_1] = a
        s[idx_2] = r1
        s[idx_3] = r2
        s[idx_4] = r3
    s = s.reshape((-1,1))
    points = s*points.reshape(num*4,3)
    points = np.sum(points.reshape(num,4,3),axis=1)
    return points
"""
@nb.jit(nopython=True)
def lines_to_cells(p0,centers,subdivisions):
    points = []
    for i in p0:
        for j in centers:
            for k in range(1,2*subdivisions):
                points.append((i+j)*(k/(2*subdivisions)))
"""
