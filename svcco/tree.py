import numpy as np
import pyvista as pv
import vtk
import sys,os
from .implicit.implicit import surface
from .branch_addition.check import *
from .branch_addition.close import *
from .branch_addition.basis import *
from .branch_addition.calculate_length import *
from .branch_addition.calculate_radii import *
from .branch_addition.set_root import *
from .branch_addition.add_edge import *
from .branch_addition.add_bifurcation import *
from .branch_addition.add_branches_v3 import *
from .forest_utils.connect import *
from .forest_utils.smooth import *
from .forest_utils.compete_add import *
from .sv_interface.get_sv_data import *
from .sv_interface.options import *
from .sv_interface.build_files import *
from .sv_interface.waveform import generate_physiologic_wave, wave
from .sv_interface.build_results import make_results
from .sv_interface.view_0d_result_plots import view_plots
from .sv_interface.build_0d_run_script import run_0d_script
from .sv_interface.locate import locate_0d_solver, locate_1d_solver
from .sv_interface.export_3d_only import export_3d_only
from .collision.collision import *
from .utils.gcode.gcode import *
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
import pickle
import tetgen
import json
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt
import platform
import pymeshfix
from .sv_interface import ROM
from .utils.remeshing.remesh import remesh_surface
#from wrapt_timeout_decorator import *
import tkinter as tk
from tkinter.filedialog import askopenfilename

def get():
    root = tk.Tk()
    root.withdraw()
    filename = askopenfilename()
    root.update()
    return filename


class tree:
    """
    Create a synthetic tree object.

    Parameters
    ----------

    Attributes
    ----------
    data : ndarray
           This is the contiguous 2d array of vessel data forming the vascular
           tree. Each row represents a single vessel within the tree. The array
           has a shape (N,31) where N is the current number of vessel segments
           within the tree.
           The following descibe the organization and importance of the column
           indices for each vessel.

           Column indicies:
                index: 0:2   -> proximal node coordinates
                index: 3:5   -> distal node coordinates
                index: 6:8   -> unit basis U
                index: 9:11  -> unit basis V
                index: 12:14 -> unit basis W (axial direction)
                index: 15,16 -> children (-1 means no child)
                index: 17    -> parent
                index: 18    -> proximal node index (only real edges)
                index: 19    -> distal node index (only real edges)
                index: 20    -> length (path length)
                index: 21    -> radius
                index: 22    -> flow
                index: 23    -> left bifurcation
                index: 24    -> right bifurcation
                index: 25    -> reduced resistance
                index: 26    -> depth
                index: 27    -> reduced downstream length
                index: 28    -> root radius scaling factor
                index: 29    -> edge that subedge belongs to
                index: 30    -> self identifying index

    parameters : dict
          This is a dictionary of values governing the formation of a synthetic
          tree.
    """
    def __init__(self):
        #self.parameters = {'gamma'   : 3.0,
        #                   'lambda'  : 2.0,
        #                   'mu'      : 1.0,
        #                   'nu'      : 3.6/100,
        #                   'Pperm'   : 100*1333.22,
        #                   'Pterm'   : 60*1333.22,
        #                   'Qterm'   : (0.125)/60,
        #                   'edge_num': 0}
        self.set_parameters()
        self.data = np.zeros((1,31))
        self.radius_buffer = 0.01
        self.fraction = None
        self.set_assumptions()
        self.rng_points = []
        self.time = {'search':[],
                     'constraints':[],
                     'local_optimize':[],
                     'collision':[],
                     'close_time':[],
                     'search_1':[],
                     'search_2':[],
                     'search_3':[],
                     'search_4':[],
                     'add_time':[],
                     'add_1':[],
                     'add_2':[],
                     'add_3':[],
                     'add_4':[],
                     'total':[],
                     'brute_optimize': [],
                     'method_optimize': [],
                     'depth':[],
                     'method_time':[],
                     'brute_time':[],
                     'brute_x_value':[],
                     'brute_value':[],
                     'method_x_value':[],
                     'method_value':[],
                     'truth_x_value':[],
                     'truth_value':[]}

    def set_parameters(self,**kwargs):
        self.parameters = {}
        self.parameters['gamma']    = kwargs.get('gamma',3.0)
        self.parameters['lambda']   = kwargs.get('lambda',2.0)
        self.parameters['mu']       = kwargs.get('mu',1.0)
        self.parameters['nu']       = kwargs.get('nu',3.6)/100
        self.parameters['Pperm']    = kwargs.get('Pperm',100)*1333.22
        self.parameters['Pterm']    = kwargs.get('Pterm',60)*1333.22
        self.parameters['Qterm']    = kwargs.get('Qterm',0.125)/60
        self.parameters['edge_num'] = kwargs.get('edge_num',0)
        self.parameters['rho']      = kwargs.get('rho',1.06)

    def set_assumptions(self,**kwargs):
        self.homogeneous = kwargs.get('homogeneous',True)
        self.directed    = kwargs.get('directed',False)
        self.convex      = kwargs.get('convex',False)
        self.hollow      = kwargs.get('hollow',False)
        self.dimension   = kwargs.get('dimension',3)
        self.clamped_root= kwargs.get('clamped_root',False)
        self.nonconvex_counter = 0

    def show_assumptions(self):
        print('homogeneous : {}'.format(self.homogeneous))
        print('directed    : {}'.format(self.directed))
        print('convex      : {}'.format(self.convex))
        print('hollow      : {}'.format(self.hollow))
        print('dimension   : {}'.format(self.dimension))
        print('clamped root: {}'.format(self.clamped_root))

    def set_boundary(self,boundary):
        self.boundary = boundary
        self.fraction = (self.boundary.volume**(1/3))/20

    def set_root(self,low=-1,high=0,start=None,direction=None,
                 limit_high=None,limit_low=None,niter=200):
        """

        Places the root of the vascular tree.

        Paramters
        ---------
                 low: float (default=-1.0)
                     the lower bound of the implicit domain through which the root
                     vessel is allowed to be placed in
                 high: float (default=0.0)
                     the upper bound of the implicit domain on which the root
                     vessel is allowed to be placed
                 start: ndarray with shape (1,3), (default=None)
                     perscribed starting point for a vascular tree. This will form
                     the proximal point to the root vessel.
                 direction: ndarray with shape (3,1), (defualt=None)
                     constrain the root vessel in a perscribed direction vector.
                     The root vessel will only lie along this direction.
                 limit_high: float (defualt=None)
                     depreciated, scheuled for removal
                 limit_low: float (defualt=None)
                     depreciated, scheuled for removal
                 niter: int (default=200)
                     number of sampling attempts before adjusting the search threshold
                     for the root vessel
        Returns
        -------
                 None
        """
        Qterm = self.parameters['Qterm']
        gamma = self.parameters['gamma']
        nu    = self.parameters['nu']
        Pperm = self.parameters['Pperm']
        Pterm = self.parameters['Pterm']
        result = set_root(self.data,self.boundary,Qterm,
                          gamma,nu,Pperm,Pterm,self.fraction,
                          self.homogeneous,self.convex,self.directed,
                          start,direction,limit_high,limit_low,self.boundary.volume,
                          low=-1,high=0,niter=niter)
        self.data = result[0]
        self.sub_division_map = result[1]
        self.sub_division_index = result[2]
        self.parameters['edge_num'] = 1
        #self.sub_division_map = [-1]
        #self.sub_division_index = np.array([0])

    def add(self,low,high,isforest=False,radius_buffer=0.01,threshold=None,method='L-BFGS-B'):
        """
        This method appends a new vessel to the existing tree object and optimizes
        the bifurcation to minimize for a defined cost function (see cost_function).

        Parameters
        ----------
                  low : float
                       the lower value acceptable within the implicit domain for
                       which a new terminal location will be considered valid
                       must be lower than the high value

                  high : float
                       the higher value acceptable within the implicit domain for
                       which a new terminal location will be considered valid
                       must be greater than the low value

                  isforest : boolean
                       This true/false flag indicates if the tree object being
                       appended to is part of a larger forest of networks. If so,
                       vessel appending is deferred until the candidate vessel is
                       checked against violating remaining networks being built
                       within the domain. (defualt is false)

                  radius_buffer : float
                       This is the padding distance for collision checking for the
                       new vessel. Becuase there are often requirement about the minimum
                       amount of distance that should exist between any two vessels
                       it is often necessary to add this padding. (default value is
                       0.01 units)

                  threshold : float
                       the minimum threshold euclidean distance that the new termainl
                       should be placed away from existing vessels in the current tree
                       (default is None, meaning that the value should be calculated
                       automatically)

                  method : str
                       a string argument supplying which type of minimizer to use
                       during bifurcation optimization. (default is 'L-BFGS-B')
                       We recommend that you use this since the default cost
                       function at bifurcations is convex; however, if the a user
                       defines a non-convex cost function at bifurcations,
                       performance may be impacted and another (non-newton) method
                       may be required to sufficiently approach the desired optimum.

                       Available minimizers are the following:

                       'L-BFGS-B' : Limited-Memory Bounded Broyden-Fletcher-Goldfarb-Shanno
                                    algorithm
                       'Nelder-Mead' : Nelder-Mead Simplex algorithm
                       'Powell' : Powell algorithm
                       'CG' : Conjugate Gradient algorithm
                       'Newton-CG' : Newton-Conjugate Gradient algorithm
                       'BFGS' : Broyden-Fletcher-Goldfarb-Shanno algorithm
                       'TNC' : Truncated Newton algorithm
                       'COBYLA' : Constrained Optimization BY Linear Approximation algorithm
                       'SLSQP' : Sequential Least Squares Programming
                       'dogleg' :  dog-leg trust-region algorithm
                       'trust-ncg' : Newton conjugate gradient trust-region algorithm
                       'trust-exact' : nearly exact trust-region algorithm
                       'trust_krylov' : nearly exact trust-region algorithm;
                                        only requires matrix vector products

                        See Scipy Optimization for futher documentation on
                        optimizers shown above.

        Returns
        -------
                  vessel : int
                       the index for the augmented data row containing the appended
                       vessel data to the vascular tree.

                  data   : ndarray
                       the augmented 2d array containing all of the relevant values
                       for the vascular tree with the newly appended vessel and
                       corresponding bifurcation, flow and radii updates. This
                       augmented data array will have two more rows than the inital
                       data array size. Thus array shape updates: (N,31) --> (N+2,31)

                  sub_division_map : list
                       the augmented list containing downstream vessels relative to each
                       vessel for fast lookup with the appended new vessel

                  sub_division_index : ndarray
                       the augmented 1d array mapping vessel index to the corresponding
                       downstream vessels in the sub_division_map with the appended
                       vessel

                  threshold : float
                       the minimum threshold euclidean distance that the new termainl
                       for which the new terminal point was placed away from existing
                       vessels in the current tree
        """
        vessel,data,sub_division_map,sub_division_index,threshold = add_branch(self,low,high,threshold_exponent=0.5,
                                                                     threshold_adjuster=0.75,all_max_attempts=40,
                                                                     max_attemps=3,sampling=20,max_skip=8,
                                                                     flow_ratio=None,radius_buffer=radius_buffer,
                                                                     isforest=isforest,threshold=threshold,method=method)
        if isforest:
            return vessel,data,sub_division_map,sub_division_index,threshold
        else:
            self.data = data
            self.parameters['edge_num'] += 2
            self.sub_division_map = sub_division_map
            self.sub_division_index = sub_division_index

    def n_add(self,n,method='L-BFGS-B'):
        self.rng_points,_ = self.boundary.pick(size=40*n,homogeneous=True)
        self.rng_points = self.rng_points.tolist()
        for i in tqdm(range(n),desc='Adding vessels'):
            #for i in range(n):
            #self.rng_points = self.rng_points.tolist()
            self.add(-1,0,method=method)

        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['search'][1:],label='search time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['constraints'][1:],label='constraint time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['local_optimize'][1:],label='local time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['collision'][1:],label='collision time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['search_1'][1:],label='search 1 time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['search_2'][1:],label='search 2 time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['search_3'][1:],label='search 3 time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['search_4'][1:],label='search 4 time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['close_time'][1:],label='close_time time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['add_time'][1:],label='add_time')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['add_1'][1:],label='add_1')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['add_2'][1:],label='add_2')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['add_3'][1:],label='add_3')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['add_4'][1:],label='add_4')
        #plt.plot(list(range(len(self.time['search'][1:]))),self.time['total'][1:],label='total')
        #plt.ylim([0,0.1])
        #plt.legend()
        #plt.show()

    def show(self,surface=False,vessel_colors='red',background_color='white',
             resolution=100,show_segments=True,save=False,name=None,show=True,
             surface_color='red',other_surface_color='blue'):
        models = []
        actors = []
        if not isinstance(vessel_colors,str):
            colors = vtk.vtkColor3d()
            colors.Set(vessel_colors[0],vessel_colors[1],vessel_colors[2])
        else:
            colors = vtk.vtkNamedColors()
            colors = colors.GetColor3d(vessel_colors)
        if not isinstance(background_color,str):
            backcolors = vtk.vtkColor3d()
            backcolors.Set(background_color[0],background_color[1],background_color[2])
        else:
            backcolors = vtk.vtkNamedColors()
            backcolors = backcolors.GetColor3d(background_color)
        if not isinstance(surface_color,str):
            surf_color = vtk.vtkColor3d()
            surf_color.Set(surface_color[0],surface_color[1],surface_color[2])
        else:
            surf_color = vtk.vtkNamedColors()
            surf_color = surf_color.GetColor3d(surface_color)
        if not isinstance(other_surface_color,str):
            other_surf_color = vtk.vtkColor3d()
            other_surf_color.Set(other_surface_color[0],other_surface_color[1],other_surface_color[2])
        else:
            other_surf_color = vtk.vtkNamedColors()
            other_surf_color = other_surf_color.GetColor3d(other_surface_color)
        if show_segments:
            if self.homogeneous:
                data_subset = self.data[self.data[:,-1] > -1]
            else:
                data_subset = self.data[self.data[:,29] > -1]
            for edge in range(data_subset.shape[0]):
                center = tuple((data_subset[edge,0:3] + data_subset[edge,3:6])/2)
                radius = data_subset[edge,21]
                direction = tuple(data_subset[edge,12:15])
                vessel_length = data_subset[edge,20]
                cyl = vtk.vtkTubeFilter()
                line = vtk.vtkLineSource()
                line.SetPoint1(data_subset[edge,0],data_subset[edge,1],data_subset[edge,2])
                line.SetPoint2(data_subset[edge,3],data_subset[edge,4],data_subset[edge,5])
                cyl.SetInputConnection(line.GetOutputPort())
                cyl.SetRadius(radius)
                cyl.SetNumberOfSides(resolution)
                models.append(cyl)
                mapper = vtk.vtkPolyDataMapper()
                actor  = vtk.vtkActor()
                mapper.SetInputConnection(cyl.GetOutputPort())
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors)
                actors.append(actor)
            if surface:
                mapper = vtk.vtkPolyDataMapper()
                actor  = vtk.vtkActor()
                mapper.SetInputData(self.boundary.polydata)
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(surf_color)
                actor.GetProperty().SetOpacity(0.25)
                actors.append(actor)
                for other_surface in self.boundary.subtracted_volumes:
                    mapper = vtk.vtkPolyDataMapper()
                    actor  = vtk.vtkActor()
                    mapper.SetInputData(other_surface.polydata)
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(other_surf_color)
                    actor.GetProperty().SetOpacity(0.5)
                    actors.append(actor)
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(backcolors)

            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetWindowName('SimVascular Vessel Network')

            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)

            for actor in actors:
                renderer.AddActor(actor)

            if save:
                render_window.Render()
                w2if = vtk.vtkWindowToImageFilter()
                w2if.SetInput(render_window)
                w2if.Update()

                writer = vtk.vtkPNGWriter()
                if name is None:
                    tag = time.gmtime()
                    name = str(tag.tm_year)+str(tag.tm_mon)+str(tag.tm_mday)+str(tag.tm_hour)+str(tag.tm_min)+str(tag.tm_sec)
                writer.SetFileName(os.getcwd()+os.sep+'{}.png'.format(name))
                writer.SetInputData(w2if.GetOutput())
                writer.Write()
            elif show==True:
                render_window.Render()
                interactor.Start()
            elif show==False:
                for m in models:
                    m.Update()
            return models

    def save(self,filename=None):
        if filename is None:
            tag = time.gmtime()
            filename = 'network_{}'.format(str(tag.tm_year)+
                                           str(tag.tm_mon) +
                                           str(tag.tm_mday)+
                                           str(tag.tm_hour)+
                                           str(tag.tm_min) +
                                           str(tag.tm_sec))
        os.mkdir(filename)
        file = open(filename+"/vessels.ccob",'wb')
        pickle.dump(self.data,file)
        file.close()
        file = open(filename+"/parameters.ccob",'wb')
        parameters = (self.parameters,self.fraction,self.homogeneous,self.directed,self.convex)
        pickle.dump(parameters,file)
        file.close()
        file = open(filename+"/boundary.ccob",'wb')
        pickle.dumps(self.boundary,file)
        file.close()

    def load(self,filename):
        file = open(filename+"/vessels.ccob",'rb')
        self.data = pickle.load(file)
        file.close()
        file = open(filename+"/parameters.ccob",'rb')
        self.parameters,self.fraction,self.homogeneous,self.directed,self.convex = pickle.load(file)
        file.close()
        file = open(filename+"/boundary.ccob",'rb')
        self.boundary = pickle.loads(file)
        file.close()

    def export(self,steady=True,apply_distal_resistance=True,gui=True,cylinders=False,make=True,global_edge_size=None,splines=False,splines_file_path='',spline_sample_points=100):
        if cylinders:
            pv_data = []
            models = self.show(show=False)
            for m in models:
                pv_data.append(pv.PolyData(var_inp=m.GetOutput()))
            merge = pv_data[0].merge(pv_data[1:])
            return merge
        interp_xyz,interp_r,interp_n,frames,branches,interp_xyzr = get_interpolated_sv_data(self.data)
        #points,radii,normals    = sv_data(interp_xyzr,interp_r,radius_buffer=self.radius_buffer)
        if steady:
            time = [0, 1]
            flow = [self.data[0,22], self.data[0,22]]
        else:
            time,flow = wave(self.data[0,22],self.data[0,21]*2) #changed wave function
            time = time.tolist()
            flow = flow.tolist()
            flow[-1] = flow[0]
        if apply_distal_resistance:
            R = self.parameters['Pterm']/self.data[0,22]
        else:
            R = 0
        if make:
            num_caps = 1+2+int((self.parameters['edge_num']-1)/2)
            options = file_options(num_caps,time=time,flow=flow,gui=gui,distal_resistance=R)
            if global_edge_size is None:
                options.set_mesh_options(global_edge_size=global_edge_size)
            points,radii,normals    = sv_data(interp_xyzr,interp_r,radius_buffer=self.radius_buffer)
            build(points,radii,normals,options)
        if splines:
            spline_file = open(splines_file_path+'b_splines.txt','w+')
            for ind,spline in enumerate(interp_xyzr):
                spline_file.write('Vessel: {}, Number of Points: {}\n\n'.format(ind,spline_sample_points))
                n = np.linspace(0,1,spline_sample_points)
                spline_data = splev(n,spline[0])
                for j in range(spline_sample_points):
                    spline_file.write('{}, {}, {}\n'.format(spline_data[0][j],spline_data[1][j],spline_data[2][j]))
                spline_file.write('\n')
            spline_file.close()
        return interp_xyz,interp_r,interp_xyzr

    def export_truncated(self,steady=True,radius=None,indicies=None,apply_distal_resistance=True,gui=True,cylinders=False,make=True):
        if cylinders:
            pv_data = []
            models = self.show(show=False)
            for m in models:
                pv_data.append(pv.PolyData(var_inp=m.GetOutput()))
            merge = pv_data[0].merge(pv_data[1:])
            return merge
        interp_xyz,interp_r,interp_n,frames,branches,interp_xyzr = get_truncated_interpolated_sv_data(self.data,radius=radius,indicies=indicies)
        points,radii,normals    = sv_data(interp_xyzr,interp_r,radius_buffer=self.radius_buffer)
        if steady:
            time = [0, 1]
            flow = [self.data[0,22], self.data[0,22]]
        else:
            time,flow = wave(self.data[0,22],self.data[0,21]*2)
            time = time.tolist()
            flow = flow.tolist()
            flow[-1] = flow[0]
        if apply_distal_resistance:
            R = self.parameters['Pterm']/self.data[0,22]
        else:
            R = 0
        if make:
            num_caps = 1+2+int((self.parameters['edge_num']-1)/2)
            options = file_options(num_caps,time=time,flow=flow,gui=gui,distal_resistance=R)
            build(points,radii,normals,options)
        return interp_xyz,interp_r

    def show_truncated(self,radius=None,indicies=None):
        large,small = truncate(self.data,radius=radius,indicies=indicies)
        combined = []
        plotter = pv.Plotter()
        models = []
        for i in large:
            combined += i
        for i in range(self.data.shape[0]):
            center = (self.data[i,0:3] + self.data[i,3:6])/2
            m = pv.Cylinder(center=center,direction=self.data[i,12:15],radius=self.data[i,21],height=self.data[i,20])
            if i in combined:
                plotter.add_mesh(m,color='red')
            else:
                plotter.add_mesh(m,color='blue',opacity=0.4)
        plotter.set_background('white')
        return plotter

    def export_3d_solid(self,outdir=None,folder="3d_tmp",watertight=False):
        if outdir is None:
            outdir = os.getcwd()+os.sep+folder
        else:
            outdir = outdir+os.sep+folder
        if not os.path.isdir(outdir):
            os.mkdir(folder)
        if not watertight:
            vessels = self.show(show=False)
            models  = [pv.PolyData(m.GetOutput()) for m in vessels]
            merge_model   = models[0]
            for i in range(1,len(models)):
                merge_model = merge_model.merge(models[i])
            merge_model.save(outdir+os.sep+"merged_model.vtp")
            return vessels,merge_model
        else:
            def polyline_from_points(pts,r):
                poly = pv.PolyData()
                poly.points = pts
                cell = np.arange(0,len(pts),dtype=np.int_)
                cell = np.insert(cell,0,len(pts))
                poly.lines = cell
                poly['radius'] = r
                return poly
            interp_xyz,interp_r,interp_xyzr = self.export(make=False)
            vessels = []
            tets = []
            surfs = []
            t_list = np.linspace(0,1,num=1000)
            for vessel_id in range(len(interp_xyz)):
                x,y,z = splev(t_list,interp_xyz[vessel_id][0])
                _,r = splev(t_list,interp_r[vessel_id][0])
                points = np.zeros((len(t_list),3))
                points[:,0] = x
                points[:,1] = y
                points[:,2] = z
                polyline = polyline_from_points(points,r)
                vessel = polyline.tube(radius=min(r),scalars='radius',radius_factor=max(r)/min(r)).triangulate()
                #vessel = polyline.tube(radius=r[0])
                vessels.append(vessel)
                vessel_fix = pymeshfix.MeshFix(remesh_surface(vessel.triangulate(),hausd=0.001,verbosity=0))
                vessel_fix.repair(verbose=False)
                surfs.append(vessel_fix.mesh)
                #tets.append(tetgen.TetGen(vessel_fix.mesh))
            #for i in tqdm(range(len(tets)),desc="Tetrahedralizing"):
            #    tets[i].tetrahedralize(minratio=1.2)
            #surfs = []
            #for i in tqdm(range(len(tets)),desc="Extracting Surfaces"):
            #    surfs.append(tets[i].grid.extract_surface().triangulate())
            #    surf_fix = pymeshfix.MeshFix(surfs[-1])
            #    surf_fix.repair(verbose=False)
            #    surfs[-1] = surf_fix.mesh
            unioned = surfs[0].boolean_union(surfs[1])
            unioned = unioned.clean()
            for i in tqdm(range(2,len(surfs)),desc="Unioning"):
                unioned = unioned.boolean_union(surfs[i])
                unioned = unioned.clean()
        unioned.save(outdir+os.sep+"unioned_solid.vtp")
        with open(outdir+os.sep+"simvascular_python_script.py","w") as file:
            file.write(export_3d_only.format(outdir+os.sep+"unioned_solid.vtp"))
        return vessels,unioned

    def export_0d_simulation(self,steady=True,outdir=None,folder="0d_tmp",number_cardiac_cycles=1,
                      number_time_pts_per_cycle=5,density=1.06,viscosity=0.04,material="olufsen",
                      olufsen={'k1':0.0,'k2':-22.5267,'k3':1.0e7,'material exponent':2.0,'material pressure':0.0},
                      linear={'material ehr':1e7,'material pressure':0.0},get_0d_solver=False,path_to_0d_solver=None,
                      viscosity_model='constant',vivo=True,distal_pressure=0):
        """
        This script builds the 0D input file for running 0D simulation.
        Parameters:
        -----------
        steady: bool, optional, [default=True]
        outdir: str, optional, [default=None]
        folder: str, optional, [default="tmp"]
        number_cardiac_cycles: int, optional, [default=1]
        """
        if outdir is None:
            outdir = os.getcwd()+os.sep+folder
        else:
            outdir = outdir+os.sep+folder
        if not os.path.isdir(outdir):
            os.mkdir(folder)
        if get_0d_solver:
            if path_to_0d_solver is None:
                path_to_0d_solver = locate_0d_solver()
            else:
                path_to_0d_solver = locate_0d_solver(windows_drive=path_to_0d_solver,linux_drive=path_to_0d_solver)
        else:
            path_to_0d_solver = None
        input_file = {'description':{'description of case':None,
                                     'analytical results':None},
                      'boundary_conditions':[],
                      'junctions':[],
                      'simulation_parameters':{},
                      'vessels':[]}
        simulation_parameters = {}
        simulation_parameters["number_of_cardiac_cycles"] = number_cardiac_cycles
        simulation_parameters["number_of_time_pts_per_cardiac_cycle"] = number_time_pts_per_cycle
        input_file['simulation_parameters'] = simulation_parameters
        terminal_vessels = np.argwhere(self.data[:,16]==-1).flatten()
        total_outlet_area = np.sum(np.pi*self.data[terminal_vessels,21]**2)
        total_resistance  = (self.parameters["Pterm"]-1333*distal_pressure)/self.data[0,22]
        for vessel in range(self.data.shape[0]):
            tmp = {}
            tmp['vessel_id'] = vessel
            tmp['vessel_length'] = self.data[vessel,20]
            tmp['vessel_name'] = 'branch'+str(vessel)+"_seg0"
            tmp['zero_d_element_type'] = "BloodVessel"
            if material == "olufsen":
                material_stiffness = olufsen['k1']*np.exp(olufsen['k2']*self.data[vessel,21])+olufsen['k3']
            else:
                material_stiffness = linear['material ehr']
            zero_d_element_values = {}
            if viscosity_model == 'constant':
                nu = self.parameters['nu']
            elif viscosity_model == 'modified viscosity law':
                W  = 1.1
                lam = 0.5
                D  = self.data[vessel,21]*2*(10000)
                nu_ref = self.parameters['nu']
                Hd = 0.45 #discharge hematocrit (dimension-less)
                C  = lambda d: (0.8+np.exp(-0.075))*(-1+(1/(1+10**(-11)*d**12)))+(1/(1+10**(-11)*d**12))
                nu_vitro_45 = lambda d: 220*np.exp(-1.3*d) + 3.2-2.44*np.exp(-0.06*d**0.645)
                nu_vivo_45  = lambda d: 6*np.exp(-0.085*d)+3.2-2.44*np.exp(-0.06*d**0.645)
                if vivo == True:
                    nu_45 = nu_vivo_45
                else:
                    nu_45 = nu_vitro_45
                nu_mod   =   lambda d: (1+(nu_45 - 1)*(((1-Hd)**C-1)/((1-0.45)**C-1))*(d/(d-W))**(4*lam))*(d/(d-W))**(4*(1-lam))
                ref = nu_ref - nu_mod(10000) # 1cm (units given in microns)
                nu  = nu_mod(D) + ref
            zero_d_element_values["R_poiseuille"] = ((8*nu/np.pi)*self.data[vessel,20])/self.data[vessel,21]**4
            zero_d_element_values["C"] = (3*self.data[vessel,20]*np.pi*self.data[vessel,21]**2)/(2*material_stiffness)
            zero_d_element_values["L"] = (self.data[vessel,20]*density)/(np.pi*self.data[vessel,21]**2)
            zero_d_element_values["stenosis_coefficient"] = 0.0
            tmp['zero_d_element_values'] = zero_d_element_values
            if vessel == 0:
                bc = {}
                bc['bc_name'] = "INFLOW"
                bc['bc_type'] = "FLOW"
                bc_values = {}
                if steady:
                    bc_values["Q"] = [self.data[vessel,22], self.data[vessel,22]]
                    bc_values["t"] = [0, 1]
                    with open(outdir+os.sep+"inflow.flow","w") as file:
                        for i in range(len(bc_values["t"])):
                            file.write("{}  {}\n".format(bc_values["t"][i],bc_values["Q"][i]))
                    file.close()
                else:
                    time,flow = wave(self.data[vessel,22],self.data[vessel,21]*2) # changed wave function
                    bc_values["Q"] = flow.tolist()
                    bc_values["t"] = time.tolist()
                    bc_values["Q"][-1] = bc_values["Q"][0]
                    simulation_parameters["number_of_time_pts_per_cardiac_cycle"] = len(bc_values["Q"])
                    with open(outdir+os.sep+"inflow.flow","w") as file:
                        for i in range(len(bc_values["t"])):
                            file.write("{}  {}\n".format(bc_values["t"][i],bc_values["Q"][i]))
                    file.close()
                bc['bc_values'] = bc_values
                input_file['boundary_conditions'].append(bc)
                tmp['boundary_conditions'] = {'inlet':"INFLOW"}
                if self.data[vessel,15] > 0 and self.data[vessel,16] > 0:
                    junction = {}
                    junction['inlet_vessels'] = [vessel]
                    junction['junction_name'] = "J"+str(vessel)
                    junction['junction_type'] = "NORMAL_JUNCTION"
                    junction['outlet_vessels'] = [int(self.data[vessel,15]), int(self.data[vessel,16])]
                    input_file['junctions'].append(junction)
            elif self.data[vessel,15] < 0 and self.data[vessel,16] < 0:
                bc = {}
                bc['bc_name'] = "OUT"+str(vessel)
                bc['bc_type'] = "RESISTANCE"
                bc_values = {}
                bc_values["Pd"] = 0#self.parameters["Pterm"]
                bc_values["R"] = total_resistance*(total_outlet_area/(np.pi*self.data[vessel,21]**2))
                bc['bc_values'] = bc_values
                input_file['boundary_conditions'].append(bc)
                tmp['boundary_conditions'] = {'outlet':'OUT'+str(vessel)}
            else:
                junction = {}
                junction['inlet_vessels'] = [vessel]
                junction['junction_name'] = "J"+str(vessel)
                junction['junction_type'] = "NORMAL_JUNCTION"
                junction['outlet_vessels'] = [int(self.data[vessel,15]), int(self.data[vessel,16])]
                input_file['junctions'].append(junction)
            input_file['vessels'].append(tmp)
        obj = json.dumps(input_file,indent=4)
        with open(outdir+os.sep+"solver_0d.in","w") as file:
            file.write(obj)
        file.close()

        with open(outdir+os.sep+"plot_0d_results_to_3d.py","w") as file:
            file.write(make_results)
        file.close()

        with open(outdir+os.sep+"plot_0d_results_at_slices.py","w") as file:
            file.write(view_plots)
        file.close()

        with open(outdir+os.sep+"run.py","w") as file:
            if platform.system() == "Windows":
                if path_to_0d_solver is not None:
                    solver_path = path_to_0d_solver.replace(os.sep,os.sep+os.sep)
                else:
                    solver_path = path_to_0d_solver
                    print("WARNING: Solver location will have to be given manually")
                    print("Current solver path is: {}".format(solver_path))
                solver_file = (outdir+os.sep+"solver_0d.in").replace(os.sep,os.sep+os.sep)
            else:
                if path_to_0d_solver is not None:
                    solver_path = path_to_0d_solver
                else:
                    solver_path = path_to_0d_solver
                    print("WARNING: Solver location will have to be given manually")
                    print("Current solver path is: {}".format(solver_path))
                solver_file = outdir+os.sep+"solver_0d.in"
            file.write(run_0d_script.format(solver_path,solver_file))
        file.close()

        geom = np.zeros((self.data.shape[0],8))
        geom[:,0:3] = self.data[:,0:3]
        geom[:,3:6] = self.data[:,3:6]
        geom[:,6]   = self.data[:,20]
        geom[:,7]   = self.data[:,21]
        np.savetxt(outdir+os.sep+"geom.csv",geom,delimiter=",")

    def export_1d_simulation(self,steady=True,outdir=None,folder="1d_tmp",number_cariac_cycles=1,num_points=1000,
                             distal_pressure=0,resistance_split=(1,0)):
        interp_xyz,interp_r,interp_xyzr = self.export(make=False)
        # Make Centerline Approximation Polydata
        #branches = get_branches(self.data)
        #centerline_ids = []
        #for id,branch in enumerate(branches):
        #    branch_ids = []
        #    for seg in branch

        if outdir is None:
            outdir = os.getcwd()
        outdir = outdir + os.sep + folder

        if not os.path.exists(outdir):
            os.mkdir(outdir)

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

        polys = []
        total_outlet_area = 0
        for ind in range(len(interp_xyz)): #(interp_xyzr):
            n = np.linspace(0,1,num_points)
            #spline_data = splev(n,spline[0])
            #spline_data_normal = splev(n,spline[0],der=1)
            spline_data = splev(n,interp_xyz[ind][0])
            spline_data_normal = splev(n,interp_xyz[ind][0],der=1)
            _,spline_r_data   = splev(n,interp_r[ind][0])
            spline_r_data = np.array(spline_r_data).flatten()
            points = make_points(spline_data[0],spline_data[1],spline_data[2])
            normal = make_points(spline_data_normal[0],spline_data_normal[1],spline_data_normal[2])
            normal = normal/np.linalg.norm(normal,axis=1).reshape(-1,1)
            poly_line = lines_from_points(points)
            poly_line['VesselId'] = np.ones(num_points,dtype=int)*ind
            poly_line['MaximumInscribedSphereRadius'] = spline_r_data #spline_data[3]
            poly_line['CenterlineSectionArea'] = np.pi*spline_r_data**2
            poly_line['BifurcationIdTmp'] = np.ones(num_points,dtype=int)*-1
            poly_line['BifurcationId'] = np.ones(num_points,dtype=int)*-1
            poly_line['BranchId'] = np.ones(num_points,dtype=int)*-1
            poly_line.point_data.set_array(normal,'CenterlineSectionNormal')
            polys.append(poly_line)
            total_outlet_area += poly_line['CenterlineSectionArea'][-1]

        for ind in range(len(polys)):
            cent_ids = np.zeros((polys[ind].n_points,len(polys)),dtype=int)
            polys[ind].point_data.set_array(cent_ids,'CenterlineId')
            polys[ind].point_data['CenterlineId'][:,ind] = 1

        bifurcation_point_ids = [] # polys ind index, polys jnd index, polys jnd point index
        for ind in range(1,len(polys)):
            current_closest_dist   = np.inf
            current_closest_branch = None
            current_closest_pt_id  = None
            for jnd in range(len(polys)):
                if jnd == ind:
                    continue
                closest_pt_id = polys[jnd].find_closest_point(polys[ind].points[0])
                closest_point = polys[jnd].points[closest_pt_id]
                closest_dist_tmp = np.linalg.norm(polys[ind].points[0] - closest_point)
                if closest_dist_tmp < current_closest_dist:
                    current_closest_branch = jnd
                    current_closest_dist   = closest_dist_tmp
                    current_closest_pt_id  = closest_pt_id
                    current_closest_point  = closest_point
            bifurcation_point_ids.append([ind,current_closest_branch,current_closest_pt_id,current_closest_point])
            polys[current_closest_branch].point_data['CenterlineId'][0:current_closest_pt_id+1,ind] = 1
            while current_closest_branch != 0:
                closest_branch = current_closest_branch
                current_closest_dist   = np.inf
                current_closest_branch = None
                current_closest_pt_id  = None
                for jnd in range(len(polys)):
                    if jnd == closest_branch:
                        continue
                    closest_pt_id = polys[jnd].find_closest_point(polys[closest_branch].points[0])
                    closest_point = polys[jnd].points[closest_pt_id]
                    closest_dist_tmp = np.linalg.norm(polys[closest_branch].points[0] - closest_point)
                    if closest_dist_tmp < current_closest_dist:
                        current_closest_branch = jnd
                        current_closest_dist   = closest_dist_tmp
                        current_closest_pt_id  = closest_pt_id
                        current_closest_point  = closest_point
                polys[current_closest_branch].point_data['CenterlineId'][0:current_closest_pt_id+1,ind] = 1

        # Determine Branch Temp Ids (CORRECT)
        branch_tmp_count = 0
        for ind in range(len(polys)):
            tmp_split = []
            for bif in bifurcation_point_ids:
                if bif[1] == ind:
                    tmp_split.append(bif[2])
            tmp_split.sort()
            tmp_split.insert(0,0)
            tmp_split.append(None)
            branch_tmp_ids = np.zeros(polys[ind].points.shape[0],dtype=int)
            for i in range(1,len(tmp_split)):
                branch_tmp_ids[tmp_split[i-1]:tmp_split[i]] = branch_tmp_count
                branch_tmp_count += 1
            polys[ind].point_data['BranchIdTmp'] = branch_tmp_ids

        # Determine BifurcationTempIds (1: bifucation point, 2: surrounding points)
        for ind in range(len(polys)):
            for id,bif in enumerate(bifurcation_point_ids):
                if bif[1] == ind:
                    rad = polys[ind].point_data['MaximumInscribedSphereRadius'][bif[2]]
                    pt  = bif[3]
                    #parent_surrounding_point_ids = np.argwhere(np.linalg.norm(polys[ind].points[:-4,:] - pt,axis=1)<rad).flatten().tolist()
                    parent_surrounding_point_ids = [bif[2]]
                    if len(parent_surrounding_point_ids) < 3:
                        if not any(np.array(parent_surrounding_point_ids) < bif[2]):
                            if bif[2] > 0:
                                parent_surrounding_point_ids.append(bif[2]-1)
                        if not any(np.array(parent_surrounding_point_ids) > bif[2]):
                            if bif[2] < polys[ind].points.shape[0] - 1:
                                parent_surrounding_point_ids.append(bif[2]+1)
                    #daughter_surrounding_point_ids = np.argwhere(np.linalg.norm(polys[bif[0]].points[:-4,:] - pt,axis=1)<rad).flatten().tolist()
                    daughter_surrounding_point_ids = []
                    if len(daughter_surrounding_point_ids) < 2:
                        if 0 not in daughter_surrounding_point_ids:
                            daughter_surrounding_point_ids.append(0)
                        if 1 not in daughter_surrounding_point_ids:
                            daughter_surrounding_point_ids.append(1)
                    parent_surrounding_point_ids.pop(parent_surrounding_point_ids.index(bif[2]))
                    parent_surrounding_point_ids = np.array(parent_surrounding_point_ids)
                    daughter_surrounding_point_ids = np.array(daughter_surrounding_point_ids)
                    polys[ind].point_data['BifurcationIdTmp'][parent_surrounding_point_ids] = 2
                    polys[ind].point_data['BifurcationIdTmp'][bif[2]] = 1
                    polys[bif[0]].point_data['BifurcationIdTmp'][daughter_surrounding_point_ids] = 2
                    polys[ind].point_data['BifurcationId'][parent_surrounding_point_ids] = id
                    polys[ind].point_data['BifurcationId'][bif[2]] = id
                    polys[bif[0]].point_data['BifurcationId'][daughter_surrounding_point_ids] = id

        # Combine bifurcation ids if there are overlaps
        #combined_bifurcations = {}
        #for ind in range(len(polys)):
        #    for jnd in range(polys[ind].n_points):
        #        if jnd == 0:
        #            if polys[ind].point_data['BifurcationIdTmp'][jnd] in list(combined_bifurcations.keys()):
        #                polys[ind].point_data['BifurcationId'][jnd] = combined_bifurcations[polys[ind].point_data['BifurcationId'][jnd]]
        #            continue
        #        if polys[ind].point_data['BifurcationIdTmp'][jnd-1] >= 0 and polys[ind].point_data['BifurcationIdTmp'][jnd] >= 0:
        #            if not polys[ind].point_data['BifurcationId'][jnd] in list(combined_bifurcations.keys()):
        #                combined_bifurcations[polys[ind].point_data['BifurcationId'][jnd]] = polys[ind].point_data['BifurcationId'][jnd-1]
        #            else:
        #                polys[ind].point_data['BifurcationId'][jnd] = combined_bifurcations[polys[ind].point_data['BifurcationId'][jnd]]

        branch_id_count = 0
        for ind in range(len(polys)):
            new = True
            for jnd in range(polys[ind].n_points):
                if polys[ind].point_data['BifurcationId'][jnd] < 0:
                    polys[ind].point_data['BranchId'][jnd] = branch_id_count
                    new = False
                elif not new and polys[ind].point_data['BifurcationId'][jnd] >= 0 and polys[ind].point_data['BifurcationId'][jnd-1] < 0: # fix this line for jnd < 0
                    branch_id_count += 1
            branch_id_count += 1

        # Set Path Values for Branches and Bifurcations also obtain outlet branches
        outlets = []
        for ind in range(len(polys)):
            branch_ids = list(set(polys[ind].point_data['BranchId'].tolist()))
            outlets.append(max(branch_ids))
            branch_ids.sort()
            path_init  = np.zeros(polys[ind].points.shape[0])
            polys[ind].point_data['Path'] = path_init
            if branch_ids[0] == -1:
                branch_ids = branch_ids[1:]
            for b_idx in branch_ids:
                poly_pt_ids = np.argwhere(polys[ind].point_data['BranchId']==b_idx).flatten()
                poly_pt_path = np.cumsum(np.insert(np.linalg.norm(np.diff(polys[ind].points[poly_pt_ids],axis=0),axis=1),0,0))
                polys[ind].point_data['Path'][poly_pt_ids] = poly_pt_path
            bif_ids = list(set(polys[ind].point_data['BifurcationId'].tolist()))
            bif_ids.sort()
            if bif_ids[0] == -1:
                bif_ids = bif_ids[1:]
            for bif_idx in bif_ids:
                poly_pt_ids = np.argwhere(polys[ind].point_data['BifurcationId']==bif_idx).flatten()
                poly_pt_path = np.cumsum(np.insert(np.linalg.norm(np.diff(polys[ind].points[poly_pt_ids],axis=0),axis=1),0,0))
                polys[ind].point_data['Path'][poly_pt_ids] = poly_pt_path

        # Set Point Ids
        Global_node_count = 0
        branch_starts = []
        for ind in range(len(polys)):
            GlobalNodeId = list(range(Global_node_count,Global_node_count+polys[ind].n_points))
            Global_node_count = GlobalNodeId[-1]+1
            if ind < len(polys):
                branch_starts.append(Global_node_count)
            GlobalNodeId = np.array(GlobalNodeId)
            polys[ind].point_data['GlobalNodeId'] = GlobalNodeId

        # Merge and Connect Lines
        centerlines_all = polys[0]
        for ind in range(1,len(polys)):
            closest_pt_id = centerlines_all.find_closest_point(polys[ind].points[0])
            closest_next_id = centerlines_all.points.shape[0]
            centerlines_all = centerlines_all.merge(polys[ind])
            new_line = [2,closest_pt_id,closest_next_id]
            centerlines_all.lines = np.hstack((centerlines_all.lines,np.array(new_line)))
        # Connect lines among spline branches
        #for branch_global_node_id in branch_starts:
        #    pt_idx = np.argwhere(centerline_all.point_data['GlobalNodeId']==branch_global_node_id)
        #    remaining_points =

        # Generate Outlet Face File
        outlet_file = open(outdir+os.sep+'outlets',"w+")
        for i in range(len(polys)):
            outlet_file.write("cap_{}\n".format(i+1))
        outlet_file.close()

        # Genrate Boundary Condition File (need to create real boundary conditions)
        total_resistance = (self.parameters['Pterm'] - 1333*distal_pressure)/self.data[0,22]
        #split = (1,0)
        rcrt_file = open(outdir+os.sep+"rcrt.dat","w+")
        rcrt_file.write("2\n")
        for i in range(len(polys)):
            inv_area = ((total_outlet_area)/(polys[i].point_data['CenterlineSectionArea'][-1]))
            rcrt_file.write("2\n")
            rcrt_file.write("cap_{}\n".format(i+1))
            rcrt_file.write("{}\n".format(total_resistance*inv_area*resistance_split[0]))
            rcrt_file.write("0.00002\n")
            rcrt_file.write("{}\n".format(total_resistance*inv_area*resistance_split[1]))
            rcrt_file.write("0.0 0.0\n")
            rcrt_file.write("1.0 0.0\n")
        rcrt_file.close()

        # Generate Inflow File
        if steady:
            flow = [self.data[0,22], self.data[0,22]]
            time = [0, 1]
            with open(outdir+os.sep+"inflow_1d.flow","w+") as file:
                for i in range(len(time)):
                    file.write("{}  {}\n".format(time[i],flow[i]))
            file.close()
        else:
            time,flow = wave(self.data[0,22],self.data[0,21]*2) # changed wave function
            time = time.tolist()
            flow = flow.tolist()
            flow[-1] = flow[0]
            period = time[-1]
            with open(outdir+os.sep+"inflow_1d.flow","w") as file:
                for i in range(len(time)):
                    file.write("{}  {}\n".format(time[i],flow[i]))
            file.close()

        # Generate 1D Solver Files
        centerlines_all.save(outdir+os.sep+'centerlines.vtp')
        param = ROM.parameters.Parameters()
        material = ROM.parameters.MaterialModel()
        param.output_directory = outdir
        param.compute_mesh = True
        param.solver_output_file = outdir + os.sep + "1d_simulation_input.json"
        param.centerlines_input_file = outdir + os.sep + "centerlines.vtp"
        param.outlet_face_names_file = outdir + os.sep + "outlets"
        param.seg_size_adaptive = True
        param.model_order = 1
        param.uniform_bc = False
        param.inflow_input_file = outdir + os.sep + "inflow_1d.flow"
        param.outflow_bc_type = ["rcrt.dat"]
        param.outflow_bc_file = outdir
        param.model_name = "1d_model_{}_vessels".format(len(polys))
        param.time_step = 0.01
        param.num_time_steps = 100
        param.olufsen_material_exponent = 2
        param.material_model = material.OLUFSEN
        param.viscosity = self.parameters['nu']
        param.density = self.parameters['rho']
        MESH = ROM.mesh.Mesh()
        centerline_data = ROM.generate_1d_mesh.read_centerlines(param)
        MESH.generate(param,centerline_data)

        # Store 1D Solver Parameters in pickle file
        param_file = open(outdir+os.sep+'params.pkl','wb+')
        pickle.dump(param,param_file)
        param_file.close()

        return centerlines_all,polys

    def export_centerlines(self,outdir=None,folder="centerlines_tmp",num_points=100):
        _,_,interp_xyzr = self.export(make=False)

        if outdir is None:
            outdir = os.getcwd()
        outdir = outdir + os.sep + folder

        if not os.path.exists(outdir):
            os.mkdir(outdir)

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

        polys = []
        for ind,spline in enumerate(interp_xyzr):
            n = np.linspace(0,1,num_points)
            spline_data = splev(n,spline[0])
            spline_data_normal = splev(n,spline[0],der=1)
            points = make_points(spline_data[0],spline_data[1],spline_data[2])
            normal = make_points(spline_data_normal[0],spline_data_normal[1],spline_data_normal[2])
            normal = normal/np.linalg.norm(normal,axis=1).reshape(-1,1)
            poly_line = lines_from_points(points)
            #poly_line['VesselId'] = np.ones(num_points,dtype=int)*ind
            poly_line['MaximumInscribedSphereRadius'] = spline_data[3]
            #poly_line['CenterlineSectionArea'] = np.pi*spline_data[3]**2
            #poly_line.point_data.set_array(normal,'CenterlineSectionNormal')
            polys.append(poly_line)

        for ind,poly in enumerate(polys):
            poly.save(outdir+os.sep+"centerline_id_{}.vtp".format(ind))
        return polys

    def collision_free(self,outside_vessels,radius_buffer=0.01):
        return no_outside_collision(self,outside_vessels,radius_buffer=radius_buffer)

    def export_gcode(self):
        interp_xyz,interp_r,interp_n,frames,branches,interp_xyzr = get_interpolated_sv_data(self.data)
        points,radii,normals    = sv_data(interp_xyzr,interp_r,radius_buffer=self.radius_buffer)
        generate_gcode(points)

class forest:
    def __init__(self,boundary=None,number_of_networks=1,
                 trees_per_network=[2],scale=None,start_points=None,
                 directions=None,root_lengths_high=None,
                 root_lengths_low=None,convex=False,compete=False):
        """
        The forest class defines some number of vascular networks which can grow
        within one given perfusion domain.

        Parameters
                   boundary: svcco.surface object
                             perfusion domain object defined by svcco.surface
                             which will serve as a boundary for all networks
                             built for this forest object

                             For more information see:
                             >>>help(svcco.surface)

                   number_of_networks: int
                             integer specifying the number of networks to be
                             built for the given forest object (default=1)

                   trees_per_network: list of ints
                             list of integers specifying the number of
                             interpenetrating trees per network. By definition
                             len(trees_per_network) = number_of_networks

                   scale:
                   start_points:
                   directions:
                   root_lengths_high:
                   root_lengths_low:
                   convex:
                   compete:

        Methods:
                   set_roots
                   add
        """
        self.networks    = []
        self.connections = []
        self.assignments = []
        self.backup      = []
        self.boundary    = boundary
        self.convex      = convex
        self.compete     = compete
        self.number_of_networks = number_of_networks
        self.trees_per_network = trees_per_network
        if isinstance(start_points,type(None)):
            start_points = [[None for j in range(trees_per_network[i])] for i in range(number_of_networks)]
        if isinstance(directions,type(None)):
            directions = [[None for j in range(trees_per_network[i])] for i in range(number_of_networks)]
        if isinstance(root_lengths_high,type(None)):
            root_lengths_high = [[None for j in range(trees_per_network[i])] for i in range(number_of_networks)]
        if isinstance(root_lengths_low,type(None)):
            root_lengths_low = [[None for j in range(trees_per_network[i])] for i in range(number_of_networks)]
        self.starting_points = start_points
        self.directions = directions
        self.root_lengths_high = root_lengths_high
        self.root_lengths_low  = root_lengths_low
        networks = []
        for idx in range(self.number_of_networks):
            network = []
            for jdx in range(self.trees_per_network[idx]):
                tmp = tree()
                tmp.set_assumptions(convex=self.convex)
                tmp.set_boundary(self.boundary)
                if self.directions[idx][jdx] is not None:
                    tmp.clamped_root = True
                network.append(tmp)
            networks.append(network)
        self.networks    = networks

    def set_roots(self,scale=None,bounds=None):
        if self.compete:
            volume = deepcopy(self.boundary.volume)
            self.boundary.volume = self.boundary.volume/self.number_of_networks
        if self.boundary is None:
            print("Need to assign boundary!")
            return
        networks = []
        connections = []
        backup = []
        for idx in range(self.number_of_networks):
            network = []
            if not isinstance(self.trees_per_network,list):
                print("trees_per_network must be list of ints"+
                      " with length equal to number_of_networks")
            for jdx in range(self.trees_per_network[idx]):
                #tmp = tree()
                tmp = self.networks[idx][jdx]
                #tmp.set_assumptions(convex=self.convex)
                #if self.directions[idx][jdx] is not None:
                #    tmp.clamped_root = True
                #tmp.set_boundary(self.boundary)
                if scale is not None:
                    tmp.parameters['Qterm'] *= scale
                if idx == 0 and jdx == 0:
                    collisions = [True]
                else:
                    collisions = [True]
                while any(collisions):
                    if bounds is None:
                        tmp.set_root(start=self.starting_points[idx][jdx],
                                     direction=self.directions[idx][jdx],
                                     limit_high=self.root_lengths_high[idx][jdx],
                                     limit_low=self.root_lengths_low[idx][jdx])
                    else:
                        print("Not implemented yet!")
                    collisions = []
                    repeat = False
                    for net in networks:
                        for t in net:
                            collisions.append(not t.collision_free(tmp.data[0,:].reshape(1,-1)))
                            if collisions[-1]:
                                repeat = True
                                break
                        if repeat:
                            break
                    for t in network:
                        collisions.append(not t.collision_free(tmp.data[0,:].reshape(1,-1)))
                        if collisions[-1]:
                            repeat = True
                            break
                    #print(collisions)
                network.append(tmp)
            networks.append(network)
            connections.append(None)
            backup.append(None)
        self.networks    = networks
        self.connections = connections
        self.backup      = backup
        if self.compete:
            self.boundary.volume = volume

    def add(self,number_of_branches,network_id=0,radius_buffer=0.01,exact=True):
        if self.compete:
            network_id = -1
        if network_id == -1:
            exit_number = []
            active_networks = list(range(len(self.networks)))
            for network in self.networks:
                exit_number.append(network[0].parameters['edge_num'] + number_of_branches)
        else:
            exit_number = []
            active_networks = [network_id]
            for nid in range(self.number_of_networks):
                if nid in active_networks:
                    exit_number.append(self.networks[nid][0].parameters['edge_num'] + number_of_branches)
                else:
                    exit_number.append(self.networks[nid][0].parameters['edge_num'])
        for AN in active_networks:
            for ATr in range(self.trees_per_network[AN]):
                self.networks[AN][ATr].rng_points,_ = self.boundary.pick(size=40*number_of_branches,homogeneous=True)
                self.networks[AN][ATr].rng_points = self.networks[AN][ATr].rng_points.tolist()
                if self.compete:
                    # new
                    #self.networks[AN][ATr].rng_points,_ = self.boundary.pick(size=len(self.boundary.tet_verts),homogeneous=True,replacement=False)
                    #self.networks[AN][ATr].rng_points = self.networks[AN][ATr].rng_points.tolist()
                    for n in tqdm(range(len(self.networks[AN][ATr].rng_points))):
                        pt = np.array(self.networks[AN][ATr].rng_points.pop(0))
                        if exact:
                            other_vessels,distances = close_exact(self.networks[AN][ATr].data,pt)
                        else:
                            other_vessels,distances = close(self.networks[AN][ATr].data,pt)
                        minimum_distance = min(distances)
                        #if minimum_distance < 4*forest.networks[nid][njd].data[other_vessels[0],21]:
                        #    continue
                        retry = False
                        for idx in active_networks:
                            for jdx in list(range(self.trees_per_network[idx])):
                                if idx == AN:
                                    continue
                                if exact:
                                    other_vessels,distances = close_exact(self.networks[idx][jdx].data,pt)
                                else:
                                    other_vessels,distances = close(self.networks[idx][jdx].data,pt)
                                if min(distances) < minimum_distance:
                                    retry = True
                                    break
                            if retry:
                                break
                        if retry:
                            continue
                        else:
                            self.networks[AN][ATr].rng_points.insert(-1,pt.tolist())
                    #new
        if not self.compete:
            while len(active_networks) > 0:
                for nid in active_networks:
                    number_of_trees = len(self.networks[nid])
                    for njd in range(number_of_trees):
                        success = False
                        threshold = None
                        while not success:
                            vessel,data,sub_division_map,sub_division_index,threshold = self.networks[nid][njd].add(-1,0,isforest=True,radius_buffer=radius_buffer,threshold=threshold)
                            new_vessels = np.vstack((data[vessel,:],data[-2,:],data[-1,:]))
                            repeat = False
                            for nikd in range(len(self.networks)):
                                if nikd == nid:
                                    check_trees = list(range(len(self.networks[nikd])))
                                    check_trees.remove(njd)
                                else:
                                    check_trees = list(range(len(self.networks[nikd])))
                                for njkd in check_trees:
                                    if not self.networks[nikd][njkd].collision_free(new_vessels,radius_buffer=radius_buffer):
                                        repeat = True
                                        break
                                if repeat:
                                    break
                            if repeat:
                                #p = self.show()
                                #p.show()
                                #print("\n\ncollision\n\n")
                                continue
                            else:
                                success = True
                        self.networks[nid][njd].data = data
                        self.networks[nid][njd].parameters['edge_num'] += 2
                        self.networks[nid][njd].sub_division_map = sub_division_map
                        self.networks[nid][njd].sub_division_index = sub_division_index
                    #print(self.networks[nid][0].parameters['edge_num'])
                    if self.networks[nid][0].parameters['edge_num'] >= exit_number[nid]:
                        active_networks.remove(nid)
        else:
            for i in tqdm(range(number_of_branches),desc="Adding branches"):
                compete_add(self,network_ids=network_id,radius_buffer=radius_buffer)

    def show(self,show=True,resolution=100,final=False,merged_trees=False,background='white',off_screen=False):
        if merged_trees:
            colors = ['r','b','g','y']
            plotter = pv.Plotter()
            merged_list = []
            for i in range(self.number_of_networks):
                for j in range(self.trees_per_network[i]):
                    merged_list.append(self.networks[i][j].export(cylinders=True))
                    plotter.add_mesh(merged_list[-1],colors[i])
            return plotter,merged_list
        model_networks = []
        for net_id,network in enumerate(self.networks):
            model_trees = []
            for tree_id,network_tree in enumerate(network):
                model = []
                for edge in range(network_tree.parameters['edge_num']):
                    #if network_tree.data[edge,15]==-1 and network_tree.data[edge,16]==-1 and edge != 0:
                    #    dis_point = (network_tree.data[edge,0:3] + network_tree.data[edge,3:6])/2
                    #    vessel_length = network_tree.data[edge,20]/2
                    #else:
                    dis_point = network_tree.data[edge,3:6]
                    vessel_length = network_tree.data[edge,20]
                    center = tuple((network_tree.data[edge,0:3] + dis_point)/2)
                    radius = network_tree.data[edge,21]
                    direction = tuple(network_tree.data[edge,12:15])
                    #vessel_length = network_tree.data[edge,20]
                    model.append(pv.Cylinder(radius=radius,height=vessel_length,
                                             center=center,direction=direction,
                                             resolution=resolution))
                if not np.any(self.connections[net_id] is None):
                    for edge in range(self.connections[net_id][tree_id].shape[0]):
                        term = self.connections[net_id][tree_id][edge,:]
                        center = tuple((term[0:3]+term[3:6])/2)
                        radius = term[21]
                        direction = term[12:15]
                        vessel_length = term[20]
                        model.append(pv.Cylinder(radius=radius,height=vessel_length,
                                                 center=center,direction=direction,
                                                 resolution=resolution))
                    #for edge in range(len(self.connections[net_id][0])):
                    #    term = network_tree.data[self.assignments[net_id][tree_id][edge],:]
                    #    conn = self.connections[net_id][0][edge]
                    #    center = tuple((term[3:6] + conn)/2)
                    #    radius = term[21]
                    #    direction = tuple(conn-term[3:6])
                    #    vessel_length = np.linalg.norm(conn-term[3:6])
                    #    model.append(pv.Cylinder(radius=radius,height=vessel_length,
                    #                             center=center,direction=direction,
                    #                             resolution=resolution))
                model_trees.append(model)
            model_networks.append(model_trees)
        """
        model_connections = []
        for connections in conn:
            if connections is None:
                continue
            else:
                for c_idx in range(connections.shape[0]):
                    center = tuple((connections[c_idx,0:3] + connections[c_idx,3:6])/2)
                    radius = connections[c_idx,21]
                    direction = tuple((connections[c_idx,3:6] - connections[c_idx,0:3])/connections[c_idx,20])
                    vessel_length = connections[c_idx,20]
                    model_conn.append(pv.Cylinder(radius=radius,height=vessel_length,
                                                  center=center,direction=direction,
                                                  resolution=resolution))
        """
        if show:
            plot = pv.Plotter(off_screen=off_screen)
            colors = ['r','b','g','y']
            plot.set_background(background)
            for model_idx,model_network in enumerate(model_networks):
                for color_idx, model_tree in enumerate(model_network):
                    for model in model_tree:
                        plot.add_mesh(model,colors[color_idx])
            #for c_model in model_conn:
            #    plot.add_mesh(c_model,'r')
                #if i == 0:
                #    plot.add_mesh(model[i],'r')
                #else:
                #    plot.add_mesh(model[i],'b')
            #path = plot.generate_orbital_path(n_points=100,viewup=(1,1,1))
            #plot.show(auto_close=False)
            #plot.open_gif('cco_gamma.gif')
            #plot.orbit_on_path(path,write_frames=True)
            #plot.close()
            return plot

    def connect(self,network_id=-1,buffer=None,curve_sample_size_min=5,curve_sample_size_max=11,curve_degree=3):
        self.forest_copy = self.copy()
        self.forest_copy.connections,self.forest_copy.assignments = connect(self.forest_copy,network_id=network_id,buffer=buffer)
        #self.forest_copy.connections,self.forest_copy.connected_forest,self.splines = smooth(self.forest_copy,curve_sample_size_min=curve_sample_size_min,curve_sample_size_max=curve_sample_size_max,curve_degree=curve_degree)
        self.forest_copy.connections,_,_ = link(self.forest_copy)

    def assign(self):
        self.connections,self.assignments = connect(self)

    def rotate(self):
        forest_copy = self.copy()
        forest_copy.assign()
        comb,pts = rotate_terminals(forest_copy)
        self.forest_copy = forest_copy
        return comb,pts

    def copy(self):
        forest_copy = forest(boundary=self.boundary,number_of_networks=self.number_of_networks,
                             trees_per_network=self.trees_per_network,scale=None,convex=self.convex,
                             compete=self.compete)
        for ndx in range(self.number_of_networks):
            for tdx in range(self.trees_per_network[ndx]):
                forest_copy.networks[ndx][tdx].data = deepcopy(self.networks[ndx][tdx].data)
                forest_copy.networks[ndx][tdx].parameters = deepcopy(self.networks[ndx][tdx].parameters)
                forest_copy.networks[ndx][tdx].sub_division_map = deepcopy(self.networks[ndx][tdx].sub_division_map)
                forest_copy.networks[ndx][tdx].sub_division_index = deepcopy(self.networks[ndx][tdx].sub_division_index)
        forest_copy.connections = deepcopy(self.connections)
        forest_copy.assignments = deepcopy(self.assignments)
        return forest_copy

    def export_solid(self,outdir=None,folder="3d_tmp",shell=False,variable_thickness=False,shell_thickness=0.01):
        if outdir is None:
            outdir = os.getcwd()+os.sep+folder
        else:
            outdir = outdir+os.sep+folder
        if not os.path.isdir(outdir):
            os.mkdir(folder)
        def polyline_from_points(pts,r):
            poly = pv.PolyData()
            poly.points = pts
            cell = np.arange(0,len(pts),dtype=np.int_)
            cell = np.insert(cell,0,len(pts))
            poly.lines = cell
            poly["radius"] = r #r/min(r)
            return poly
        final_points,final_radii,final_normals,CONNECTED_COPY,ALL_INTERP_XYZ,ALL_INTERP_RADII,_ = self.export()
        ALL_vessels = []
        ALL_tets = []
        ALL_surfs = []
        ALL_vessel_shells = []
        ALL_tet_shells = []
        ALL_surf_shells = []
        t_list = np.linspace(0,1,num=1000)
        P = self.show()
        for net_id in range(len(final_points)):
            for tree in range(self.trees_per_network[net_id]-1):
                interp_xyz = ALL_INTERP_XYZ[net_id][tree]
                interp_r = ALL_INTERP_RADII[net_id][tree]
                vessels = []
                tets = []
                surfs = []
                if shell:
                    vessel_shells = []
                    tet_shells    = []
                    surf_shells   = []
                for vessel_id in range(len(self.ALL_POINTS[net_id][tree])):
                    #x,y,z = splev(t_list,interp_xyz[vessel_id][0])
                    #_,r = splev(t_list,interp_r[vessel_id][0])
                    #points = np.zeros((len(t_list),3))
                    #points[:,0] = x
                    #points[:,1] = y
                    #points[:,2] = z
                    points = np.array(self.ALL_POINTS[net_id][tree][vessel_id])
                    r = np.array(self.ALL_RADII[net_id][tree][vessel_id])
                    #print("Points: {}".format(points))
                    #print("Radii:  {}".format(r))
                    if shell:
                        polyline_shell = polyline_from_points(points,r+shell_thickness)
                        vessel_shell   = polyline_shell.tube(radius=min(r+shell_thickness),scalars='radius',radius_factor=max(r+shell_thickness)/min(r+shell_thickness)).triangulate()
                        #vessel_shell   = remesh_surface(vessel_shell,auto=False,hausd=min(r+shell_thickness)/10,verbosity=0)
                        vessel_shells.append(vessel_shell)
                        tet_shells.append(tetgen.TetGen(vessel_shell))
                    polyline = polyline_from_points(points,r)
                    vessel = polyline.tube(radius=min(r),scalars='radius',radius_factor=max(r)/min(r)).triangulate()
                    #vessel = remesh_surface(vessel,auto=False,hausd=min(r)/10,verbosity=0)
                    vessels.append(vessel)
                    P.add_mesh(vessel,show_edges=True,opacity=0.4)
                    P.add_points(points,render_points_as_spheres=True,point_size=20)
                    tets.append(tetgen.TetGen(vessel))
                desc1 = 'Tetrahedralizing'
                desc1 = desc1+' '*(40-len(desc1))

                # Disable
                def blockPrint():
                    sys.stdout = open(os.devnull, 'w')

                # Restore
                def enablePrint():
                    sys.stdout = sys.__stdout__

                #@timeout(60)
                def tetrahedralize(tet):
                    tet.tetrahedralize(minratio=1.2)
                    return tet

                tet_success = True
                tet_fail_list = []
                tet_fail_shell_list = []
                for i in tqdm(range(len(tets)),position=0,leave=False,bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}',desc=desc1):
                    blockPrint()
                    try:
                        tets[i] = tetrahedralize(tets[i])
                    except:
                        try:
                            fixed_mesh = pymeshfix.MeshFix(vessels[i])
                            fixed_mesh.repair(verbose=False)
                            tets[i] = tetgen.TetGen(fixed_mesh.mesh)
                            tets[i] = tetrahedralize(tets[i])
                        except:
                            tet_success = False
                            tet_fail_list.append(i)
                    enablePrint()
                    if shell:
                        blockPrint()
                        try:
                            tet_shells[i] = tetrahedralize(tet_shells[i])
                        except:
                            try:
                                fixed_mesh = pymeshfix.MeshFix(vessel_shells[i])
                                fixed_mesh.repair(verbose=False)
                                tet_shells[i] = tetgen.TetGen(fixed_mesh.mesh)
                                tet_shells[i] = tetrahedralize(tet_shells[i])
                            except:
                                tet_success = False
                                tet_fail_shell_list.append(i)
                        enablePrint()
                desc2 = 'Extracting Surfaces'
                desc2 = desc2+' '*(40-len(desc2))
                surf_success = True
                surf_fail_list = []
                surf_fail_shell_list = []
                for i in tqdm(range(len(tets)),position=0,leave=False,bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}',desc=desc2):
                    try:
                        surf_fix  = pymeshfix.MeshFix(tets[i].grid.extract_surface().triangulate())
                        surf_fix.repair(verbose=False)
                        surfs.append(surf_fix.mesh)
                    except:
                        surf_success = False
                        surf_fail_list.append(i)
                    if shell:
                        try:
                            surf_shell_fix = pymeshfix.MeshFix(tet_shells[i].grid.extract_surface().triangulate())
                            surf_shell_fix.repair(verbose=False)
                            surf_shells.append(surf_shell_fix.mesh)
                        except:
                            surf_success = False
                            surf_fail_shell_list.append(i)
                ALL_vessels.append(vessels)
                ALL_tets.append(tets)
                ALL_surfs.append(surfs)
                if shell:
                    ALL_vessel_shells.append(vessel_shells)
                    ALL_tet_shells.append(tet_shells)
                    ALL_surf_shells.append(surf_shells)
                unioned = surfs[0].boolean_union(surfs[1])
                unioned = unioned.clean()
                if shell:
                    unioned_shell = surf_shells[0].boolean_union(surf_shells[1])
                    unioned_shell = unioned_shell.clean()
                    last_unioned_shell = deepcopy(unioned_shell)
                desc3 = 'Unioning'
                desc3 = desc3+' '*(40-len(desc3))
                union_success = True
                union_fail_list = []
                union_fail_shell_list = []
                last_unioned = deepcopy(unioned)
                for i in tqdm(range(2,len(surfs)),position=0,leave=False,bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}',desc=desc3):
                    try:
                        unioned = unioned.boolean_union(surfs[i])
                        unioned = unioned.clean()
                        last_unioned = deepcopy(unioned)
                    except:
                        union_success = False
                        union_fail_list.append(i)
                    unioned = last_unioned
                    if shell:
                        try:
                            unioned_shell = unioned_shell.boolean_union(surf_shells[i])
                            unioned_shell = unioned_shell.clean()
                            last_unioned_shell = deepcopy(unioned_shell)
                        except:
                            union_success = False
                            union_fail_shell_list.append(i)
                        unioned_shell = last_unioned_shell
                if union_success:
                    unioned.save(outdir+os.sep+"unioned_solid_network_{}_tree_{}.vtp".format(net_id,tree))
                    unioned.save(outdir+os.sep+"unioned_solid_network_{}_tree_{}.stl".format(net_id,tree))
                    if shell:
                        unioned_shell.save(outdir+os.sep+"unioned_solid_network_shell_{}_tree_{}.vtp".format(net_id,tree))
                        unioned_shell.save(outdir+os.sep+"unioned_solid_network_shell_{}_tree_{}.stl".format(net_id,tree))
                else:
                    unioned.save(outdir+os.sep+"partial_unioned_solid_network_{}_tree_{}.vtp".format(net_id,tree))
                    unioned.save(outdir+os.sep+"partial_unioned_solid_network_{}_tree_{}.stl".format(net_id,tree))
                    if shell:
                        unioned_shell.save(outdir+os.sep+"unioned_solid_network_shell_{}_tree_{}.vtp".format(net_id,tree))
                        unioned_shell.save(outdir+os.sep+"unioned_solid_network_shell_{}_tree_{}.stl".format(net_id,tree))
                if tet_success:
                    print('Tetrahedralize    : PASS')
                else:
                    print('Tetrahedralize    : Fail ---> {}'.format(tet_fail_list))
                if surf_success:
                    print('Surface Extraction: PASS')
                else:
                    print('Surface Extraction: Fail ---> {}'.format(surf_fail_list))
                if tet_success:
                    print('Union             : PASS')
                else:
                    print('Union             : Fail ---> {}'.format(union_fail_list))
        if shell:
            unioned_fix = pymeshfix.MeshFix(unioned)
            unioned_fix.repair(verbose=True)
            unioned = unioned_fix.mesh
            unioned.save(outdir+os.sep+"partial_unioned_solid_network_{}_tree_{}.stl".format(net_id,tree))
            unioned_shell_fix = pymeshfix.MeshFix(unioned_shell)
            unioned_shell_fix.repair(verbose=True)
            unioned_shell = unioned_shell_fix.mesh
            unioned_shell.save(outdir+os.sep+"unioned_solid_network_shell_{}_tree_{}.stl".format(net_id,tree))
            return ALL_vessels,ALL_tets,ALL_surfs,P,unioned,ALL_vessel_shells,ALL_tet_shells,ALL_surf_shells,unioned_shell
        else:
            return ALL_vessels,ALL_tets,ALL_surfs,P,unioned

    def export(self,make=False,spline=False,write_splines=False,spline_sample_points=100,steady=True,apply_distal_resistance=False,gui=False):
        ALL_POINTS  = []
        ALL_RADII   = []
        ALL_NORMALS = []
        ALL_INTERP_XYZ = []
        ALL_INTERP_RADII = []
        ALL_SPLINES = []
        CONNECTED_COPY = []
        desc1 = 'Extracting Data from Networks'
        desc1 = desc1+' '*(40-len(desc1))
        for network in tqdm(range(self.number_of_networks),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=0,leave=False,desc=desc1):
            network_points  = []
            network_radii   = []
            network_normals = []
            network_endpoints = []
            network_copy = []
            for tr in range(self.trees_per_network[network]):
                tree_copy = deepcopy(self.networks[network][tr].data)
                tree_copy_terminals = tree_copy[np.argwhere(tree_copy[:,15]==-1).flatten(),:]
                tree_copy_terminals = tree_copy_terminals[np.argwhere(tree_copy_terminals[:,16]==-1).flatten(),-1].astype(int)
                tree_copy[tree_copy_terminals,3:6] = (tree_copy[tree_copy_terminals,0:3] + tree_copy[tree_copy_terminals,3:6])/2
                tree_copy[tree_copy_terminals,20] = tree_copy[tree_copy_terminals,20]/2
                max_vessel_id = max(tree_copy[:,-1])
                conn_parents = np.argwhere(self.connections[network][tr][:,17] <= max_vessel_id).flatten()
                tree_copy[self.connections[network][tr][conn_parents,17].astype(int),15] = self.connections[network][tr][conn_parents,-1]
                tree_copy = np.vstack((tree_copy,self.connections[network][tr]))
                network_copy.append(tree_copy)
                #tmp_tree = tree()
                #tmp_tree.data = tree_copy
                #tmp_tree.show()

            for tr,net in enumerate(network_copy):
                update(net,self.networks[network][tr].parameters['gamma'],self.networks[network][tr].parameters['nu'])
                update_radii(net,self.networks[network][tr].parameters['Pperm'],self.networks[network][tr].parameters['Pterm'])
                #tmp_tree = tree()
                #tmp_tree.data = net
                #tmp_tree.show()
            #P = self.show()
            total = self.trees_per_network[network]
            desc2 = 'Extracting Data from Network {} Trees'.format(network)
            desc2 = desc2+' '*(40-len(desc2))
            with tqdm(total=total,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=1,leave=False,desc=desc2) as pbar_tree:
                for tree_copy in network_copy:
                    interp_xyz,interp_r,interp_n,frames,branches,interp_xyzr = get_interpolated_sv_data(tree_copy)
                    points,radii,normals = sv_data(interp_xyzr,interp_r,radius_buffer=0.01)
                    #for p_list in points:
                    #    P.add_points(np.array(p_list),render_points_as_spheres=True,color='g',point_size=30)
                    network_points.append(points)
                    network_radii.append(radii)
                    network_normals.append(normals)
                    pts = [p[-1] for p in points]
                    network_endpoints.append(pts)
                    #P.show()
                    pbar_tree.update(1)
            CONNECTED_COPY.append(network_copy)

            # Calculate Vessel Reordering relative to the first vessel of the network
            order = [list(range(len(network_endpoints[0])))]
            for tree in range(1,self.trees_per_network[network]):
                tmp = []
                for idx in order[0]:
                    #tmp.append(np.argwhere(np.all(np.isclose(e[0][0],network_endpoints[]),axis=1)).flatten()[0])
                    tmp.append(network_endpoints[tree].index(network_endpoints[0][idx]))
                order.append(tmp)
            # Reorder Vessels
            reordered_points  = []
            reordered_radii   = []
            reordered_normals = []
            for tree in range(self.trees_per_network[network]):
                tmp_points  = []
                tmp_radii   = []
                tmp_normals = []
                for idx in order[tree]:
                    tmp_points.append(network_points[tree][idx])
                    tmp_radii.append(network_radii[tree][idx])
                    tmp_normals.append(network_normals[tree][idx])
                reordered_points.append(tmp_points)
                reordered_radii.append(tmp_radii)
                reordered_normals.append(tmp_normals)
                #print('reorder: {}'.format(len(reordered_points[-1])))
            # Reverse and Combine
            final_points  = [reordered_points[0]]
            final_radii   = [reordered_radii[0]]
            final_normals = [reordered_normals[0]]
            final_interp_xyz = []
            final_interp_radii = []
            for tree in range(1,self.trees_per_network[network]):
                if tree == 1:
                    for vessel in range(len(reordered_points[tree])):
                        vessel_normals_flip = np.array(reordered_normals[tree][vessel])*-1
                        reordered_normals[tree][vessel] = vessel_normals_flip.tolist()
                        final_points[0][vessel].extend(list(reversed(reordered_points[tree][vessel]))[2:]) #changed
                        final_radii[0][vessel].extend(list(reversed(reordered_radii[tree][vessel]))[2:]) #changed
                        final_normals[0][vessel].extend(list(reversed(reordered_normals[tree][vessel]))[2:]) #changed 9-8-22
                        #make the connecting contours have equal average radii
                else:
                    tmp_points  = []
                    tmp_radii   = []
                    tmp_normals = []
                    for vessel in range(len(reordered_points[tree])):
                        tmp_points.append(list(reversed(reordered_points[tree][vessel])))
                        tmp_radii.append(list(reversed(reordered_radii[tree][vessel])))
                        tmp_normals.append(list(reversed(reordered_normals[tree][vessel])))
                        # make sure the radii is less than the average radii of the first vessel
                    final_points.append(tmp_points)
                    final_radii.append(tmp_radii)
                    final_normals.append(tmp_normals)
                interp_xyz  = []
                interp_r    = []
                for vessel in range(len(final_points[-1])):
                    pass
                    #P.add_points(np.array(final_points[-1][vessel]),render_points_as_spheres=True,color='g',point_size=30)
                    #interp_xyz.append(splprep(np.array(final_points[-1][vessel]).T,s=0))
                    #r = np.vstack((interp_xyz[-1][1],np.array(final_radii[-1][vessel]).T))
                    #interp_r.append(splprep(r,s=0))
                final_interp_xyz.append(interp_xyz)
                final_interp_radii.append(interp_r)
            #P.show()
            ALL_POINTS.append(final_points)
            ALL_RADII.append(final_radii)
            ALL_NORMALS.append(final_normals)
            ALL_INTERP_XYZ.append(final_interp_xyz)
            ALL_INTERP_RADII.append(final_interp_radii)
            self.ALL_POINTS = ALL_POINTS
            self.ALL_RADII = ALL_RADII
            # Export Components of Network to SV Builder for Code Generation
            if make:
                for component in range(len(final_points)):
                    if steady:
                        time = [0, 1]
                        flow = [self.networks[network][0].data[0,22], self.networks[network][0].data[0,22]]
                    else:
                        time,flow = wave(self.networks[network][0].data[0,22],self.networks[network][0].data[0,21]*2) # changed wave function
                        time = time.tolist()
                        flow = flow.tolist()
                        flow[-1] = flow[0]
                    if apply_distal_resistance:
                        R = self.networks[network][0].parameters['Pterm']/self.networks[network][0].data[0,22]
                    else:
                        R = 0
                    num_caps = self.trees_per_network[network]
                    options = file_options(num_caps,time=time,flow=flow,gui=gui,distal_resistance=R)
                    build(final_points[component],final_radii[component],final_normals[component],options)
        if spline:
            for network in range(len(ALL_POINTS)):
                network_splines = []
                if write_splines:
                    spline_file = open(os.getcwd()+os.sep+"network_{}_b_splines.txt".format(network),"w+")
                for tree in range(len(ALL_POINTS[network])):
                    for vessel in range(len(ALL_POINTS[network][tree])):
                        pt_array = np.array(ALL_POINTS[network][tree][vessel])
                        r_array  = np.array(ALL_RADII[network][tree][vessel]).reshape(-1,1)
                        pt_r_combined = deepcopy(np.hstack((pt_array,r_array)).T)
                        print(pt_r_combined.shape)
                        vessel_ctr = splprep(pt_r_combined,s=0)
                        vessel_spline = deepcopy(lambda t: splev(t,deepcopy(vessel_ctr[0])))
                        network_splines.append(deepcopy(vessel_spline))
                        if write_splines:
                            spline_file.write('Vessel: {}, Number of Points: {}\n\n'.format(vessel,spline_sample_points))
                            t = np.linspace(0,1,num=spline_sample_points)
                            data = deepcopy(vessel_spline(t))
                            for k in range(spline_sample_points):
                                spline_file.write('{}, {}, {}, {}\n'.format(data[0][k],data[1][k],data[2][k],data[3][k]))
                        spline_file.write('\n')
                spline_file.close()
                ALL_SPLINES.append(network_splines)
        return final_points,final_radii,final_normals,CONNECTED_COPY,ALL_INTERP_XYZ,ALL_INTERP_RADII,ALL_SPLINES

def perfusion_territory(tree,subdivisions=0,mesh_file=None,tree_file=None):
    terminals = tree.data[tree.data[:,15]<0,:]
    terminals = terminals[terminals[:,16]<0,:]
    territory_id = []
    territory_volumes = np.zeros(len(terminals))
    vol = tree.boundary.tet.grid.compute_cell_sizes().cell_data['Volume']
    #for idx, cell_center in tqdm(enumerate(tree.boundary.tet.grid.cell_centers().points),desc='Calculating Perfusion Territories'):
    for idx, cell_center in enumerate(tree.boundary.tet.grid.cell_centers().points):
        closest_terminal = np.argmin(np.linalg.norm(cell_center.reshape(1,-1)-terminals[:,3:6],axis=1))
        territory_id.append(closest_terminal)
        territory_volumes[closest_terminal] += vol[idx]
    territory_id = np.array(territory_id)
    tree.boundary.tet.grid['perfusion_territory_id'] = territory_id
    if mesh_file is not None:
        tree.boundary.tet.grid.save(mesh_file)
    else:
        tree.boundary.tet.grid.save('example.vtu')
    models = tree.show(show=False)
    append = vtk.vtkAppendPolyData()
    #append.UserManagedInputsOn()
    #append.SetInputDataObject(models[0])
    #append.SetNumberOfInputs(len(models))
    for i in range(len(models)):
        append.AddInputData(models[i].GetOutput())
    append.Update()
    total_tree = append.GetOutput()
    writer = vtk.vtkXMLPolyDataWriter()
    if tree_file is not None:
        writer.SetFileName(tree_file)
    else:
        writer.SetFileName(os.getcwd()+os.sep+'example_tree.vtp')
    writer.SetInputDataObject(total_tree)
    writer.Update()
    return territory_id,territory_volumes/np.sum(territory_volumes)
