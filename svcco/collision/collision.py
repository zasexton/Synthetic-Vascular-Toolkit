import numpy as np
import numba as nb
import vtk
import time
from .sphere_proximity import *
from copy import deepcopy
from .obb import *
"""
@nb.jit(nopython=True)
def separating_axis(position,plane,U1,V1,W1,U2,V2,W2,U1_scale,
                    V1_scale,W1_scale,U2_scale,V2_scale,W2_scale):
    return np.abs(np.dot(position,plane)) > (np.abs(np.dot(U1*U1_scale,plane))+
                                             np.abs(np.dot(V1*V1_scale,plane))+
                                             np.abs(np.dot(W1*W1_scale,plane))+
                                             np.abs(np.dot(U2*U2_scale,plane))+
                                             np.abs(np.dot(V2*V2_scale,plane))+
                                             np.abs(np.dot(W2*W2_scale,plane)))
"""
"""
#@nb.jit(nopython=True)
def obb2(data,edge):
    C1 = (edge[0:3] + edge[3:6]) / 2
    C2 = (data[:,0:3] + data[:,3:6]) / 2
    Position = C2 - C1

    # Basis vector construction
    edge[12:15] = ((edge[3:6] - edge[0:3]) /
                   np.linalg.norm(edge[3:6] -
                                  edge[0:3]))
    if edge[14] == -1:
        edge[6:9] = np.array([-1,0,0])
        edge[9:12] = np.array([0,-1,0])
    else:
        edge[6:9] = np.array([1-edge[12]**2/(1+edge[14]),
                              (-edge[12]*edge[13])/(1+edge[14]),
                              -edge[12]])
        edge[9:12] = np.array([(-edge[12]*edge[13])/(1+edge[14]),
                               1 - edge[13]**2/(1+edge[14]),
                               -edge[13]])
    result = []
    for i in range(data.shape[0]):
        # U1 axis
        if separating_axis(Position[i,:],edge[6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # V1 axis
        if separating_axis(Position[i,:],edge[9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # W1 axis
        if separating_axis(Position[i,:],edge[12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # U2 axis
        if separating_axis(Position[i,:],data[i,6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # V2 axis
        if separating_axis(Position[i,:],data[i,9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # W2 axis
        if separating_axis(Position[i,:],data[i,12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # Mixed Plane Separation Tests
        # U1 X U2
        plane = np.cross(edge[6:9],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # U1 X V2
        plane = np.cross(edge[6:9],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # U1 X W2
        plane = np.cross(edge[6:9],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # V1 X U2
        plane = np.cross(edge[9:12],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # V1 X V2
        plane = np.cross(edge[9:12],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # V1 X W2
        plane = np.cross(edge[9:12],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # W1 X U2
        plane = np.cross(edge[12:15],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # W1 X V2
        plane = np.cross(edge[12:15],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        # W1 X W2
        plane = np.cross(edge[12:15],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            continue
        else:
            pass
            #return True
    return False
"""

def collision_free(data,results,idx,terminal,vessel,radius_buffer):
    # proximal = 0
    # distal   = 1
    # terminal = 2
    new_vessels = np.ones((3,data.shape[1]))*-1
    new_vessels[0,0:3] = data[vessel,0:3]
    new_vessels[0,3:6] = results[5][idx]
    new_vessels[0,-1] = data[vessel,-1]
    if vessel == 0:
        new_vessels[0,21]  = results[0][idx] + radius_buffer
    else:
        new_vessels[0,21]  = results[0][idx]*results[4][idx] + radius_buffer
        new_vessels[0,17]  = data[vessel,17]
    new_vessels[1,0:3] = results[5][idx]
    new_vessels[1,3:6] = data[vessel,3:6]
    new_vessels[1,21]  = results[0][idx]*results[3][idx] + radius_buffer
    new_vessels[1,15:17] = data[vessel,15:17]
    new_vessels[1,17]  = np.float(vessel)
    new_vessels[2,0:3] = results[5][idx]
    new_vessels[2,3:6] = terminal
    new_vessels[2,21]  = results[0][idx]*results[2][idx] + radius_buffer
    new_vessels[2,17]  = np.float(vessel)
    #print(new_vessels[:,-1])
    vessel_parent = data[vessel,17]
    vessel_child1 = data[vessel,15]
    vessel_child2 = data[vessel,16]
    vessel_sister_i = data[int(vessel_parent),15]
    vessel_sister_j = data[int(vessel_parent),16]
    for i in range(new_vessels.shape[0]):
        ##Check Self
        #start = time.time()
        n = sphere_proximity(data,new_vessels[i,:])
        #print(time.time()-start)
        if len(np.argwhere(n==vessel)) > 0:
            n = np.delete(n,np.argwhere(n==vessel).flatten())
        if len(np.argwhere(n==int(vessel_parent))) > 0:
            n = np.delete(n,np.argwhere(n==int(vessel_parent)).flatten())
        if len(np.argwhere(n==int(vessel_child1))) > 0:
            n = np.delete(n,np.argwhere(n==int(vessel_child1)).flatten())
        if len(np.argwhere(n==int(vessel_child2))) > 0:
            n = np.delete(n,np.argwhere(n==int(vessel_child2)).flatten())
        if len(np.argwhere(n==int(vessel_sister_i))) > 0:
            n = np.delete(n,np.argwhere(n==int(vessel_sister_i)).flatten())
        if len(np.argwhere(n==int(vessel_sister_j))) > 0:
            n = np.delete(n,np.argwhere(n==int(vessel_sister_j)).flatten())
        if len(np.argwhere(n==int(new_vessels[i,-1]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,-1])).flatten())
            #print('self')
        #Check Parent
        if len(np.argwhere(n==int(new_vessels[i,17]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,17])).flatten())
            #print('parent')
        #Check Children
        if len(np.argwhere(n==int(new_vessels[i,15]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,15])).flatten())
            #print('children')
        if len(np.argwhere(n==int(new_vessels[i,16]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,16])).flatten())
            #print('children')
        #Check Sister
        if int(new_vessels[i,17]) != -1:
            parent = int(new_vessels[i,17])
            if len(np.argwhere(n==int(data[parent,15]))) > 0:
                n = np.delete(n,np.argwhere(n==int(data[parent,15])).flatten())
            if len(np.argwhere(n==int(data[parent,16]))) > 0:
                n = np.delete(n,np.argwhere(n==int(data[parent,16])).flatten())
                #print('sister')
        #else:
        #    if len(np.argwhere(n==int(new_vessels[i,-1]))) > 0:
        #        n = np.delete(n,np.argwhere(n==int(new_vessels[i,-1])).flatten())

        if len(n) == 0:
            continue
        else:
            if obb(data[n,:],new_vessels[i,:]):
                #print(n)
                #print(new_vessels[:,-1])
                """
                colors = vtk.vtkNamedColors()
                models = []
                actors = []
                background_color='white'
                resolution=100
                center = tuple((new_vessels[i,0:3] + new_vessels[i,3:6])/2)
                radius = new_vessels[i,21]
                direction = tuple(new_vessels[i,12:15])
                vessel_length = new_vessels[i,20]
                cyl = vtk.vtkTubeFilter()
                line = vtk.vtkLineSource()
                line.SetPoint1(new_vessels[i,0],new_vessels[i,1],new_vessels[i,2])
                line.SetPoint2(new_vessels[i,3],new_vessels[i,4],new_vessels[i,5])
                cyl.SetInputConnection(line.GetOutputPort())
                cyl.SetRadius(radius)
                cyl.SetNumberOfSides(resolution)
                models.append(cyl)
                mapper = vtk.vtkPolyDataMapper()
                actor  = vtk.vtkActor()
                mapper.SetInputConnection(cyl.GetOutputPort())
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors.GetColor3d('green'))
                actors.append(actor)
                for edge in n:
                    center = tuple((data[edge,0:3] + data[edge,3:6])/2)
                    radius = data[edge,21]
                    direction = tuple(data[edge,12:15])
                    vessel_length = data[edge,20]
                    cyl = vtk.vtkTubeFilter()
                    line = vtk.vtkLineSource()
                    line.SetPoint1(data[edge,0],data[edge,1],data[edge,2])
                    line.SetPoint2(data[edge,3],data[edge,4],data[edge,5])
                    cyl.SetInputConnection(line.GetOutputPort())
                    cyl.SetRadius(radius)
                    cyl.SetNumberOfSides(resolution)
                    models.append(cyl)
                    mapper = vtk.vtkPolyDataMapper()
                    actor  = vtk.vtkActor()
                    mapper.SetInputConnection(cyl.GetOutputPort())
                    actor.SetMapper(mapper)
                    if edge in n:
                        #print('True')
                        actor.GetProperty().SetColor(colors.GetColor3d('blue'))
                    if edge == int(new_vessels[i,-1]):
                        print(edge)
                        actor.GetProperty().SetColor(colors.GetColor3d('green'))
                    else:
                        actor.GetProperty().SetColor(colors.GetColor3d('red'))
                    actors.append(actor)
                renderer = vtk.vtkRenderer()
                renderer.SetBackground(colors.GetColor3d(background_color))

                render_window = vtk.vtkRenderWindow()
                render_window.AddRenderer(renderer)
                render_window.SetWindowName('SimVascular Vessel Collision')

                interactor = vtk.vtkRenderWindowInteractor()
                interactor.SetRenderWindow(render_window)
                for actor in actors:
                     renderer.AddActor(actor)

                render_window.Render()
                interactor.Start()
                """
                return False
    return True

def no_outside_collision(tree,outside_vessels,radius_buffer=0):
    outside_vessels[:,21] += radius_buffer
    for i in range(outside_vessels.shape[0]):
        n = sphere_proximity(tree.data,outside_vessels[i,:])
        if len(n) == 0:
            continue
        else:
            if obb(tree.data[n,:],outside_vessels[i,:]):
                return False
    return True

def collision_free_fd(data,vessel):
    ii       = np.array(list(range(data.shape[0]-2)))
    ii       = ii[ii!=vessel]
    tree     = data[:-2,:]
    new      = data[-2:,:]
    proximal = data[vessel,:]
    new_vessels = np.vstack((new,proximal))
    for i in range(new_vessels.shape[0]):
        ##Check Self
        n = sphere_proximity(tree,new_vessels[i,:])
        if len(np.argwhere(n==int(new_vessels[i,-1]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,-1])).flatten())
        #Check Parent
        if len(np.argwhere(n==int(new_vessels[i,17]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,17])).flatten())
        #Check Children
        if len(np.argwhere(n==int(new_vessels[i,15]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,15])).flatten())
        if len(np.argwhere(n==int(new_vessels[i,16]))) > 0:
            n = np.delete(n,np.argwhere(n==int(new_vessels[i,16])).flatten())
        #Check Sister
        if int(new_vessels[i,17]) != -1:
            parent = int(new_vessels[i,17])
            if len(np.argwhere(n==int(data[parent,15]))) > 0:
                n = np.delete(n,np.argwhere(n==int(data[parent,15])).flatten())
            if len(np.argwhere(n==int(data[parent,16]))) > 0:
                n = np.delete(n,np.argwhere(n==int(data[parent,16])).flatten())
        #else:
        #    if len(np.argwhere(n==int(new_vessels[i,-1]))) > 0:
        #        n = np.delete(n,np.argwhere(n==int(new_vessels[i,-1])).flatten())

        if len(n) == 0:
            continue
        else:
            if obb(tree[n,:],new_vessels[i,:]):
                """
                colors = vtk.vtkNamedColors()
                models = []
                actors = []
                background_color='white'
                resolution=100
                for edge in range(new_vessels.shape[0]):
                    center = tuple((new_vessels[edge,0:3] + new_vessels[edge,3:6])/2)
                    radius = new_vessels[edge,21]
                    direction = tuple(new_vessels[edge,12:15])
                    vessel_length = new_vessels[edge,20]
                    cyl = vtk.vtkTubeFilter()
                    line = vtk.vtkLineSource()
                    line.SetPoint1(new_vessels[edge,0],new_vessels[edge,1],new_vessels[edge,2])
                    line.SetPoint2(new_vessels[edge,3],new_vessels[edge,4],new_vessels[edge,5])
                    cyl.SetInputConnection(line.GetOutputPort())
                    cyl.SetRadius(radius)
                    cyl.SetNumberOfSides(resolution)
                    models.append(cyl)
                    mapper = vtk.vtkPolyDataMapper()
                    actor  = vtk.vtkActor()
                    mapper.SetInputConnection(cyl.GetOutputPort())
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(colors.GetColor3d('red'))
                    actors.apend(actor)
                for edge in n:
                    center = tuple((data[edge,0:3] + data[edge,3:6])/2)
                    radius = data[edge,21]
                    direction = tuple(data[edge,12:15])
                    vessel_length = data[edge,20]
                    cyl = vtk.vtkTubeFilter()
                    line = vtk.vtkLineSource()
                    line.SetPoint1(data[edge,0],data[edge,1],data[edge,2])
                    line.SetPoint2(data[edge,3],data[edge,4],data[edge,5])
                    cyl.SetInputConnection(line.GetOutputPort())
                    cyl.SetRadius(radius)
                    cyl.SetNumberOfSides(resolution)
                    models.append(cyl)
                    mapper = vtk.vtkPolyDataMapper()
                    actor  = vtk.vtkActor()
                    mapper.SetInputConnection(cyl.GetOutputPort())
                    actor.SetMapper(mapper)
                    if edge in n:
                        actor.GetProperty().SetColor(colors.GetColor3d('blue'))
                    if edge == int(new_vessels[i,-1]):
                        actor.GetProperty().SetColor(colors.GetColor3d('green'))
                    else:
                        actor.GetProperty().SetColor(colors.GetColor3d('red'))
                    actors.append(actor)
                renderer = vtk.vtkRenderer()
                renderer.SetBackground(colors.GetColor3d(background_color))

                render_window = vtk.vtkRenderWindow()
                render_window.AddRenderer(renderer)
                render_window.SetWindowName('SimVascular Vessel Collision')

                interactor = vtk.vtkRenderWindowInteractor()
                interactor.SetRenderWindow(render_window)
                for actor in actors:
                     renderer.AddActor(actor)

                render_window.Render()
                interactor.Start()
                """
                return False
    return True


def line_distance(data,edge):
    a = np.sum((data[:,3:6]-data[:,0:3])*(data[:,3:6]-data[:,0:3]),axis=1)
    b = np.dot(data[:,3:6]-data[:,0:3],edge[3:6].T-edge[0:3].T)
    c = np.dot(edge[3:6]-edge[0:3],edge[3:6].T-edge[0:3].T)
    d = np.sum((data[:,3:6]-data[:,0:3])*(data[:,0:3]-edge[0:3]),axis=1)
    e = np.dot(edge[3:6]-edge[0:3],(data[:,0:3]-edge[0:3]).T)
    f00 = d
    f10 = f00 + a
    f01 = f00 + b
    f11 = f10 - b

    g00 = -e
    g10 = g00 - b
    g01 = g00 + c
    g11 = g10 + c

    p0 = np.zeros((len(f00),3))
    p1 = np.zeros((len(f00),3))
    dist = np.zeros((len(f00),1))
    for j in range(len(f00)):
        if a[j] > 0 and c > 0:
            s = [0,0]
            s[0] = clamped_root(a[j],f00[j],f10[j])
            s[1] = clamped_root(a[j],f01[j],f11[j])
            idd = [0,0]
            parameters = [0,0]
            for ii in range(2):
                if s[ii] <= 0:
                    idd[ii] = -1
                elif s[ii] >= 1:
                    idd[ii] = 1
                else:
                    idd[ii] = 0
            if idd[0] == -1 and idd[1] == -1:
                parameters[0] = 0
                parameters[1] = clamped_root(c,g00[j],g01[j])
            elif idd[0] == 1 and idd[0] == 1:
                parameters[0] = 1
                parameters[1] = clamped_root(c,g10[j],g11[j])
            else:
                ed,M = line_intersection(s,idd,f10=f10[j],f00=f00[j],b=b[j])
                parameters = find_minimum(ed,M,b=b[j],c=c,e=e[j],g00=g00[j],
                                          g01=g01[j],g10=g10[j],g11=g11[j])
        else:
            if a[j] > 0:
                parameters[0] = clamped_root(a[j],f00[j],f10[j])
                parameters[1] = 0
            elif c > 0:
                parameters[0] = 0
                parameters[1] = clamped_root(c,g00[j],g01[j])
            else:
                parameters[0] = 0
                parameters[0] = 0
        p0[j,:] = (1 - parameters[0])*data[j,0:3] + parameters[0]*data[j,3:6]
        p1[j,:] = (1 - parameters[1])*edge[0:3] + parameters[1]*edge[3:6]
        dist[j] = np.sqrt(np.sum(np.square(p0[j,:]-p1[j,:])))
    return p0,p1,dist

def clamped_root(slope,h0,h1):
    if h0 < 0:
        if h1 > 0:
            r = -h0/slope
            if r > 1:
                r = 0.5
        else:
            r = 1
    else:
        r = 0
    return r

def line_intersection(s,id,f10=None,f00=None,b=None):
    edge = [0,0]
    M = np.zeros((2,2))
    if id[0] < 0:
        edge[0] = 0
        M[0,0] = 0
        M[0,1] = f00/b
        if M[0,1] < 0 or M[0,1] > 1:
            M[0,1] = 0.5
        if id[1] == 0:
            edge[1] = 3
            M[1,0] = s[1]
            M[1,1] = 1
        else:
            edge[1] = 1
            M[1,0] = 1
            M[1,1] = f10/b
            if M[1,1] < 0 or M[1,1] > 1:
                M[1,1] = 0.5
    elif id[0] == 0:
        edge[0] = 2
        M[0,0] = s[0]
        M[0,1] = 0

        if id[1] < 0:
            edge[1] = 0
            M[1,0] = 0
            M[1,1] = f00/b
            if M[1,1] < 0 or M[1,1] > 1:
                M[1,1] = 0.5
        elif id[1] == 0:
            edge[1] = 3
            M[1,0] = s[1]
            M[1,1] = 1
        else:
            edge[1] = 1
            M[1,0] = 1
            M[1,1] = f10/b
            if M[1,1] < 0 or M[1,1] > 1:
                M[1,1] = 0.5
    else:
        edge[0] = 1
        M[0,0] = 1
        M[0,1] = f10/b
        if M[0,1] < 0 or M[0,1] > 1:
            M[0,1] = 0.5
        if id[1] == 0:
            edge[1] = 3
            M[1,0] = s[1]
            M[1,1] = 1
        else:
            edge[1] = 0
            M[1,0] = 0
            M[1,1] = f00/b
            if M[1,1] < 0 or M[1,1] > 1:
                M[1,1] = 0.5
    return edge, M


def find_minimum(edge,M,b=None,c=None,e=None,
                 g00=None,g01=None,g10=None,g11=None):
    delta = M[1,1] - M[0,1]
    h0 = delta * (-b * M[0,0] + c*M[0,1] - e)
    parameters = [0,0]
    if h0 >= 0:
        if edge[0] == 0:
            parameters[0] = 0
            parameters[1] = clamped_root(c,g00,g01)
        elif edge[0] == 1:
            parameters[0] = 1
            parameters[1] = clamped_root(c,g10,g11)
        else:
            parameters[0] = M[0,0]
            parameters[1] = M[0,1]
    else:
        h1 = delta*(-b * M[1,0] + c*M[1,1] - e)
        if h1 <= 0:
            if edge[1] == 0:
                parameters[0] = 0
                parameters[1] = clamped_root(c,g00,g01)
            elif edge[1] == 1:
                parameters[0] = 1
                parameters[1] = clamped_root(c,g10,g11)
            else:
                parameters[0] = M[1,0]
                parameters[1] = M[1,1]
        else:
            k = min(max(h0/(h0-h1),0),1)
            mk = 1 - k
            parameters[0] = mk*M[0,0]+k*M[1,0]
            parameters[1] = mk*M[0,1]+k*M[1,1]
    return parameters

def pairwise_tree_collisions(tree1_data,tree2_data,radius_buffer=0):
    outside_vessels = deepcopy(tree2_data)
    outside_vessels[:,21] += radius_buffer
    pairs = []
    for i in range(outside_vessels.shape[0]):
        n = sphere_proximity(tree1_data,outside_vessels[i,:])
        if len(n) == 0:
            continue
        else:
            results = obbc(tree1_data[n,:],outside_vessels[i,:])
            if len(results) == 0:
                continue
            else:
                for r in results:
                    pairs.append([r,i])
    return pairs

def fix_suggestions(tree1_data,tree2_data,radius_buffer=0):
    pairs = pairwise_tree_collisions(tree1_data,tree2_data,radius_buffer=radius_buffer)
    pair_points     = []
    pair_lengths    = []
    pair_adjustment = []
    pair_vectors    = []
    for pr in pairs:
        p0,p1,length = line_distance(tree1_data[pr[0],:].reshape(1,-1),tree2_data[pr[1],:])
        pair_points.append([p0,p1])
        pair_lengths.append(length)
        pair_adjustment.append([abs(tree1_data[pr[0],21]+tree2_data[pr[1],21]+radius_buffer)/2,
                                abs(tree1_data[pr[0],21]+tree2_data[pr[1],21]+radius_buffer)/2])
        pair_vectors.append([(p0-p1)/np.linalg.norm(p0-p1),(p1-p0)/np.linalg.norm(p1-p0)])
    return pairs,pair_points,pair_lengths,pair_adjustment,pair_vectors
