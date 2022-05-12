import numpy as np
from .update import update
from .calculate_radii import radii
from .calculate_length import length
from .add_depths import add_depths
from .add_flow import add_flow
from .basis import basis
import vtk

def finite_difference(data,precomputed,terminal,vessel,gamma,nu,Qterm,Pperm,Pterm):
    volumes =  []
    trials = []
    #print('parents initial: {}'.format(data[:,17]))
    for i in range(precomputed.shape[0]):
        trial_data = np.vstack((data,np.zeros((2,data.shape[1]))))
        trial_data[-1,0:3] = precomputed[i,:]
        trial_data[-2,0:3] = precomputed[i,:]
        trial_data[-2,3:6] = terminal
        trial_data[-1,3:6] = trial_data[vessel,3:6]
        trial_data[vessel,3:6] = precomputed[i,:]
        trial_data[-2,15] = -1.0
        trial_data[-2,16] = -1.0
        trial_data[-2,18] = max(trial_data[:,19]) + 1
        trial_data[-2,19] = max(trial_data[:,19]) + 2
        trial_data[-1,18] = trial_data[-2,18]
        trial_data[-1,19] = trial_data[vessel,19]
        trial_data[-1,15] = trial_data[vessel,15]
        trial_data[-1,16] = trial_data[vessel,16]
        trial_data[-2,17] = vessel
        trial_data[-1,17] = vessel
        trial_data[vessel,15] = trial_data.shape[0]-2
        trial_data[vessel,16] = trial_data.shape[0]-1
        trial_data[vessel,19] = trial_data[-2,18]
        trial_data[-2,22] = Qterm
        trial_data[-1,22] = trial_data[vessel,22]
        trial_data[-2,-1] = trial_data.shape[0] - 2
        trial_data[-1,-1] = trial_data.shape[0] - 1
        trial_data[-2,26] = trial_data[vessel,26]
        trial_data[-1,26] = trial_data[vessel,26]
        if trial_data[-1,15] >= 0:
            child = int(trial_data[-1,15])
            trial_data[child,17] = trial_data[-1,-1]
        if trial_data[-1,16] >= 0:
            child = int(trial_data[-1,16])
            trial_data[child,17] = trial_data[-1,-1]
        length(trial_data,-1)
        length(trial_data,-2)
        length(trial_data,vessel)
        basis(trial_data,-1)
        basis(trial_data,-2)
        basis(trial_data,vessel)
        add_depths(trial_data,vessel)
        updated_flows = add_flow(trial_data,vessel,Qterm)
        #print('Flow')
        #print(trial_data[:,22])
        #print('Res')
        #print(trial_data[:,25])
        #print('lengths')
        #print(trial_data[:,20])
        #print('right daughters')
        #print(trial_data[:,15])
        #print('left daughters')
        #print(trial_data[:,16])
        #print('depths')
        #print(trial_data[:,26])
        update(trial_data,gamma,nu)
        updated_radii = radii(trial_data,Pperm,Pterm)
        # calculate tree volume
        volumes.append(np.sum(np.pi*trial_data[:,21]**2*trial_data[:,20]))
        trials.append(trial_data)
    idx = np.argmin(volumes)
    flow_check = []
    #parent = vessel
    #while parent >= 0:
    #    #data[parent,22] = flow + data[parent,22]
    #    flow_check.append(parent)
    #    parent = int(data[parent,17].item())
    #print(flow_check)
    fd_data = trials[idx]
    #print('parents: {}'.format(fd_data[:,17]))
    """
    colors = vtk.vtkNamedColors()
    models = []
    actors = []
    background_color='white'
    resolution=100
    #print(updated_flows)
    for edge in range(fd_data.shape[0]):
        center = tuple((fd_data[edge,0:3] + fd_data[edge,3:6])/2)
        radius = fd_data[edge,21]
        direction = tuple(fd_data[edge,12:15])
        vessel_length = fd_data[edge,20]
        cyl = vtk.vtkTubeFilter()
        line = vtk.vtkLineSource()
        line.SetPoint1(fd_data[edge,0],fd_data[edge,1],fd_data[edge,2])
        line.SetPoint2(fd_data[edge,3],fd_data[edge,4],fd_data[edge,5])
        cyl.SetInputConnection(line.GetOutputPort())
        cyl.SetRadius(radius)
        cyl.SetNumberOfSides(resolution)
        models.append(cyl)
        mapper = vtk.vtkPolyDataMapper()
        actor  = vtk.vtkActor()
        mapper.SetInputConnection(cyl.GetOutputPort())
        actor.SetMapper(mapper)
        if edge in updated_flows and edge != vessel:
            actor.GetProperty().SetColor(colors.GetColor3d('blue'))
        elif edge == vessel:
            actor.GetProperty().SetColor(colors.GetColor3d('green'))
        else:
            actor.GetProperty().SetColor(colors.GetColor3d('red'))
        #if edge in updated_radii:
        #    actor.GetProperty().SetColor(colors.GetColor3d('yellow'))
        actors.append(actor)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d(background_color))

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('SimVascular Vessel Flow Updating')

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    for actor in actors:
        renderer.AddActor(actor)

    render_window.Render()
    interactor.Start()
    """
    return precomputed[idx,:],idx,volumes,trials[idx]
