import numpy as np
import numba as nb
import pyvista as pv

class grid:
    def __init__(self):
        self.P = pv.Plotter(shape=(4,4))
        self.next = [0,0]
        self.size = [4,4]

    def plot(self,cyl1,cyl2,plane=None,title="None"):
        self.P.subplot(self.next[0],self.next[1])
        self.P.add_mesh(cyl1,color='g')
        self.P.add_mesh(cyl2)
        if plane is not None:
            self.P.add_mesh(plane)
        self.P.add_title(title)
        if self.next[1]+1 < self.size[1]:
            self.next[1] += 1
        else:
            self.next[0] += 1
            self.next[1] = 0

    def show(self):
        self.P.show()

@nb.jit(nopython=True)
def separating_axis(position,plane,U1,V1,W1,U2,V2,W2,U1_scale,
                    V1_scale,W1_scale,U2_scale,V2_scale,W2_scale):
    return np.abs(np.dot(position,plane)) > (np.abs(np.dot(U1*U1_scale,plane))+
                                             np.abs(np.dot(V1*V1_scale,plane))+
                                             np.abs(np.dot(W1*W1_scale,plane))+
                                             np.abs(np.dot(U2*U2_scale,plane))+
                                             np.abs(np.dot(V2*V2_scale,plane))+
                                             np.abs(np.dot(W2*W2_scale,plane)))

@nb.jit(nopython=True)
def obb(data,edge):
    #C2 = (edge[0:3] + edge[3:6]) / 2
    #C1 = (data[:,0:3] + data[:,3:6]) / 2
    """
    start = time.time()
    collision = False
    start = time.time()
    cyl_2 = vtk.vtkTubeFilter()
    line2 = vtk.vtkLineSource()
    line2.SetPoint1(edge[0],edge[1],edge[2])
    line2.SetPoint2(edge[3],edge[4],edge[5])
    cyl_2.SetInputConnection(line2.GetOutputPort())
    cyl_2.SetRadius(edge[21])
    cyl_2.SetNumberOfSides(20)
    cyl_2.CappingOn()
    for i in range(data.shape[0]):
        cyl_1 = vtk.vtkTubeFilter()
        line1 = vtk.vtkLineSource()
        line1.SetPoint1(data[i,0],data[i,1],data[i,2])
        line1.SetPoint2(data[i,3],data[i,4],data[i,5])
        cyl_1.SetInputConnection(line1.GetOutputPort())
        cyl_1.SetRadius(data[i,21])
        cyl_1.SetNumberOfSides(20)
        cyl_1.CappingOn()
        collision = vtk.vtkCollisionDetectionFilter()
        collision.SetInputConnection(0,cyl_1.GetOutputPort())
        collision.SetInputConnection(1,cyl_2.GetOutputPort())
        transform = vtk.vtkTransform()
        matrix = vtk.vtkMatrix4x4()
        collision.SetTransform(0,transform)
        collision.SetMatrix(1,matrix)
        collision.SetCollisionModeToAllContacts()
        collision.GenerateScalarsOn()
        collision.Update()
        if collision.GetNumberOfContacts() > 0:
            collision = True
            break
    #print('OBB: {}'.format(time.time()-start))
    end = time.time() - start
    #print(end)
    return collision
    """
    C1 = (edge[0:3] + edge[3:6]) / 2
    C2 = (data[:,0:3] + data[:,3:6]) / 2
    Position = C2 - C1
    #CENTER_ALL = (C2 + C1)/2
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
    edge[20] = np.linalg.norm(edge[3:6] - edge[0:3])
    #print(np.linalg.norm(edge[6:9]))
    #print(np.linalg.norm(edge[9:12]))
    #print(np.linalg.norm(edge[12:15]))
    result = []
    for i in range(data.shape[0]):
        #CENTER = CENTER_ALL[i]
        #G = grid()
        # U1 axis
        #cyl1 = pv.Cylinder(center=(data[i,0:3]+data[i,3:6])/2,direction=data[i,12:15],
        #                   radius=data[i,21],height=data[i,20])
        #cyl2 = pv.Cylinder(center=(edge[0:3]+edge[3:6])/2,direction=edge[12:15],
        #                   radius=edge[21],height=edge[20])
        #G.plot(cyl1,cyl2,plane=None,title="Objects")
        #plane = pv.Plane(center=CENTER,direction=edge[6:9])
        #G.plot(cyl1,cyl2,plane=plane,title="U1 axis")
        if separating_axis(Position[i,:],edge[6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # V1 axis
        #plane = pv.Plane(center=CENTER,direction=edge[9:12])
        #G.plot(cyl1,cyl2,plane=plane,title="V1 axis")
        if separating_axis(Position[i,:],edge[9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # W1 axis
        #plane = pv.Plane(center=CENTER,direction=edge[12:15])
        #G.plot(cyl1,cyl2,plane=plane,title="W1 axis")
        if separating_axis(Position[i,:],edge[12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # U2 axis
        #plane = pv.Plane(center=CENTER,direction=data[i,6:9])
        #G.plot(cyl1,cyl2,plane=plane,title="U2 axis")
        if separating_axis(Position[i,:],data[i,6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # V2 axis
        #plane = pv.Plane(center=CENTER,direction=data[i,9:12])
        #G.plot(cyl1,cyl2,plane=plane,title="V2 axis")
        if separating_axis(Position[i,:],data[i,9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # W2 axis
        #plane_obj = pv.Plane(center=CENTER,direction=data[i,12:15])
        #G.plot(cyl1,cyl2,plane=plane,title="W2 axis")
        if separating_axis(Position[i,:],data[i,12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # Mixed Plane Separation Tests
        # U1 X U2
        plane = np.cross(edge[6:9],data[i,6:9])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="U1 x U2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # U1 X V2
        plane = np.cross(edge[6:9],data[i,9:12])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="U1 x V2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # U1 X W2
        plane = np.cross(edge[6:9],data[i,12:15])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="U1 x W2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # V1 X U2
        plane = np.cross(edge[9:12],data[i,6:9])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="V1 x U2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # V1 X V2
        plane = np.cross(edge[9:12],data[i,9:12])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="V1 x V2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # V1 X W2
        plane = np.cross(edge[9:12],data[i,12:15])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="V1 x W2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # W1 X U2
        plane = np.cross(edge[12:15],data[i,6:9])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="W1 x U2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # W1 X V2
        plane = np.cross(edge[12:15],data[i,9:12])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="W1 x V2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        # W1 X W2
        plane = np.cross(edge[12:15],data[i,12:15])
        #plane_obj = pv.Plane(center=CENTER,direction=plane)
        #G.plot(cyl1,cyl2,plane=plane_obj,title="W1 x W2 axis")
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            result.append(False)
            #G.show()
            continue
        else:
            #print('Collision')
            #G.show()
            """
            colors = vtk.vtkNamedColors()
            background_color = 'white'
            cyl = vtk.vtkTubeFilter()
            line = vtk.vtkLineSource()
            line.SetPoint1(edge[0],edge[1],edge[2])
            line.SetPoint2(edge[3],edge[4],edge[5])
            cyl.SetInputConnection(line.GetOutputPort())
            cyl.SetRadius(edge[21])
            cyl.SetNumberOfSides(100)
            #models.append(cyl)
            mapper = vtk.vtkPolyDataMapper()
            actor1  = vtk.vtkActor()
            mapper.SetInputConnection(cyl.GetOutputPort())
            actor1.SetMapper(mapper)
            actor1.GetProperty().SetColor(colors.GetColor3d('green'))
            #actors.append(actor)
            colors = vtk.vtkNamedColors()
            cyl = vtk.vtkTubeFilter()
            line = vtk.vtkLineSource()
            line.SetPoint1(data[i,0],data[i,1],data[i,2])
            line.SetPoint2(data[i,3],data[i,4],data[i,5])
            cyl.SetInputConnection(line.GetOutputPort())
            cyl.SetRadius(data[i,21])
            cyl.SetNumberOfSides(100)
            #models.append(cyl)
            mapper = vtk.vtkPolyDataMapper()
            actor2  = vtk.vtkActor()
            mapper.SetInputConnection(cyl.GetOutputPort())
            actor2.SetMapper(mapper)
            actor2.GetProperty().SetColor(colors.GetColor3d('green'))
            #actors.append(actor)
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(colors.GetColor3d(background_color))

            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            render_window.SetWindowName('SimVascular Vessel Collision')

            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
            renderer.AddActor(actor1)
            renderer.AddActor(actor2)
            render_window.Render()
            interactor.Start()
            """
            return True

    return False

@nb.jit(nopython=True)
def obbc(data,edge):
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
    edge[20] = np.linalg.norm(edge[3:6] - edge[0:3])
    result = []
    for i in range(data.shape[0]):
        # U1 axis
        #cyl1 = pv.Cylinder(center=(data[i,0:3]+data[i,3:6])/2,direction=data[i,12:15],
        #                   radius=data[i,21],height=data[i,20])
        #cyl2 = pv.Cylinder(center=(edge[0:3]+edge[3:6])/2,direction=edge[12:15],
        #                   radius=edge[21],height=edge[20])
        if separating_axis(Position[i,:],edge[6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # V1 axis
        if separating_axis(Position[i,:],edge[9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # W1 axis
        if separating_axis(Position[i,:],edge[12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # U2 axis
        if separating_axis(Position[i,:],data[i,6:9],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # V2 axis
        if separating_axis(Position[i,:],data[i,9:12],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # W2 axis
        if separating_axis(Position[i,:],data[i,12:15],edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # Mixed Plane Separation Tests
        # U1 X U2
        plane = np.cross(edge[6:9],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # U1 X V2
        plane = np.cross(edge[6:9],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # U1 X W2
        plane = np.cross(edge[6:9],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # V1 X U2
        plane = np.cross(edge[9:12],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # V1 X V2
        plane = np.cross(edge[9:12],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # V1 X W2
        plane = np.cross(edge[9:12],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # W1 X U2
        plane = np.cross(edge[12:15],data[i,6:9])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # W1 X V2
        plane = np.cross(edge[12:15],data[i,9:12])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        # W1 X W2
        plane = np.cross(edge[12:15],data[i,12:15])
        if separating_axis(Position[i,:],plane,edge[6:9],edge[9:12],edge[12:15],
                           data[i,6:9],data[i,9:12],data[i,12:15],edge[21],edge[21],
                           edge[20]/2,data[i,21],data[i,21],data[i,20]/2):
            continue
        else:
            result.append(i)
    return result
