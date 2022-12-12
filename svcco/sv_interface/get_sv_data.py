from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, proj3d, art3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import vtk
from scipy import optimize
from tqdm import tqdm
import pyvista as pv

from ..branch_addition.basis import tangent_basis

from mpl_toolkits.mplot3d import art3d

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def get_longest_path(data,seed_edge):
    dig = True
    temp_edges = [seed_edge]
    while dig:
        keep_digging = []
        for edge in temp_edges:
            if data[edge,15] > -1:
                temp_edges.extend([int(data[edge,15])])
                if data[edge,16] > -1:
                    temp_edges.extend([int(data[edge,16])])
                temp_edges.remove(edge)
                keep_digging.append(True)
            else:
                keep_digging.append(False)
        dig = any(keep_digging)
    if len(temp_edges) == 1:
        return temp_edges
    edge_depths = []
    for edge in temp_edges:
        edge_depths.append(data[edge,26])
    max_depth = max(edge_depths)
    max_edge_depths = [i for i,j in enumerate(edge_depths) if j == max_depth]
    paths = [[temp_edges[i]] for i in max_edge_depths]
    path_lengths = [data[i[0],20] for i in paths]
    retrace = [True]*len(paths)
    while np.any(retrace):
        for i, path in enumerate(paths):
            if retrace[i]:
                path.insert(0,int(data[path[0],17]))
                path_lengths[i] += data[path[0],20]
                if paths[i][0] == seed_edge:
                    retrace[i] = False
    return paths[path_lengths.index(max(path_lengths))]

def get_alternate_path(data,seed_edge,reference=None):
    #print('reference: {}'.format(reference))
    if reference is None:
        reference = get_longest_path(data,seed_edge)
    else:
        pass
    seed_edge_idx = reference.index(seed_edge)
    if seed_edge_idx == len(reference)-1:
        print('Seed edge is terminal and no alternative is '+
              'possible. \n Computation finished.')
        return None
    else:
        children = [int(data[seed_edge,15]),int(data[seed_edge,16])]
        if children[0] not in reference:
            alternate_path = get_longest_path(data,children[0])
        else:
            alternate_path = get_longest_path(data,children[1])
    alternate_path.insert(0,seed_edge)
    return alternate_path

def get_branches(data):
    branches = []
    seed_edge = 0
    path = get_longest_path(data,seed_edge)
    branches.append(path)
    upcoming_evaluations = []
    upcoming_evaluations.extend(path[:-1])
    counter = [len(path[:-1])]
    idx = 0
    while len(upcoming_evaluations) > 0:
        #print(upcoming_evaluations)
        if data[upcoming_evaluations[-1],15] > -1 and data[upcoming_evaluations[-1],16] > -1:
            pass
        else:
            upcoming_evaluations.pop(-1)
            counter[idx] -= 1
            if counter[idx] == 0:
                counter[idx] = None
                for i in reversed(range(len(counter))):
                    if counter[i] is not None:
                        idx = i
                        break
            continue
        path = get_alternate_path(data,upcoming_evaluations.pop(-1),reference=branches[idx])
        counter[idx] -= 1
        if counter[idx] == 0:
            counter[idx] = None
            for i in reversed(range(len(counter))):
                if counter[i] is not None:
                    idx = i
                    break
        branches.append(path)
        if len(path) > 2:
            upcoming_evaluations.extend(path[1:-1])
            counter.append(len(path[1:-1]))
            idx = len(counter) - 1
        else:
            counter.append(None)
    return branches

def get_points(data,branches):
    path_points = []
    primed = False
    for path in branches:
        branch_points = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_points.insert(0,data[edge,3:6].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                branch_points.insert(0,data[edge,3:6].tolist())
                three_fourths = data[edge,0:3]*(1/4) + data[edge,3:6]*(3/4) #new
                branch_points.insert(0,three_fourths.tolist()) #new
                mid_point = (data[edge,0:3] + data[edge,3:6])/2
                branch_points.insert(0,mid_point.tolist())
                one_fourth = data[edge,0:3]*(3/4) + data[edge,3:6]*(1/4) #new
                branch_points.insert(0,one_fourth.tolist()) #new
                branch_points.insert(0,data[edge,0:3].tolist())
                primed = True
            elif len(branch_points) == 0:
                branch_points.insert(0,data[edge,3:6].tolist())
                three_fourths = data[edge,0:3]*(1/4) + data[edge,3:6]*(3/4) #new
                branch_points.insert(0,three_fourths.tolist()) #new
                mid_point = (data[edge,0:3] + data[edge,3:6])/2
                branch_points.insert(0,mid_point.tolist())
                one_fourth = data[edge,0:3]*(3/4) + data[edge,3:6]*(1/4) #new
                branch_points.insert(0,one_fourth.tolist()) #new
            else:
                branch_points.insert(0,data[edge,3:6].tolist())
                three_fourths = data[edge,0:3]*(1/4) + data[edge,3:6]*(3/4) #new
                branch_points.insert(0,three_fourths.tolist()) #new
                mid_point = (data[edge,0:3] + data[edge,3:6])/2
                branch_points.insert(0,mid_point.tolist())
                one_fourth = data[edge,0:3]*(3/4) + data[edge,3:6]*(1/4) #new
                branch_points.insert(0,one_fourth.tolist()) #new
        path_points.append(branch_points)
    return path_points

def get_points_v1(data,branches):
    path_points = []
    primed = False
    for path in branches:
        branch_points = []
        for edge in reversed(path):
            if edge == path[-1]:
                branch_points.insert(0,data[edge,3:6].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                branch_points.insert(0,data[edge,3:6].tolist())
                branch_points.insert(0,data[edge,0:3].tolist())
                primed = True
            else:
                branch_points.insert(0,data[edge,3:6].tolist())
        path_points.append(branch_points)
    return path_points

def get_radii(data,branches):
    path_radii = []
    primed = False
    for path in branches:
        branch_radii = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_radii.insert(0,data[path[1],21])
            elif edge == path[0] and edge == 0 and not primed:
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21]) #new
                branch_radii.insert(0,data[edge,21]) #new
                primed = True
            elif len(branch_radii) == 0:
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21]) #new
                branch_radii.insert(0,data[edge,21]) #new
            else:
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21])
                branch_radii.insert(0,data[edge,21]) #new
                branch_radii.insert(0,data[edge,21]) #new
        path_radii.append(branch_radii)
    return path_radii

def get_radii_v1(data,branches):
    path_radii = []
    primed = False
    for path in branches:
        branch_radii = []
        for edge in reversed(path):
            if edge == path[-1]:
                branch_radii.insert(0,data[edge,21])
            elif edge == path[0] and edge == 0 and not primed:
                branch_radii.insert(0,(data[edge,21]+data[previous,21])/2)
                branch_radii.insert(0,data[edge,21])
                primed = True
            else:
                branch_radii.insert(0,(data[edge,21]+data[previous,21])/2)
            previous = edge
        path_radii.append(branch_radii)
    return path_radii

def get_normals(data,branches):
    path_normals = []
    primed = False
    for path in branches:
        branch_normals = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_normals.insert(0,data[path[1],12:15].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                vector_1 = data[edge,12:15]
                mid_vector = (vector_1+vector_2)/2
                branch_normals.insert(0,mid_vector.tolist())
                branch_normals.insert(0,data[edge,12:15].tolist())
                branch_normals.insert(0,data[edge,12:15].tolist())
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                primed = True
            elif len(branch_normals) == 0:
                branch_normals.insert(0,data[edge,12:15].tolist())
                branch_normals.insert(0,data[edge,12:15].tolist())
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                vector_2 = data[edge,12:15]
            else:
                vector_1 = data[edge,12:15]
                mid_vector = (vector_1+vector_2)/2
                branch_normals.insert(0,mid_vector.tolist())
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                branch_normals.insert(0,data[edge,12:15].tolist()) #new
                branch_normals.insert(0,data[edge,12:15].tolist())
                vector_2 = data[edge,12:15]
        path_normals.append(branch_normals)
    return path_normals

def get_normals_v1(data,branches):
    path_normals = []
    primed = False
    for path in branches:
        branch_normals = []
        for edge in reversed(path):
            if edge == path[-1]:
                branch_normals.insert(0,data[edge,12:15].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                branch_normals.insert(0,((data[edge,12:15]+data[previous,12:15])/2).tolist())
                branch_normals.insert(0,data[edge,12:15].tolist())
                primed = True
            else:
                branch_normals.insert(0,((data[edge,12:15]+data[previous,12:15])/2).tolist())
            previous = edge
        path_normals.append(branch_normals)
    return path_normals

def plot_sv_data(data):
    branches = get_branches(data)
    points   = get_points(data,branches)
    radii    = get_radii(data,branches)
    normals  = get_normals(data,branches)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for path in points:
        tmp = np.array(path)
        ax.plot3D(tmp[:,0],tmp[:,1],tmp[:,2])
        ax.scatter3D(tmp[:,0],tmp[:,1],tmp[:,2])
    plt.show()

def get_interpolated_sv_data(data):
    branches = get_branches(data)
    points   = get_points(data,branches)
    #print("Points: {}".format(points[0]))
    radii    = get_radii(data,branches)
    #print("Radii:  {}".format(radii[0]))
    normals  = get_normals(data,branches)
    path_frames = []
    for idx in range(len(branches)):
        frames = []
        for jdx in range(len(points[idx])):
            frame = []
            frame.extend(points[idx][jdx])
            frame.append(radii[idx][jdx])
            frame.extend(normals[idx][jdx])
            frames.append(frame)
        path_frames.append(frames)
    interp_xyz = []
    interp_r   = []
    interp_n   = []
    interp_xyzr = []
    for idx in range(len(branches)):
        p = np.array(points[idx]).T
        r = np.array(radii[idx]).T
        n = np.array(normals[idx]).T
        if len(points[idx]) == 2:
            interp_xyz.append(splprep(p,k=1,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,k=1,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,k=1,s=0))
            #interp_n.append(splprep(n,k=1,s=0))
        elif len(points[idx]) == 3:
            interp_xyz.append(splprep(p,k=2,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,k=1,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,k=2,s=0))
            #interp_n.append(splprep(n,k=2,s=0))
        else:
            interp_xyz.append(splprep(p,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,k=1,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,s=0))
            #interp_n.append(splprep(n,s=0))

    return interp_xyz,interp_r,interp_n,path_frames,branches,interp_xyzr

def get_sv_data(network):
    if isinstance(network,tree):
        data     = network.data
        branches = get_branches(data)
        points   = get_points(data,branches)
        radii    = get_radii(data,branches)
        normals  = get_normals(data,branches)
    elif isinstance(network,forest):
        for idx, grafts in enumerate(network.grafts):
            data        = []
            branches    = []
            connections = []
            for jdx, graft in enumerate(grafts):
                data.append(graft.data)
                branches.append(get_branches(graft.data))
                connections.append(network.connections[idx][jdx])

def plot_interp_vessels(interp_xyz,interp_r,normals=True):
    n = np.linspace(0,1,200)
    fig = plt.figure()
    ax1  = fig.add_subplot(111,projection='3d')
    #ax2  = fig.add_subplot(232,projection='3d')
    #ax3  = fig.add_subplot(233,projection='3d')
    #ax4  = fig.add_subplot(234,projection='3d')
    #ax5  = fig.add_subplot(235,projection='3d')
    #ax6  = fig.add_subplot(236,projection='3d')
    for idx in enumerate(range(interp_xyz)):
        if idx > 0:
            break
        x,y,z = splev(n,interp_xyz[idx][0])
        nx,ny,nz = splev(n,interp_xyz[idx][0],der=1)
        nxx,nyy,nzz = splev(n,inerp_xyz[idx][0],der=2)
        #nxxx,nyyy,nzzz,_,_,_,_ = splev(n,b[0],der=3)
        ax1.plot3D(x,y,z,c='b')
        ax1.scatter3D(x[0],y[0],z[0],c='black')
        ax1.scatter3D(x[-1],y[-1],z[-1],c='black')
        #ax2.plot3D(x,y,z,c='b')
        #ax3.plot3D(x,y,z,c='b')
        idx_x = np.argwhere(np.diff(np.sign(nx))!=0).flatten()
        idx_y = np.argwhere(np.diff(np.sign(ny))!=0).flatten()
        idx_z = np.argwhere(np.diff(np.sign(nz))!=0).flatten()
        ax1.scatter3D(x[idx_x],y[idx_x],z[idx_x],c='r')
        ax1.scatter3D(x[idx_y],y[idx_y],z[idx_y],c='r')
        ax1.scatter3D(x[idx_z],y[idx_z],z[idx_z],c='r')
        idx_xx = np.argwhere(np.diff(np.sign(nxx))!=0).flatten()
        idx_yy = np.argwhere(np.diff(np.sign(nyy))!=0).flatten()
        idx_zz = np.argwhere(np.diff(np.sign(nzz))!=0).flatten()
        ax1.scatter3D(x[idx_xx],y[idx_xx],z[idx_xx],c='g')
        ax1.scatter3D(x[idx_yy],y[idx_yy],z[idx_yy],c='g')
        ax1.scatter3D(x[idx_zz],y[idx_zz],z[idx_zz],c='g')
        if b[0][-1] >= 3:
            nxxx,nyyy,nzzz = splev(n,interp_xyz[idx][0],der=3)
            idx_xxx = np.argwhere(np.diff(np.sign(nxxx))!=0).flatten()
            idx_yyy = np.argwhere(np.diff(np.sign(nyyy))!=0).flatten()
            idx_zzz = np.argwhere(np.diff(np.sign(nzzz))!=0).flatten()
            ax1.scatter3D(x[idx_xxx],y[idx_xxx],z[idx_xxx],c='y')
            ax1.scatter3D(x[idx_yyy],y[idx_yyy],z[idx_yyy],c='y')
            ax1.scatter3D(x[idx_zzz],y[idx_zzz],z[idx_zzz],c='y')
        #ax.plot3D(x,y,z)
        #x_knot,y_knot,z_knot,_,_,_,_ = splev(b[1],b[0])
        #ax.plot3D(x_knot,y_knot,z_knot,'go')
        """
        if normals:
            nx = -1/nx
            ny = -1/ny
            nz = -1/nz
            l  = np.linalg.norm(np.array([nx,ny,nz]),axis=0)
            nx = (nx/l)*r
            ny = (ny/l)*r
            nz = (nz/l)*r
            points0 = np.array([x,y,z]).T.reshape(-1,1,3)
            points1 = np.array([x+nx,y+ny,z+nz]).T.reshape(-1,1,3)
            segments = np.concatenate([points0,points1],axis=1)
            lc = Line3DCollection(segments,color='black')
            ax.add_collection3d(lc)
        """
    plt.show()

from sympy import Point3D,Line3D,Plane,Float
def contour_collision(x1,y1,z1,nx1,ny1,nz1,r1,x2,y2,z2,nx2,ny2,nz2,r2,radius_buffer,contours,t1,t2,n_sides=4):
    """
    d1 = x1*nx1+y1*ny1+z1*nz1
    d2 = x2*nx2+y2*ny2+z2*nz2
    length = ((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**(1/2)
    if radius_buffer is not None:
        r1 += radius_buffer*r1
        r2 += radius_buffer*r2
    if not np.isclose(nx1,0):
        alpha = (nz1*nx2-nz2*nx1)/(ny2*nx1-ny1*nx2)
        beta  = (d1*nx2-d2*nx1)/(ny2*nx1-ny1*nx2)
        c     = ny1/nx1
        a_pt  = np.array([-d1/nx1-beta*c,beta,0])
        pt1   = np.array([x1,y1,z1])
        pt2   = np.array([x2,y2,z2])
        n_line= np.array([-(alpha*c+nz1/nx1),alpha,1])
        n_line=n_line/np.linalg.norm(n_line)
        dist1 = np.linalg.norm((pt1-a_pt)-np.dot(pt1-a_pt,n_line)*n_line)
        dist2 = np.linalg.norm((pt2-a_pt)-np.dot(pt2-a_pt,n_line)*n_line)
        print('dist1: {} r1: {}'.format(dist1,r1))
        print('dist2: {} r2: {}'.format(dist2,r2))
        if dist1 < r1 or dist2 < r2:
            return True
        else:
            return False
    elif not np.isclose(ny1,0):
        alpha = (nz1*ny2-nz2*ny1)/(nx2*ny1-nx1*ny2)
        beta  = (d1*ny2-d2*ny1)/(nx2*ny1-nx1*ny2)
        c     = nx1/ny1
        a_pt  = np.array([beta,-d1/ny1-beta*c,0])
        pt1   = np.array([x1,y1,z1])
        pt2   = np.array([x2,y2,z2])
        n_line= np.array([alpha,-(alpha*c+nz1/ny1),1])
        n_line=n_line/np.linalg.norm(n_line)
        dist1 = np.linalg.norm((pt1-a_pt)-np.dot(pt1-a_pt,n_line)*n_line)
        dist2 = np.linalg.norm((pt2-a_pt)-np.dot(pt2-a_pt,n_line)*n_line)
        print('dist1: {} r1: {}'.format(dist1,r1))
        print('dist2: {} r2: {}'.format(dist2,r2))
        if dist1 < r1 or dist2 < r2:
            return True
        else:
            return False
    else:
        return None
    """
    n = np.linspace(t1,t2)
    length = ((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**(1/2)
    if radius_buffer is not None:
        r1 += r1*radius_buffer
        r2 += r2*radius_buffer
    """
    contour_1 = vtk.vtkRegularPolygonSource()
    contour_2 = vtk.vtkRegularPolygonSource()
    contour_1.SetCenter([x1,y1,z1])
    contour_2.SetCenter([x2,y2,z2])
    contour_1.SetRadius(r1)
    contour_2.SetRadius(r2)
    contour_1.SetNormal([nx1,ny1,nz1])
    contour_2.SetNormal([nx2,ny2,nz2])
    contour_1.SetNumberOfSides(n_sides)
    contour_2.SetNumberOfSides(n_sides)
    contour_11 = vtk.vtkContourTriangulator()
    contour_22 = vtk.vtkContourTriangulator()
    contour_11.SetInputConnection(contour_1.GetOutputPort())
    contour_22.SetInputConnection(contour_2.GetOutputPort())
    collision = vtk.vtkCollisionDetectionFilter()
    collision.SetInputConnection(0,contour_11.GetOutputPort())
    collision.SetInputConnection(1,contour_22.GetOutputPort())
    transform = vtk.vtkTransform()
    matrix = vtk.vtkMatrix4x4()
    collision.SetTransform(0,transform)
    collision.SetMatrix(1,matrix)
    collision.SetCollisionModeToAllContacts()
    collision.GenerateScalarsOn()
    collision.Update()
    """
    contour_1 = pv.Disc(center=[x1,y1,z1],inner=0,outer=r1,normal=[nx1,ny1,nz1],r_res=5,c_res=20).triangulate()
    contour_2 = pv.Disc(center=[x2,y2,z2],inner=0,outer=r2,normal=[nx2,ny2,nz2],r_res=5,c_res=20).triangulate()
    collision,ncol = contour_1.collision(contour_2,contact_mode=0)
    #if collision.GetNumberOfContacts() > 0:
    if ncol > 0:
        """
        actors = []
        colors = vtk.vtkNamedColors()
        mapper_1 = vtk.vtkPolyDataMapper()
        mapper_2 = vtk.vtkPolyDataMapper()
        actor_1 = vtk.vtkActor()
        actor_2 = vtk.vtkActor()
        mapper_1.SetInputConnection(contour_11.GetOutputPort())
        mapper_2.SetInputConnection(contour_22.GetOutputPort())
        actor_1.SetMapper(mapper_1)
        actor_2.SetMapper(mapper_2)
        actor_1.GetProperty().SetColor(colors.GetColor3d('blue'))
        actor_2.GetProperty().SetColor(colors.GetColor3d('blue'))
        actors.append(actor_1)
        actors.append(actor_2)
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(colors.GetColor3d('white'))

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetWindowName('Pathline with Contours')

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        for actor in actors:
            renderer.AddActor(actor)
        render_window.Render()
        interactor.Start()
        """
        return True
    else:
        return False
    """
    #print('{},{},{}'.format(x1,y1,z1))
    p1 = Point3D(Float(x1),Float(y1),Float(z1))
    p2 = Point3D(Float(x2),Float(y2),Float(z2))
    a = Plane(p1,normal_vector=(Float(nx1),Float(ny1),Float(nz1)))
    b = Plane(p2,normal_vector=(Float(nx2),Float(ny2),Float(nz2)))
    intersect = a.intersection(b)[0]
    dist1 = intersect.distance(p1)
    dist2 = intersect.distance(p2)
    p_seg1 = intersect.perpendicular_line(p1)
    p_seg2 = intersect.perpendicular_line(p2)
    pi1 = intersect.intersection(p_seg1)[0]
    pi2 = intersect.intersection(p_seg2)[0]
    line_1 = np.array([[p1.x,p1.y,p1.z],[pi1.x,pi1.y,pi1.z]])
    line_2 = np.array([[p2.x,p2.y,p2.z],[pi2.x,pi2.y,pi2.z]])
    n1_vec = np.array([[p1.x,p1.y,p1.z],[p1.x+nx1,p1.y+ny1,p1.z+nz1]])
    n2_vec = np.array([[p2.x,p2.y,p2.z],[p2.x+nx2,p2.y+ny2,p2.z+nz2]])
    #print('dist1: {} r1: {}'.format(dist1,r1))
    #print('dist2: {} r2: {}'.format(dist2,r2))
    if dist1 < r1 or dist2 < r2:
        return True
    else:
        return False
    """
def contour_check(interp_xyz,interp_r,t0,t1,radius_buffer,n_sides=4):
    tl  = (t1 - t0)/10
    x1,y1,z1,r1      = splev(t0,interp_xyz[0])
    #_,r1            = splev(t0,interp_r[0])
    nx1,ny1,nz1,_ = splev(t0,interp_xyz[0],der=1)
    n1  = np.array([nx1,ny1,nz1])
    mag = np.linalg.norm(n1,axis=0)
    n1 = n1/mag
    x2,y2,z2,r2            = splev(t1,interp_xyz[0])
    #_,r2                  = splev(t1,interp_r[0])
    nx2,ny2,nz2,_ = splev(t1,interp_xyz[0],der=1)
    n2  = np.array([nx2,ny2,nz2])
    mag = np.linalg.norm(n2,axis=0)
    n2 = n2/mag
    collision = contour_collision(x2,y2,z2,nx2,ny2,nz2,r2,x1,y1,z1,nx1,ny1,nz1,r1,radius_buffer,interp_xyz,t0,t1,n_sides=n_sides)
    return collision

def contour_check_all(interp_xyz,interp_r,radius_buffer):
    ANGLE_THRESHOLD = 20
    sample_t = []
    #n = np.linspace(0,1,1000)
    desc1 = 'Extracting Loft Contours'
    desc1 = desc1+' '*(40-len(desc1))
    for kdx in tqdm(range(len(interp_xyz)),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=2,leave=False,desc=desc1):
        tmp_t = []
        t_list = []
        t_list = np.linspace(0,1,10*len(interp_xyz[kdx][1]))
        #plot_path_contours(interp_xyz[kdx],interp_r[kdx],t_list)
        normal_list = []
        x,y,z,r = splev(0,interp_xyz[kdx][0])
        previous = [x,y,z,r]
        nx,ny,nz,_ = splev(0,interp_xyz[kdx][0],der=1)
        normal     = np.array([nx,ny,nz])
        normal     = normal/np.linalg.norm(normal)
        normal_list.append(normal)
        # Clean out all contours that provide little information
        tmp_t.append(t_list[0])
        desc2 = 'Cleaning Out Redundant Contours'
        desc2 = desc2+' '*(40-len(desc2))
        for ct in tqdm(t_list[1:-1],bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=3,leave=False,desc=desc2):
            x,y,z,r = splev(ct,interp_xyz[kdx][0])
            nx,ny,nz,_ = splev(ct,interp_xyz[kdx][0],der=1)
            normal     = np.array([nx,ny,nz])
            normal     = normal/np.linalg.norm(normal)
            angle      = abs((np.arccos(np.dot(normal_list[-1],normal.T))/np.pi)*180)
            dist       = ((previous[0]-x)**2+(previous[1]-y)**2+(previous[2]-z)**2)**(1/2)
            if angle > ANGLE_THRESHOLD or (dist>(r+previous[3])):
                normal_list.append(normal)
                tmp_t.append(ct)
                previous = [x,y,z,r]
        tmp_t.append(t_list[-1])
        #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_t)
        """
        if len(tmp_t) > 2:
            tmp_tmp_t = [tmp_t[0]]
            for ct in tqdm(tmp_t[1:-1],position=2,desc="Cleaning Consecutive Colliding Contours"):
                _,_,_,r = splev(ct,interp_xyz[kdx][0])
                if contour_check(interp_xyz[kdx],interp_r[kdx],tmp_tmp_t[-1],ct,r*0.25,n_sides=20):
                    if len(tmp_tmp_t) > 1:
                        tmp_tmp_t[-1] = (ct+tmp_tmp_t[-1])/2
                else:
                    tmp_tmp_t.append(ct)
            tmp_tmp_t.append(tmp_t[-1])
        else:
            tmp_tmp_t = tmp_t
        """
        tmp_tmp_t = tmp_t
        #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_t)
        if len(tmp_t) > 2:
            tmp_tmp_tmp_t = [tmp_tmp_t[0]]
            desc3 = 'Cleaning remaining collisions'
            desc3 = desc3+' '*(40-len(desc3))
            for ct in tqdm(tmp_tmp_t[1:-1],bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=3,leave=False,desc=desc3):
                _,_,_,r = splev(ct,interp_xyz[kdx][0])
                include = True
                for cct in tmp_tmp_t[1:-1]:
                    if cct==ct:
                        continue
                    if contour_check(interp_xyz[kdx],interp_r[kdx],ct,cct,r*0.25,n_sides=20):
                        include = False
                        break
                if include:
                    tmp_tmp_tmp_t.append(ct)
            tmp_tmp_tmp_t.append(tmp_t[-1])
        else:
            tmp_tmp_tmp_t = tmp_tmp_t
        #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_t)
        if len(tmp_tmp_t) > 2:
            repeat = True
            old_count = len(tmp_tmp_tmp_t)
            resample_count = 1
            while repeat:
                tmp_tmp_tmp_tmp_t = [tmp_tmp_tmp_t[0]]
                normal_list = []
                nx,ny,nz,_ = splev(0,interp_xyz[kdx][0],der=1)
                normal     = np.array([nx,ny,nz])
                normal     = normal/np.linalg.norm(normal)
                normal_list.append(normal)
                highlight = []
                count = 1
                swap_list = []
                total = len(tmp_tmp_tmp_t[1:-1])
                desc4 = 'Resampling {}'.format(resample_count)
                desc4 = desc4+' '*(40-len(desc4))
                with tqdm(total=total,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=3,leave=False,desc=desc4) as pbar:
                    for ct_idx,ct in enumerate(tmp_tmp_tmp_t[1:-1]):
                        t_next = tmp_tmp_tmp_t[ct_idx + 2]
                        x,y,z,r = splev(ct,interp_xyz[kdx][0])
                        nx,ny,nz,_ = splev(ct,interp_xyz[kdx][0],der=1)
                        normal     = np.array([nx,ny,nz])
                        normal     = normal/np.linalg.norm(normal)
                        angle      = abs((np.arccos(np.dot(normal_list[-1],normal.T))/np.pi)*180)
                        if angle > ANGLE_THRESHOLD+5:
                            t_op,f_op = optimize_contour(tmp_tmp_tmp_tmp_t[-1],ct,interp_xyz[kdx],interp_r[kdx],0.25,n_sides=20)
                            tnx,tny,tnz,_ = splev(t_op,interp_xyz[kdx][0],der=1)
                            next_nx,next_ny,next_nz,_ = splev(t_next,interp_xyz[kdx][0],der=1)
                            t_op_normal     = np.array([tnx,tny,tnz])
                            t_op_normal     = t_op_normal/np.linalg.norm(t_op_normal)
                            angle0      = abs((np.arccos(np.dot(normal_list[-1],t_op_normal.T))/np.pi)*180)
                            t_next_normal     = np.array([next_nx,next_ny,next_nz])
                            t_next_normal     = t_next_normal/np.linalg.norm(t_next_normal)
                            angle1      = abs((np.arccos(np.dot(t_op_normal,t_next_normal.T))/np.pi)*180)
                            if f_op < 0 and (angle0 > ANGLE_THRESHOLD*0.5 and angle1 > ANGLE_THRESHOLD*0.5):
                                #t_op = (tmp_tmp_tmp_t[-1]+ct)/2
                                tmp_tmp_tmp_tmp_t.append(t_op)
                                if count+1 < len(tmp_tmp_tmp_t)-1:
                                    swap_list.append(count+1)
                                if count-1 > 0:
                                    swap_list.append(count-1)
                                highlight.append(count)
                                count += 1
                            tmp_tmp_tmp_tmp_t.append(ct)
                            count += 1
                        else:
                            tmp_tmp_tmp_tmp_t.append(ct)
                            count += 1
                        normal_list.append(normal)
                        pbar.update(1)
                tmp_tmp_tmp_tmp_t.append(tmp_tmp_t[-1])
                #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_tmp_t,highlight=highlight)
                if len(tmp_tmp_tmp_tmp_t) == old_count:
                    repeat = False
                #else:
                #    better_t,better_idx = swap_contour(interp_xyz[kdx],tmp_tmp_tmp_tmp_t,0.25)
                #    tmp_tmp_tmp_tmp_t[better_idx] = better_t
                #    plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_tmp_t,highlight=[better_idx])
                for sw in swap_list:
                    tmp_tmp_tmp_tmp_t[sw] = swap_contour(interp_xyz[kdx],tmp_tmp_tmp_tmp_t,sw,0.25)
                #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_tmp_t,highlight=swap_list)
                tmp_tmp_tmp_tmp_t,better_idx = swap_worst_contour(interp_xyz[kdx],tmp_tmp_tmp_tmp_t,0.25,thresh=2*ANGLE_THRESHOLD)
                #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_tmp_t,highlight=better_idx)
                old_count = len(tmp_tmp_tmp_tmp_t)
                tmp_tmp_tmp_t = tmp_tmp_tmp_tmp_t
                resample_count += 1
        else:
            tmp_tmp_tmp_tmp_t = tmp_tmp_tmp_t
        #plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_tmp_tmp_tmp_t,highlight=highlight)
        sample_t.append(tmp_tmp_tmp_tmp_t)
    return sample_t
"""
ORIGINAL CONTOUR CHECKING
def contour_check_all(interp_xyz,interp_r,radius_buffer):
    sample_t = []
    n = np.linspace(0,1,200)
    total_collisions = 0
    for kdx in tqdm(range(len(interp_xyz)),desc='Resolving Overlaps'):
        tmp_t  = []
        t_list = []
        t_list = np.linspace(0,1,4*len(interp_xyz[kdx][1]))
        collision_contours = set([])
        collision_free = set([])
        checking = set(list(range(1,len(t_list)-1)))
        #t_op,f_op = optimize_contour(t_range[first],t_range[second],contours)
        for idx in checking:
            collide = False
            for jdx in checking.difference(set([idx])):
                if contour_check(interp_xyz[kdx],interp_r[kdx],t_list[idx],t_list[jdx],radius_buffer,n_sides=20):
                    collision_contours.add(idx)
                    collision_contours.add(jdx)
                    collide = True
                    break
            if not collide:
                collision_free.add(idx)
            #checking = checking.difference(collision_contours)
            checking = checking.difference(collision_free)
        collision_free = list(collision_free)
        collision_free.sort()
        if len(collision_contours) > 0:
            pass
            plot_path_contours(interp_xyz[kdx],interp_r[kdx],t_list,highlight=list(collision_contours))
        tmp_t.append(t_list[0])
        for idx in range(len(collision_free)-1):

            if collision_free[idx+1] - collision_free[idx]>1:
                tmp_tmp_t = [t_list[collision_free[idx]],t_list[collision_free[idx+1]]]
                #nx,ny,nz,_,_,_,_ = splev(tmp_tmp_t,contours,der=1)
                #normals = np.array([nx,ny,nz]).T
                #tmp_tmp_angles = [np.dot(normals[i,:],normals[i+1,:]) for i in range(normals.shape[0]-1)]
                for jdx in range(collision_free[idx+1] - collision_free[idx]):
                    #max_diff = np.argmax(np.diff(tmp_tmp_angles))
                    nx,ny,nz,_ = splev(tmp_tmp_t,interp_xyz[kdx][0],der=1)
                    normals = np.array([nx,ny,nz]).T
                    tmp_tmp_angles = [np.dot(normals[i,:],normals[i+1,:]) for i in range(normals.shape[0]-1)]
                    max_diff = np.argmin(tmp_tmp_angles)
                    t_op,f_op = optimize_contour(tmp_tmp_t[max_diff],tmp_tmp_t[max_diff+1],interp_xyz[kdx],interp_r[kdx],radius_buffer,n_sides=20)
                    if f_op < 0:
                        tmp_tmp_t.append(t_op)
                        tmp_tmp_t.sort()
                    else:
                        break
                tmp_tmp_t.pop(-1)
                tmp_t.extend(tmp_tmp_t)
            else:
                tmp_t.append(t_list[collision_free[idx]])
        tmp_t.append(t_list[collision_free[-1]])
        tmp_t.append(t_list[-1])
        if len(collision_contours) > 0:
            pass
            plot_path_contours(interp_xyz[kdx],interp_r[kdx],tmp_t)
        if len(tmp_t) == 1:
            tmp_t.append(1)
        else:
            tmp_t[-1] = 1
        sample_t.append(tmp_t)
    print('Total Collisions Resolved: '+str(total_collisions))
    return sample_t
"""

def sv_data(interp_xyz,interp_r,radius_buffer=0.25):
    points  = []
    radii   = []
    normals = []
    sample_t = contour_check_all(interp_xyz,interp_r,radius_buffer)
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    for idx in range(len(interp_xyz)):
        x,y,z,r    = splev(np.array(sample_t[idx]),interp_xyz[idx][0])
        #_,r        = splev(np.array(sample_t[idx]),interp_r[idx][0])
        nx,ny,nz,_ = splev(np.array(sample_t[idx]),interp_xyz[idx][0],der=1)
        l  = np.linalg.norm(np.array([nx,ny,nz]),axis=0)
        nx = nx/l
        ny = ny/l
        nz = nz/l
        #ax.plot3D(x,y,z)
        #for i in range(len(x)):
        #    patch = Circle((0,0),r[i],facecolor='r')
        #    ax.add_patch(patch)
        #    pathpatch_2d_to_3d(patch,z=0,normal=np.array([nx[i],ny[i],nz[i]]))
        #    pathpatch_translate(patch,(x[i],y[i],z[i]))
        tmp_points = np.zeros((len(x),3))
        tmp_points[:,0] = x
        tmp_points[:,1] = y
        tmp_points[:,2] = z
        points.append(tmp_points.tolist())
        tmp_normals = np.zeros((len(nx),3))
        tmp_normals[:,0] = nx
        tmp_normals[:,1] = ny
        tmp_normals[:,2] = nz
        radii.append(r.tolist())
        normals.append(tmp_normals.tolist())
    #plt.show()
    return points,radii,normals

def plot_frames(frames):
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    for idx,f in enumerate(frames):
        f = np.array(f)
        x,y,z = (f[:,0],f[:,1],f[:,2])
        ax.plot3D(x,y,z)
    plt.show()

def plot_branches(data,branches):
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection='3d')
    for i,branch in enumerate(branches):
        for idx in branch:
            ax.plot3D([data[idx,0],data[idx,3]],
                      [data[idx,1],data[idx,4]],
                      [data[idx,2],data[idx,5]],label=str(i))
    plt.legend()
    plt.show()

def plot_path_contours(interp_xyz,interp_r,t_list,highlight=[]):
    disks = []
    lines = []
    points = vtk.vtkPoints()
    pts = []
    radii = []
    normals = []
    actors = []
    colors = vtk.vtkNamedColors()
    for t in t_list:
        x,y,z,r = splev(t,interp_xyz[0])
        #_,r     = splev(t,interp_r[0])
        nx,ny,nz,_ = splev(t,interp_xyz[0],der=1)
        points.InsertNextPoint([x,y,z])
        radii.append(r)
        normals.append([nx,ny,nz])
        pts.append([x,y,z])
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(t_list))
    for i in range(len(t_list)):
        polyline.GetPointIds().SetId(i,i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()
    mapper.SetInputDataObject(polydata)
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('red'))
    actors.append(actor)
    for r in range(len(radii)):
        tmp_disk = vtk.vtkRegularPolygonSource()
        tmp_disk.SetRadius(radii[r])
        tmp_disk.SetCenter(pts[r])
        tmp_disk.SetNormal(normals[r])
        tmp_disk.SetNumberOfSides(100)
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        mapper.SetInputConnection(tmp_disk.GetOutputPort())
        actor.SetMapper(mapper)
        if r in highlight:
            actor.GetProperty().SetColor(colors.GetColor3d('blue'))
        else:
            actor.GetProperty().SetColor(colors.GetColor3d('red'))
        disks.append(tmp_disk)
        actors.append(actor)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('white'))

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('Pathline with Contours')

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    for actor in actors:
        renderer.AddActor(actor)
    render_window.Render()
    interactor.Start()

def optimize_contour(t0,t1,interp_xyz,interp_r,radius_buffer,t_buffer=0.1,n_sides=50):
    nx0,ny0,nz0,_ = splev(t0,interp_xyz[0],der=1)
    nx1,ny1,nz1,_ = splev(t1,interp_xyz[0],der=1)
    n0 = np.array([nx0,ny0,nz0])
    n1 = np.array([nx1,ny1,nz1])
    n0 = n0/np.linalg.norm(n0) #.reshape(-1,1)
    n1 = n1/np.linalg.norm(n1) #.reshape(-1,1)
    t_bound_buff = (t1-t0)*t_buffer
    def func(t,interp_xyz=interp_xyz,interp_r=interp_r,n0=n0,n1=n1,t0=t0,t1=t1):
        x0,y0,z0,r0 = splev(t0,interp_xyz[0])
        x1,y1,z1,r1 = splev(t1,interp_xyz[0])
        x,y,z,r = splev(t,interp_xyz[0])
        nx,ny,nz,_ = splev(t,interp_xyz[0],der=1)
        n = np.array([nx,ny,nz])
        n = n/np.linalg.norm(n) #.reshape(-1,1)
        #print(-np.dot(n0,n)*np.dot(n1,n))
        n_avg = (n0+n1)/2
        a = np.dot(n_avg,n)
        first_angle = np.arccos(np.dot(n0,n))
        second_angle = np.arccos(np.dot(n1,n))
        #l0 = ((x0-x)**(2)+(y0-y)**(2)+(z0-z)**(2))**(1/2)
        #l1 = ((x1-x)**(2)+(y1-y)**(2)+(z1-z)**(2))**(1/2)
        mx  = (x0+x1)/2
        my  = (y0+y1)/2
        mz  = (z0+z1)/2
        lm1 = ((x0-x)**(2)+(y0-y)**(2)+(z0-z)**(2))**(1/2)
        lm2 = ((x1-x)**(2)+(y1-y)**(2)+(z1-z)**(2))**(1/2)
        #d   = (((x1-x0)**(2)+(y1-y0)**(2)+(z1-z0)**(2))**(1/2))/2
        #d   = d/2
        lm = abs(lm1-lm2)/(lm1+lm2)
        c = 0
        if contour_check(interp_xyz,interp_r,t0,t,radius_buffer=radius_buffer,n_sides=n_sides):
            c = 100
        if contour_check(interp_xyz,interp_r,t1,t,radius_buffer=radius_buffer,n_sides=n_sides):
            c += 100
        if np.sin(first_angle)*(r0+r) > lm1:
            c += 100
        if np.sin(second_angle)*(r1+r) > lm2:
            c += 100
        #return -(a*b+(1-abs(a-b))) + c # + abs(l0-l1)
        return -(a) + c + lm/100
    result = optimize.minimize_scalar(func,bounds=(t0+t_bound_buff,t1-t_bound_buff),method='bounded')
    return result.x,result.fun

def swap_worst_contour(interp_xyz,t_list,radius_buffer,thresh=20):
    normal_list = []
    nx,ny,nz,_ = splev(0,interp_xyz[0],der=1)
    normal     = np.array([nx,ny,nz])
    normal     = normal/np.linalg.norm(normal)
    normal_list.append(normal)
    #next_nx,next_ny,next_nz,_ = splev(t_list[1],interp_xyz[kdx][0],der=1)
    #next_normal     = np.array([next_nx,next_ny,next_nz])
    #next_normal     = next_normal/np.linalg.norm(next_normal)
    #angle = abs((np.arccos(np.dot(normal_list[-1],next_normal.T))/np.pi)*180)
    #scores = [angle]
    scores = []
    total = len(t_list[1:-1])
    desc5 = 'Computing swap scores'
    desc5 = desc5+' '*(40-len(desc5))
    with tqdm(total=total,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',position=4,leave=False,desc=desc5) as pbar:
        for idx,ct in enumerate(t_list[1:-1]):
            next_t            = t_list[idx+2]
            nx,ny,nz,_        = splev(ct,interp_xyz[0],der=1)
            normal            = np.array([nx,ny,nz])
            normal            = normal/np.linalg.norm(normal)
            angle0            = abs((np.arccos(np.dot(normal_list[-1],normal.T))/np.pi)*180)
            next_nx,next_ny,next_nz,_ = splev(next_t,interp_xyz[0],der=1)
            t_next_normal     = np.array([next_nx,next_ny,next_nz])
            t_next_normal     = t_next_normal/np.linalg.norm(t_next_normal)
            angle1            = abs((np.arccos(np.dot(normal,t_next_normal.T))/np.pi)*180)
            scores.append(angle0+angle1)
            normal_list.append(normal)
            pbar.update(1)
    worst_idx = np.argwhere(np.array(scores)>thresh).flatten()+1
    worst_idx = worst_idx.tolist()
    for wi in worst_idx:
        t   = t_list[wi]
        t0,f0 = optimize_contour(t_list[wi-1],t_list[wi+1],interp_xyz,None,radius_buffer,t_buffer=0.1,n_sides=50)
        if f0 < 0:
            t_list[wi] = t0
    return t_list,worst_idx

def swap_contour(interp_xyz,t_list,c_idx,radius_buffer):
    value = t_list[c_idx]
    t0,f0 = optimize_contour(t_list[c_idx-1],t_list[c_idx+1],interp_xyz,None,radius_buffer,t_buffer=0.1,n_sides=50)
    if f0 < 0:
        value = t0
    return value

def truncate(data,radius=None,indicies=None):
    if radius is None and indicies is None:
        radius = np.median(data[:,21])
    branches = get_branches(data)
    include = []
    exclude = []
    for branch in branches:
        tmp_include = []
        tmp_exclude = []
        for idx,vdx in enumerate(branch):
            if indicies is None:
                if data[vdx,21] >= radius:
                    tmp_include.append(vdx)
                else:
                    tmp_exclude.append(vdx)
            else:
                if vdx in indicies:
                    tmp_include.append(vdx)
                else:
                    tmp_exclude.append(vdx)
        if len(tmp_include) > 1:
            include.append(tmp_include)
        if len(tmp_exclude) > 0:
            exclude.append(tmp_exclude)
    return include, exclude

def get_truncated_interpolated_sv_data(data,radius=None,indicies=None):
    branches,_ = truncate(data,radius=radius,indicies=indicies)
    points   = get_points(data,branches)
    #print("Points: {}".format(points[0]))
    radii    = get_radii(data,branches)
    #print("Radii:  {}".format(radii[0]))
    normals  = get_normals(data,branches)
    path_frames = []
    for idx in range(len(branches)):
        frames = []
        for jdx in range(len(points[idx])):
            frame = []
            frame.extend(points[idx][jdx])
            frame.append(radii[idx][jdx])
            frame.extend(normals[idx][jdx])
            frames.append(frame)
        path_frames.append(frames)
    interp_xyz = []
    interp_r   = []
    interp_n   = []
    interp_xyzr = []
    for idx in range(len(branches)):
        p = np.array(points[idx]).T
        r = np.array(radii[idx]).T
        n = np.array(normals[idx]).T
        if len(points[idx]) == 2:
            interp_xyz.append(splprep(p,k=1,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,k=1,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,k=1,s=0))
            #interp_n.append(splprep(n,k=1,s=0))
        elif len(points[idx]) == 3:
            interp_xyz.append(splprep(p,k=2,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,k=2,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,k=2,s=0))
            #interp_n.append(splprep(n,k=2,s=0))
        else:
            interp_xyz.append(splprep(p,s=0))
            rr = np.vstack((interp_xyz[-1][1],r))
            interp_r.append(splprep(rr,s=0))
            xyzr = np.vstack((p,r))
            interp_xyzr.append(splprep(xyzr,s=0))
            #interp_n.append(splprep(n,s=0))

    return interp_xyz,interp_r,interp_n,path_frames,branches,interp_xyzr
