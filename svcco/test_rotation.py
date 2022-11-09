import numpy as np
import pyvista as pv

def rotate(R0,R1,P0,P1,theta):
    theta = (theta/180)*np.pi
    Rvector = R1-R0
    Rvector = Rvector/np.linalg.norm(Rvector)
    vector = P1-P0
    vector = vector/np.linalg.norm(vector)
    vector_parallel = (np.dot(vector,Rvector.T)/np.dot(Rvector,Rvector.T))*Rvector
    vector_perp     = vector-vector_parallel
    w      = np.cross(Rvector,vector_perp)
    x1     = np.cos(theta)/np.linalg.norm(vector_perp)
    x2     = np.sin(theta)/np.linalg.norm(w)
    rotated_vector = np.linalg.norm(vector_perp)*(x1*vector_perp+x2*w)+vector_parallel
    new_point = P0+rotated_vector*(np.linalg.norm(P1-P0))
    return new_point

def line(P0,P1):
    mesh = pv.Line(P0,P1)
    return mesh

def disk(R0,R1,P0,P1):
    norm = R1-R0
    norm = norm/np.linalg.norm(norm)
    L = np.linalg.norm(P1-P0)
    mesh = pv.Disc(center=R1,normal=norm,outer=L,inner=0)
    return mesh

def test():
    R0 = np.random.random(3).reshape(1,-1)
    R1 = np.random.random(3).reshape(1,-1)
    P0 = R1
    P1 = np.random.random(3).reshape(1,-1)
    theta = 180*np.random.random(1)
    plotter = pv.Plotter()
    axis = line(R0,R1)
    original = line(P0,P1)
    P_new = rotate(R0,R1,P0,P1,theta)
    new = line(P0,P_new)
    d = disk(R0,R1,P0,P1)
    plotter.add_mesh(d,opacity=0.4)
    plotter.add_mesh(axis,color='black')
    plotter.add_mesh(original,color='r')
    plotter.add_mesh(new,color='g')
    plotter.show()
    return
