import numpy as np
import numba as nb

#@nb.jit(nopython=True,cache=True,nogil=True)
def sample_triad(p1,p2,p3):
    c = (p1+p2+p3)/3
    p12 = (p1+p2)/2
    p23 = (p2+p3)/2
    p31 = (p3+p1)/2
    return np.array([[p1,p12,c],[p12,p2,c],[p2,p23,c],[p23,p3,c],[p3,p31,c],[p31,p1,c]])

#@nb.jit(nopython=True,cache=True,nogil=True)
def triangle_subdivide(triangles,i=1):
    for u in range(i):
        subdivisions = []
        for triangle in triangles:
            divisions = sample_triad(triangle[0],triangle[1],triangle[2])
            for tri in divisions:
               subdivisions.append(tri)
        triangles = np.array(subdivisions)
    return triangles

#@nb.jit(nopython=True,cache=True,nogil=True)
def unique_triangle_points(triangles):
    a,b,c = triangles.shape
    pts = triangles.reshape(a*b,c)
    return np.unique(pts,axis=0)