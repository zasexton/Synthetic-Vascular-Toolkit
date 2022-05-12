import numpy as np
import numba as nb
from scipy import interpolate

def get_local_points(data,vessel,terminal,num,isclamped):
    proximal = data[vessel,0:3]
    distal   = data[vessel,3:6]
    direction = distal - proximal
    if np.all(vessel == 0) and isclamped:
        length = data[vessel,20]
        shift = direction.reshape(-1,1)*length*np.linspace(0,1,num)
        points = proximal.reshape(1,-1) + shift.T
    else:
        line   = np.linspace(0,1,num)
        s,t    = np.meshgrid(line,line)
        s      = s.flatten().reshape(-1,1)
        t      = t.flatten().reshape(-1,1)
        points = proximal.reshape(1,-1)*(1-t)*s + \
                   distal.reshape(1,-1)*(t*s)+\
                   terminal.reshape(1,-1)*(1-s)
        #print('SHAPE: {}'.format(points.shape))
    return points

"""
@nb.jit(nopython=True,cache=True)
def angle_constraint(p0,p1,p2,constraint,greater):
    tmp = []
    for i in range(p0.shape[0]):
        vector_1 = p1 - p0[i]
        vector_2 = p2 - p0[i]
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = vector_2 / np.linalg.norm(vector_2)
        if greater:
            if np.dot(vector_1,vector_2) > constraint:
                tmp.append(p0[i])
        else:
            if np.dot(vector_1,vector_2) < constraint:
                tmp.append(p0[i])
    return tmp
"""

def angle_constraint(p0,p1,p2,constraint,greater):
    vector_1 = p1 - p0
    vector_2 = p2 - p0
    vector_1 = vector_1 / np.linalg.norm(vector_1,axis=1).reshape(-1,1)
    vector_2 = vector_2 / np.linalg.norm(vector_2,axis=1).reshape(-1,1)
    angle    = np.array([np.dot(vector_1[i],vector_2[i]) for i in range(len(vector_1))])
    if greater:
        p0   = p0[angle>constraint,:]
    else:
        p0   = p0[angle<constraint,:]
    return p0


@nb.jit(nopython=True,cache=True)
def relative_length_constraint(p0,p1,p2,p3,ratio):
    d_max    = min([np.linalg.norm(p1-p2),
                    np.linalg.norm(p1-p3),
                    np.linalg.norm(p2-p3)])*ratio
    tmp = []
    for i in range(p0.shape[0]):
        if not np.linalg.norm(p1-p0[i])>d_max:
            continue
        if not np.linalg.norm(p2-p0[i])>d_max:
            continue
        if not np.linalg.norm(p3-p0[i])>d_max:
            continue
        tmp.append(p0[i])
    #p0 = np.array(tmp)
    #p0 = p0[np.linalg.norm(p1-p0,axis=1)>d_max]
    #p0 = p0[np.linalg.norm(p2-p0,axis=1)>d_max]
    #p0 = p0[np.linalg.norm(p3-p0,axis=1)>d_max]
    return tmp

def boundary_constraint(p0,boundary,patches):
    points = []
    for idx in range(p0.shape[0]):
        if boundary.within(p0[idx,0],p0[idx,1],p0[idx,2],patches):
            points.append(p0[idx,:])
    return np.array(points)

def absolute_length_constraint():
    pass

"""
class triangle:
    def __init__(self,level,ct,p0,p1,p2,p0_val,p1_val,degree=2,nodes=None,target=None):
        self.degree = degree
        if np.all(nodes) is not None:
            self.nodes = nodes
            self.targets = np.ones(self.nodes.shape[0])*target
        else:
            self.nodes,self.targets = self.sample(level,ct,degree,p0,p1,p2,p0_val,p1_val)

    def sample(self,level,ct,degree,p0,p1,p2,p0_val,p1_val):
        if degree == 1:
            nodes = []
            pts = np.array([p0,p1,p2])
            #for pt in pts:
            #    nodes.append(level(*pt,ct).x)
            nodes = pts
        elif degree == 2:
            nodes = []
            pts = np.array([p0,(p0+p1)/2,(p0+p2)/2,p1,(p1+p2)/2,p2])
            tar = np.array([p0_val,(p0_val+p1_val)/2,(p0_val+ct)/2,p1_val,(p1_val+ct)/2,ct])
            for idx,pt in enumerate(pts):
                if idx == 0 or idx==3 or idx == 5:
                    nodes.append(pt)
                else:
                    nodes.append(level(pt[0],pt[1],pt[2],tar[idx]).x)
        elif degree == 3:
            nodes = []
            pts = np.array([p0,(2*p0+p1)/3,(2*p0+p2)/3,(p0+2*p1)/3,(p0+p1+p2)/3,(p0+2*p2)/3,p1,(2*p1+p2)/3,(p1+2*p2)/3,p2])
            tar = np.array([p0_val,(2*p0_val+p1_val)/3,(2*p0_val+ct)/3,(p0_val+2*p1_val)/3,(p0_val+p1_val+ct)/3,(p0_val+2*ct)/3,p1_val,(2*p1_val+ct)/3,(p1_val+2*ct)/3,ct])
            for idx,pt in enumerate(pts):
                if idx == 0 or idx == 6 or idx == 9:
                    nodes.append(pt)
                else:
                    nodes.append(level(*pt,tar[idx]).x)
        nodes = np.array(nodes)
        return nodes,tar

    def cubic(self,s,t,u):
        coef = np.array([t**3,3*s*t**2,3*u*t**2,3*s**2*t,6*s*t*u,
                         3*t*u**2,s**3,3*s**2*u,3*s*u**2,u**3])
        #point = np.sum(coef*self.nodes,axis=0)
        point = coef[0,:].reshape(-1,1)*self.nodes[0,:].reshape(1,-1) +\
                coef[1,:].reshape(-1,1)*self.nodes[1,:].reshape(1,-1) +\
                coef[2,:].reshape(-1,1)*self.nodes[2,:].reshape(1,-1) +\
                coef[3,:].reshape(-1,1)*self.nodes[3,:].reshape(1,-1) +\
                coef[4,:].reshape(-1,1)*self.nodes[4,:].reshape(1,-1) +\
                coef[5,:].reshape(-1,1)*self.nodes[5,:].reshape(1,-1) +\
                coef[6,:].reshape(-1,1)*self.nodes[6,:].reshape(1,-1) +\
                coef[7,:].reshape(-1,1)*self.nodes[7,:].reshape(1,-1) +\
                coef[8,:].reshape(-1,1)*self.nodes[8,:].reshape(1,-1) +\
                coef[9,:].reshape(-1,1)*self.nodes[9,:].reshape(1,-1)
        target = coef[0,:].reshape(-1,1)*self.targets[0].reshape(1,-1) +\
                 coef[1,:].reshape(-1,1)*self.targets[1].reshape(1,-1) +\
                 coef[2,:].reshape(-1,1)*self.targets[2].reshape(1,-1) +\
                 coef[3,:].reshape(-1,1)*self.targets[3].reshape(1,-1) +\
                 coef[4,:].reshape(-1,1)*self.targets[4].reshape(1,-1) +\
                 coef[5,:].reshape(-1,1)*self.targets[5].reshape(1,-1) +\
                 coef[6,:].reshape(-1,1)*self.targets[6].reshape(1,-1) +\
                 coef[7,:].reshape(-1,1)*self.targets[7].reshape(1,-1) +\
                 coef[8,:].reshape(-1,1)*self.targets[8].reshape(1,-1) +\
                 coef[9,:].reshape(-1,1)*self.targets[9].reshape(1,-1)
        return point,target

    def quart(self,s,t,u):
        coef = np.array([t**2,2*s*t,2*t*u,s**2,2*s*u,u**2])
        #point = np.sum(coef*self.nodes,axis=0)
        point = coef[0,:].reshape(-1,1)*self.nodes[0,:].reshape(1,-1) +\
                coef[1,:].reshape(-1,1)*self.nodes[1,:].reshape(1,-1) +\
                coef[2,:].reshape(-1,1)*self.nodes[2,:].reshape(1,-1) +\
                coef[3,:].reshape(-1,1)*self.nodes[3,:].reshape(1,-1) +\
                coef[4,:].reshape(-1,1)*self.nodes[4,:].reshape(1,-1) +\
                coef[5,:].reshape(-1,1)*self.nodes[5,:].reshape(1,-1)
        target = coef[0,:].reshape(-1,1)*self.targets[0].reshape(1,-1) +\
                 coef[1,:].reshape(-1,1)*self.targets[1].reshape(1,-1) +\
                 coef[2,:].reshape(-1,1)*self.targets[2].reshape(1,-1) +\
                 coef[3,:].reshape(-1,1)*self.targets[3].reshape(1,-1) +\
                 coef[4,:].reshape(-1,1)*self.targets[4].reshape(1,-1) +\
                 coef[5,:].reshape(-1,1)*self.targets[5].reshape(1,-1)
        return point,target

    def linear(self,s,t,u):
        coef = np.array([t,s,u])
        #point = np.sum(coef*self.nodes,axis=0)
        point = coef[0,:].reshape(-1,1)*self.nodes[0,:].reshape(1,-1) +\
                coef[1,:].reshape(-1,1)*self.nodes[1,:].reshape(1,-1) +\
                coef[2,:].reshape(-1,1)*self.nodes[2,:].reshape(1,-1)
        target = coef[0,:].reshape(-1,1)*self.targets[0].reshape(1,-1) +\
                 coef[1,:].reshape(-1,1)*self.targets[1].reshape(1,-1) +\
                 coef[2,:].reshape(-1,1)*self.targets[2].reshape(1,-1)
        return point,target

    def generate(self,n=10):
        s = np.ones((n,n))*np.array([np.flip(np.linspace(0,1,num=n))]).T
        t = np.linspace(0,1-np.linspace(0,1,num=n),num=n)
        u = np.ones((n,n)) - s - t
        s = s[1:].flatten()
        t = t[1:].flatten()
        u = u[1:].flatten()
        if self.degree == 1:
            points,targets = self.linear(s,t,u)
        elif self.degree == 2:
            points,targets = self.quart(s,t,u)
        elif self.degree == 3:
            points,targets = self.cubic(s,t,u)
        else:
            points = None
            targets = None
        return points,targets,(s,t,u)

    def get_jit(self):
        if self.degree == 3:
            @nb.jit(nopython=True,cache=True)
            def cubic(s,t,u,nodes=self.nodes):
                coef = np.array([t**3,3*s*t**2,3*u*t**2,3*s**2*t,6*s*t*u,
                                 3*t*u**2,s**3,3*s**2*u,3*s*u**2,u**3])
                point = coef[0,:].reshape(-1,1)*nodes[0,:].reshape(1,-1) +\
                        coef[1,:].reshape(-1,1)*nodes[1,:].reshape(1,-1) +\
                        coef[2,:].reshape(-1,1)*nodes[2,:].reshape(1,-1) +\
                        coef[3,:].reshape(-1,1)*nodes[3,:].reshape(1,-1) +\
                        coef[4,:].reshape(-1,1)*nodes[4,:].reshape(1,-1) +\
                        coef[5,:].reshape(-1,1)*nodes[5,:].reshape(1,-1) +\
                        coef[6,:].reshape(-1,1)*nodes[6,:].reshape(1,-1) +\
                        coef[7,:].reshape(-1,1)*nodes[7,:].reshape(1,-1) +\
                        coef[8,:].reshape(-1,1)*nodes[8,:].reshape(1,-1) +\
                        coef[9,:].reshape(-1,1)*nodes[9,:].reshape(1,-1)
                return point
            return cubic

class b_tri:
    def __init__(self,points):
        #Check for the correct number of control points
        degree = (-3+(9-4*(2-2*points.shape[0]))**(1/2))/2
        degree_int = round(degree)
        self.points = points
        if np.isclose(degree,degree_int,0.0001):
            self.degree = degree_int
        else:
            print('Incorrect number of control points')
        if degree > 6:
            print('Number of control points greater than degrees.')
    def sample(self,n=10):
        s = np.ones((n,n))*np.array([np.flip(np.linspace(0,1,num=n))]).T
        t = np.linspace(0,1-np.linspace(0,1,num=n),num=n)
        u = np.ones((n,n)) - s - t
        s = s[1:].flatten()
        t = t[1:].flatten()
        u = u[1:].flatten()
        if self.degree == 1:
            coef = np.array([t,s,u]).T
            samples = []
            for i in range(coef.shape[0]):
                tmp = np.sum(self.points*coef[i,:],axis=0)
                samples.append(tmp)
        elif self.degree == 2:
            coef = np.array([t**2,2*s*t,2*t*u,s**2,2*s*u,u**2]).T
            samples = []
            for i in range(coef.shape[0]):
                tmp = np.sum(self.points*coef[i,:],axis=0)
                samples.append(tmp)
        elif self.degree == 3:
            coef = np.array([t**3,3*s*t**2,3*u*t**2,3*s**2*t,6*s*t*u,
                             3*t*u**2,s**3,3*s**2*u,3*s*u**2,u**3]).T
            samples = []
            for i in range(coef.shape[0]):
                tmp = np.sum(self.points*coef[i,:],axis=0)
                samples.append(tmp)
        elif self.degree == 4:
            coef = np.array([t**4,4*s*t**3,4*u*t**3,6*t**2*s**2,12*t**2*u,6*t**2*u**2,
                             4*t*s**3,12*t*s**2*u,12*t*s*u**2,s**4,4*s**3*u,6*s**2*u**2,
                             4*s*u**3,u**4]).T
            samples = []
            for i in range(coef.shape[0]):
                tmp = np.sum(self.points*coef[i,:],axis=0)
                samples.append(tmp)
        samples = np.array(samples)
        return samples
"""
