import numpy as np
from .kernel_functions.fitness import fitness
from .kernel_functions.grad import grad
from .kernel_functions.hess import hess
#import sys #should remove

#sys.path.append('..')
from ..core.m_matrix import M
from ..core.n_matrix import N
from ..core.a_matrix import A
from ..core.h_matrix import H
from ..core.utils import norm

class kernel:
    def __init__(self,points):
        self.ndim = points.shape[0]
        self.ddim = points.shape[1]
        self.A_inv,self.K00,self.K01,self.K11 = A(points)
        self.H_0 = H(self.K00,self.K01,self.K11,0)
        self.fit  = lambda x: fitness(x,self.H_0)
        self.jac  = lambda x: grad(x,self.H_0,n=points.shape[0])
        self.hess = lambda x: hess(x,self.H_0,n=points.shape[0])
    def fitness(self,x):
        return self.fit(x)
    def get_bounds(self):
        bounds = []
        lb = []
        ub = []
        for i in range(self.ndim):
            for j in range(self.ddim-1):
                if j < self.ddim-2:
                    lb.append(-2*np.pi)
                    ub.append(2*np.pi)
                else:
                    lb.append(-2*np.pi)
                    ub.append(2*np.pi)
        bounds.append(lb)
        bounds.append(ub)
        return tuple(bounds)
    def gradient(self,x):
        return self.jac(x)
    def hessian(self,x):
        return self.hess(x)
