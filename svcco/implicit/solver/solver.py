import numpy as np
#import pygmo as pg
from scipy import optimize
from functools import partial
from .solver_functions.init_normals_given import init_normals_given
from .solver_functions.init_normals_not_given import init_normals_not_given
#import sys #should remove

#sys.path.append('..') # should remove
from ..kernel.kernel import kernel

class solver:
    def __init__(self,points,normals):
        self.points  = points
        self.normals = normals
        self.kernel  = kernel(points)
        #self.problem = pg.problem(self.kernel)

    def set_solver(self,verbose):
        #nlopt_solvers = ['cobyla','bobyqa','newuoa','newuoa','newuoa_bound',
        #                 'praxis','neldermead','sbplx','mma','ccsaq','slsqp',
        #                 'lbfgs','tnewton_precond_restart','tnewton_precond',
        #                 'tnewton','var2','var1','auglag','auglag_eq']
        scipy_solvers = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG',
                         'L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr',
                         'dogleg','trust-ncg','trust-exact','trust-krylov']
        #if self.method in nlopt_solvers:
        #    nl = pg.nlopt(self.method)
        #    nl.xtol_rel = 1E-7
        #    nl.ftol_rel = 1E-7
        #    if self.method == 'lbfgs':
        #        nl.maxeval = 3000
        #    algorithm = pg.algorithm(nl)
        #    if verbose:
        #        algorithm.set_verbosity(1)
        if self.method in scipy_solvers:
            if verbose:
                if self.method == 'trust-constr':
                    options = {'disp':True}
                elif self.method == 'L-BFGS-B':
                    options = {'disp':99}
                elif self.method in ['Newton-CG','CG','BFGS','Nelder-Mead',
                                     'Powell','TNC','COBYLA','SLSQP']:
                    options = {'disp':True}
                else:
                    print('No verbosity allowed for this method.')
                    options = {}
            else:
                options = {}
            algorithm = partial(optimize.minimize,method=self.method,
                                    tol=1e-07,options=options)
            #algorithm = pg.algorithm(scp)
        else:
            print('Not an available solver method.')
            #print('See nlopt methods: {}'.format(nlopt_solvers))
            print('See scipy methods: {}'.format(scipy_solvers))
        self.algorithm = algorithm

    def vector_solver(self):
        #self.population = pg.population(self.problem)
        initial_solution = init_normals_given(self.normals)
        #self.population.push_back(initial_solution)
        #results = self.algorithm.evolve(self.population)
        bounds = []
        kernel_bounds = self.kernel.get_bounds()
        for i in range(len(kernel_bounds[0])):
            bounds.append(tuple([kernel_bounds[0][i],kernel_bounds[1][i]]))
        results = self.algorithm(self.kernel.fitness,x0=initial_solution,
                                 jac=self.kernel.jac,bounds=bounds)
        return results

    def variational_solver(self,lam):
        #self.population = pg.population(self.problem)
        initial_solution = init_normals_not_given(lam,self.kernel.K00,
                                                      self.kernel.K01,
                                                      self.kernel.K11)
        #self.population.push_back(initial_solution)
        #results = self.algorithm.evolve(self.population)
        bounds = []
        kernel_bounds = self.kernel.get_bounds()
        for i in range(len(kernel_bounds)):
            bounds.append(tuple([kernel_bounds[0][i],kernel_bounds[1][i]]))
        results = self.algorithm(self.kernel.fitness,x0=initial_solution,
                                 jac=self.kernel.jac,bounds=bounds)
        return results

    def solve(self,seed_number=1,lb=0.01,ub=1,perturb=0.01,
               local_verbosity=False,local_method='L-BFGS-B',
               variational=False,solver_method='Bounded',
               solver_verbosity=True):
        self.method = local_method
        self.set_solver(local_verbosity)
        if variational:
            fit = lambda x: self.variational_solver(x).champion_f
            lam = optimize.minimize_scalar(fit,bounds=(lb,ub),tol=1e-07,
                                           method=solver_method,options={'disp':3})
            result = self.variational_solver(lam.x)
        else:
            result = self.vector_solver()
        gg = result.x
        n = self.kernel.ndim
        g = np.ones(n*self.kernel.ddim)
        M_inv = self.kernel.A_inv[:n*(self.kernel.ddim+1),:n*(self.kernel.ddim+1)]
        N_inv = self.kernel.A_inv[:n*(self.kernel.ddim+1),n*(self.kernel.ddim+1):]
        ##########################################
        # Check normal direction
        ##########################################
        for i in range(n):
            g[i] = np.cos(gg[i*2+1])*np.sin(gg[i*2])
            g[i+n] = np.sin(gg[i*2+1])*np.sin(gg[i*2])
            g[i+2*n] = np.cos(gg[i*2])
        s = np.zeros(n)
        l_side = np.zeros(M_inv.shape[1])
        l_side[:len(s)] = s
        l_side[len(s):(len(s)+len(g))] = g
        a = np.matmul(M_inv,l_side)
        b = np.matmul(N_inv.T,l_side)
        return a,b
