# maintainer: Martin Pfaller
# 
# created by Martina Weigl
# adapted from Matlab toolbox
# http://de.mathworks.com/matlabcentral/fileexchange/27826-fast-assembly-of-stiffness-and-matrices-in-finite-element-method-using-nodal-elements
# http://www.mis.mpg.de/preprints/tr/report-1111.pdf

import sys, timeit
from numpy import abs, array, ones, ravel, squeeze, sum, zeros
import numpy as np
from scipy.special import factorial
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

# help functions for stiffness matrix calculation
def phider(coord,point,etype,nargout):
    
    jacout = False
    if nargout > 2:
        jacout = True
    detout = False
    if nargout > 1:
        detout = True
        
    nod = coord.shape[1]
    nop = point.shape[1]
    nos = coord.shape[2]
    noe = coord.shape[0]
     
    # gradients of the shape functions
    dshape = shapeder(point,etype)  
    
    if jacout == True:
        jac = zeros((noe,nop,nod,nod))
    if detout == True:
        detj = zeros((noe,nop,1))
    dphi = zeros((noe,nop,nod,nos))

    for poi in range(nop):
         
        tjac = smamt(dshape[poi],coord)
        tjacinv, tjacdet = aminv(tjac)
        dphi[:,poi,:,:] = amsm(tjacinv,dshape[poi])
        
        if jacout == True:
            jac[:,poi,:,:] = tjac
        if detout == True:
            detj[:,0,:] = abs(tjacdet)
        
    return dphi, detj

    
def shapeder(point,etype):
     
    nod = point.shape[0]
    nop = point.shape[1]

    if nod == 1:
         
        l1 = point[0]
        l2 = 1 - l1
         
        if etype == "P0":
              
            dshape = zeros((nop,1,1))
             
        elif etype == "P1":
            
            dshape = array([1, -1])
            dshape = dshape.reshape(2,1)
            dshape = dshape * ones((nop,1))
            dshape = dshape.reshape(nop,1,2)
            
        elif etype == "P2":
            
            dshape = array([[4 * l1 - 1],
                            [-4 * l2 + 1],
                            [4 * (l2 - l1)]])
            dshape = dshape.reshape(nop,1,3)
            
        else:
            print("Error: Only P1, P2 elements implemented.")
    
    
    if nod == 2:
        
        l1 = point[0]
        l2 = point[1]
        l3 = 1 - l1 - l2
        
        if etype == "P0":
            
            dshape = zeros((nop,2,1))
            
        elif etype == "P1":
            
            dshape = array([[1, 0, -1],
                            [0, 1, -1]])
            dshape = dshape.reshape(6,1)
            dshape = dshape * ones((nop,1))
            dshape = dshape.reshape(nop,2,3)
            
        elif etype == "P2":
            
            dshape = array([[- 4 * l3 + 1],
                            [- 4 * l3 + 1],
                            [4 * l1 - 1],
                            [zeros((nop,1))],
                            [zeros((nop,1))],
                            [4 * l2 - 1],
                            [4 * l2],
                            [4 * l1],
                            [-4 * l2],
                            [4 * (l3 - l2)],
                            [4 * (l3 - l1)],
                            [-4 * l1]])
            dshape = dshape.reshape(nop,2,6)
            
        else:
            print("Error: Only P1, P2 elements implemented.")
            
            
    if nod == 3:
        
        l1 = point[0]
        l2 = point[1]
        l3 = point[2]
        l4 = 1 - l1 - l2 - l3
        
        if etype == "P0":
            
            dshape = zeros((nop,1,1))
            
        elif etype == "P1":
            
            dshape = array([[1, 0, 0, -1],
                            [0, 1, 0, -1],
                            [0, 0, 1, -1]])
            dshape = dshape.reshape(12,1)
            dshape = dshape * ones((nop,1))
            dshape = dshape.reshape(nop,3,4)
            
        elif etype =="P2":
            
            dshape = array([[-4 * l4 + 1],
                            [-4 * l4 + 1],
                            [-4 * l4 + 1],
                            [4 * l1 - 1],
                            [zeros((nop,1))],
                            [zeros((nop,1))],
                            [zeros((nop,1))],
                            [4 * l2 - 1],
                            [zeros((nop,1))],
                            [zeros((nop,1))],
                            [zeros((nop,1))],
                            [4 * l3 - 1],
                            [4 * (l4 - l1)],
                            [-4 * l1],
                            [-4 * l1],
                            [-4 * l2],
                            [4 * (l4 - l2)],
                            [-4 * l2],
                            [-4 * l3],
                            [-4 * l3],
                            [4 * (l4 - l3)],
                            [4 * l2],
                            [4 * l1],
                            [zeros((nop,1))],
                            [4 * l3],
                            [zeros((nop,1))],
                            [4 * l1],
                            [zeros((nop,1))],
                            [4 * l3],
                            [4 * l2]])
            dshape = dshape.reshape(nop,3,10)
            
        elif etype == "Q1":
             
            x = point[0]
            y = point[1]
            z = point[2]
             
            S = array([[0,0,0],
                       [1,0,0],
                       [1,1,0],
                       [0,1,0],
                       [0,0,1],
                       [1,0,1],
                       [1,1,1],
                       [0,1,1]]) * 2 - 1
                        
            dshape = zeros((nop,3,8))
             
            for n in range(nop):
                for m in range(8):
                    dshape[n][0][m] = 1./8 * S[m][0] * (1 + S[m][1] * y) * (1 + S[m][2] * z)
                    dshape[n][1][m] = 1./8 * (1 + S[m][0] * x) * S[m][1] * (1 + S[m][2] * z)
                    dshape[n][2][m] = 1./8 * (1 + S[m][0] * x) * (1 + S[m][1] * y) * S[m][2]
                    
        else:
            print("Error: Only P1, P2, Q1 elements implemented.")
     
                
    return dshape        


def smamt(smx,ama):
    
    ny = ama.shape[1]
    nx = ama.shape[2]
    nz = ama.shape[0]
    
    nk = smx.shape[0]
    
    amb = zeros((nz,nk,ny))
    for row in range(nk):
        
        amb[:,row,:] = svamt(smx[row],ama)
        
    return amb
    
    
def svamt(svx,ama):
    
    ny = ama.shape[1]
    nx = ama.shape[2]
    nz = ama.shape[0]
    
    avx = zeros((nz,ny,nx))
    
    for n in range(nz):
        avx[n] = svx
        
    avb = ama * avx
    avb = sum(avb, axis=2)    
   
    return avb


def aminv(ama):
    
    nx = ama.shape[1]
    nz = ama.shape[0]
    
    
    if nx == 1:
        
        dem = squeeze(ama)
        
        amb = zeros((nz,nx,nx))
        for n in range(nz):
            amb[n] = 1./ama[n]
  
        return amb, dem
    
    
    elif nx == 2:
        
        x1,x2,y1,y2 = zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1))
        
        for n in range(nz):
            x1[n],x2[n] = squeeze(ama[n][0][0]), squeeze(ama[n][0][1])
            y1[n],y2[n] = squeeze(ama[n][1][0]), squeeze(ama[n][1][1])
            
        dem = x1 * y2 - x2 * y1
            
        amb = zeros((nz,nx,nx))
        for n in range(nz):
            amb[n][0][0] = y2[n] / dem[n]
            amb[n][1][1] = x1[n] / dem[n]
            amb[n][0][1] = -x2[n] / dem[n]
            amb[n][1][0] = -y1[n] / dem[n]
            
        return amb, dem
    
    
    elif nx == 3:
        
        x1,x2,x3,y1,y2,y3,z1,z2,z3 = zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1)),zeros((nz,1))
        
        for n in range(nz):
            x1[n],x2[n],x3[n] = squeeze(ama[n][0][0]), squeeze(ama[n][0][1]), squeeze(ama[n][0][2])
            y1[n],y2[n],y3[n] = squeeze(ama[n][1][0]), squeeze(ama[n][1][1]), squeeze(ama[n][1][2])
            z1[n],z2[n],z3[n] = squeeze(ama[n][2][0]), squeeze(ama[n][2][1]), squeeze(ama[n][2][2])  
            
        dem = x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1)  
        
        C11 = y2 * z3 - y3 * z2
        C12 = -(y1 * z3 - y3 * z1)
        C13 = y1 * z2 - y2 * z1
        C21 = -(x2 * z3 - x3 * z2)
        C22 = x1 * z3 - x3 * z1
        C23 = -(x1 * z2 - x2 * z1)
        C31 = x2 * y3 - x3 * y2
        C32 = -(x1 * y3 - x3 * y1)
        C33 = x1 * y2 - x2 * y1
        
        amb = zeros((nz,nx,nx))
        for n in range(nz):
            amb[n][0][0] = C11[n] / dem[n]
            amb[n][1][0] = C12[n] / dem[n]
            amb[n][2][0] = C13[n] / dem[n]
            amb[n][0][1] = C21[n] / dem[n]
            amb[n][1][1] = C22[n] / dem[n]
            amb[n][2][1] = C23[n] / dem[n]
            amb[n][0][2] = C31[n] / dem[n]
            amb[n][1][2] = C32[n] / dem[n]
            amb[n][2][2] = C33[n] / dem[n]
            
        return amb, dem
    
    else:
        print("Error: Array operation for inverting matrices of dimension more than three is not available; performance may be bad.")
        sys.exit()


def amsm(ama,smx):  
    
    nx = ama.shape[1]
    ny = ama.shape[2]
    nz = ama.shape[0]
    
    nk = smx.shape[1]
    
    amb = zeros((nz,nx,nk))
    for col in range(nk):
        amb[:,:,col] = amsv(ama,smx[:,col])
        
    return amb
              
            
def amsv(ama,svx):
    
    nx = ama.shape[1]
    ny = ama.shape[2]
    nz = ama.shape[0]
            
    avx = zeros((nz,nx,ny))
    
    for n in range(nz):
        avx[n] = svx
        
    avb = ama * avx
    avb = sum(avb, axis=2) 

    return avb


def astam(asx, ama):
    
    nx = ama.shape[1]
    ny = ama.shape[2]
    nz = ama.shape[0]
    
    asm = zeros((nz,nx,ny))
    
    for n in range(nz):
        asm[n] = asx[n]
    
    amb = ama * asm
    
    return amb


def amtam(amx,ama):
    
    ny = ama.shape[2]
    
    nk = amx.shape[2]
    nz = amx.shape[0]
 
    amb = zeros((nz,nk,ny))
    
    for row in range(nk):
        amb[:,row] = avtam(amx[:,:,row],ama)
      
    return amb


def avtam(avx,ama):
    
    nx = ama.shape[1]
    ny = ama.shape[2]
    nz = ama.shape[0]

    avm = zeros((nz,nx,ny))
    
    for n in range(nz):
        for m in range(ny):
            avm[n,:,m] = avx[n]
           
    avb = ama * avm
    avb = sum(avb, axis=1)

    return avb


# class to store StiffnessMatrix and solve laplace problem
class StiffnessMatrix:
    # create stiffness matrix on initialization and store as class variable
    def __init__(self, element, coordinates, DOF_d = []):
        start = timeit.default_timer()
        
        # numder of nodes, problem dimension
        NN, DIM = coordinates.shape
        
        # number of elements, number of local basic functions
        NE, NLB = element.shape
        
        coord = zeros((NE,DIM,NLB))
        for x in range(NE):
            for n in range(NLB):
                for m in range(DIM):
                    coord[x][m][n] = coordinates[element[x][n]][m]
        
        # the coordinates of the points on the reference element
        IP = array([[0.25],
                    [0.25],
                    [0.25]])
    
        # element type for shape function       
        etype = "P1"
    
        # gradients of the basis functions
        nargout = 2
        dphi, detj = phider(coord,IP,etype,nargout)
        areas = abs(squeeze(detj)) / factorial(DIM, exact = True)
        
        dphi = squeeze(dphi)
        
        Z = astam(areas,amtam(dphi,dphi))
        
        Y = zeros((NE,NLB,NLB))
        for n in range(NE):
            for m in range(NLB):
                Y[n][m] = element[n][m]
                
        X = zeros((NE,NLB,NLB))
        for n in range(NE):
            X[n] = element[n]
        
        # stiffness matrix
        A = sparse.csc_matrix((ravel(Z),(ravel(Y),ravel(X))),shape=[NN,NN])

        # store number of nodes
        self._n = A.shape[0]
        
        # store stiffness matrix
        self._A = A
        
        stop = timeit.default_timer()
        print('created stiffness matrix in %.1fs' %(stop-start))
        
        # check if StiffnessMatrix is initialized with dirichlet boundary conditions
        if any(DOF_d):
            start = timeit.default_timer()
            
            # detect free DOFs
            DOF_f = np.setdiff1d(range(self._n),DOF_d)
            
            # LU decomposition for sparse matrix
            self._LU = splu(A[DOF_f,:][:,DOF_f])
            
            # store free DOFs with which LU-decomposition was initialized
            self._DOF_f_init = DOF_f
        
            stop = timeit.default_timer()
            print('created LU-decomposition in %.1fs' %(stop-start))
    
    # harmonic lifting through solving the laplace equation
    def HarmonicLift(self, DOF_d, X_d):
        # detect free DOFs
        DOF_f = np.setdiff1d(range(self._n),DOF_d)

        # condensate dirichlet DOFs
        rhs = -self._A[DOF_f,:][:,DOF_d].dot(X_d)
        
        # solve harmonic lifting
        if hasattr(self, '_LU') and np.array_equal(np.sort(DOF_f), np.sort(self._DOF_f_init)):
            # DBC DOFs are equal to the ones with which StiffnessMatrix was initialized
            # -> use LU-decomposition to solve (fast)
            X_f = self._LU.solve(rhs)
        else:
            # different DBC DOFs
            # -> solve by inverting A (slow)
            X_f = spsolve(self._A[DOF_f,:][:,DOF_f],rhs)
        
        # assemble solution vector
        X = zeros((self._n))
        X[DOF_d] = X_d
        X[DOF_f] = X_f

        return X