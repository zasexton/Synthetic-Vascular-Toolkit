import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from vtk_functions import read_geo, write_geo, add_array


def get_velocities(geo):
    narrs = geo.GetPointData().GetNumberOfArrays()
    velocities = []
    count = 0
    for ind in range(narrs):
        name = geo.GetPointData().GetArrayName(ind)
        if 'velocity_' in name and not 'average_' in name:
            if count % 1 == 0:
                print(name)
                cur_velocity = v2n(geo.GetPointData().GetArray(ind))
                velocities.append(np.reshape(cur_velocity, cur_velocity.size))
                print(count)
            count = count + 1
    velocities = np.array(velocities)
    return velocities
    

if __name__ == "__main__":
    geo = read_geo('0069_0001.vtu')
    velocities = get_velocities(geo).transpose()
    svd = np.linalg.svd(velocities, full_matrices=False)
    lerror = 0
    sol_index = 0
    # find maximum error
    for i in range(velocities.shape[1]):
        A = np.matmul(np.diag(svd[1]),svd[2][:,i])
        C = svd[0][:,0] * A[0]
        error = np.linalg.norm(velocities[:,i] - C)
        if error > lerror:
            error = lerror
            sol_index = i
    print(error)
    print(i)
    ref_solution = np.copy(velocities[:,sol_index])
    A = np.matmul(np.diag(svd[1]),svd[2][:,sol_index])
    print(A[0:10])
    C1 = svd[0][:,0] * A[0]
    C2 = np.matmul(svd[0][:,0:2], A[0:2])
    C5 = np.matmul(svd[0][:,0:5], A[0:5])
    C10 = np.matmul(svd[0][:,0:10], A[0:10])
    node_all = geo.GetPointData()
    for i in range(node_all.GetNumberOfArrays() - 1):
        node_all.RemoveArray(1)
    ncomps = int(svd[0].shape[0]/3)
    add_array(geo, 'approx_1 modes', np.reshape(C1,(ncomps,3)))
    add_array(geo, 'approx_2 modes', np.reshape(C2,(ncomps,3)))
    add_array(geo, 'approx_5 modes', np.reshape(C5,(ncomps,3)))
    add_array(geo, 'approx_10 modes', np.reshape(C10,(ncomps,3)))
    add_array(geo, 'solution', np.reshape(ref_solution,(ncomps,3)))
    write_geo('test.vtu', geo)