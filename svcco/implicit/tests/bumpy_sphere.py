import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bumpy_sphere(samples=10,scale=5,a=3,b=1):
    data = np.zeros((samples**2,3))
    theta = np.linspace(0,2*np.pi,num=samples)
    phi = np.linspace(0,np.pi,num=samples)
    #P,T = np.meshgrid(theta,phi)
    count = 0
    for t in theta:
        for p in phi:
            r = scale + np.sin(a*t)*np.sin(b*p)
            data[count,0] = r*np.cos(t)*np.sin(p)
            data[count,1] = r*np.sin(t)*np.sin(p)
            data[count,2] = r*np.cos(p)
            count += 1
    final_data = []
    #def bs(t,p,a=a,b=b,scale=scale):
    #    r = scale + np.sin(a*t)*np.sin(b*p)
    #    x = r*np.cos(t)*np.sin(p)
    #    y = r*np.sin(t)*np.sin(p)
    #    z = r*np.cos(p)
    #    return x,y,z
    #x,y,z = bs(T,P)
    #x = x.flatten()
    #y = y.flatten()
    #z = z.flatten()
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    #ax.scatter3D(x,y,z)
    #plt.show()
    #final_data = np.array([x,y,z]).T
    #print(final_data.shape)
    #final_data = np.unique(final_data,axis=0)
    for i in range(data.shape[0]):
        if i == 0:
            final_data.append(data[i,:])
            continue
        add = True
        for j in range(i):
            if np.all(np.isclose(data[i,:],data[j,:])):
                add = False
        if add:
            final_data.append(data[i,:])
    return np.array(final_data)
