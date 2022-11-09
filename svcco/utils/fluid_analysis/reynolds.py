import matplotlib.pyplot as plt
import numpy as np

def reynolds(*args,**kwargs):
    object = args[0]
    mean_velocity = object.data[:,22]/(object.data[:,21]**2*np.pi)
    length = object.data[:,21]*2
    dynamic_viscosity = object.parameters['nu']
    density = object.parameters['rho']
    re = (density*mean_velocity*length)/dynamic_viscosity
    return re
