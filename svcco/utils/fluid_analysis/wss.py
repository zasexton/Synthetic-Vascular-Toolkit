import numpy as np

def wss(*args,**kwargs):
    object = args[0]
    radius = object.data[:,21]
    flow   = object.data[:,22]
    dynamic_viscosity = object.parameters['nu']
    wss = (4*dynamic_viscosity*flow)/(np.pi*radius**3)
    return wss
