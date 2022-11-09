import numpy as np

"""
Function to approximate wall
shear stress for a cylindrical
pipe under poiseuille assumptions
with ideal circular cross-sections.

Multifidelity Estimators For Coronary Circulation
Models under clinically-informed Data Uncertainty

Reference: https://arxiv.org/pdf/1911.11266.pdf
"""

def wss(*args,**kwargs):
    """
    Args
    ----

    Kwargs
    ------

    Returns
    -------
    """
    object = args[0]
    radius = object.data[:,21]
    flow   = object.data[:,22]
    dynamic_viscosity = object.parameters['nu']
    wss = (4*dynamic_viscosity*flow)/(np.pi*radius**3)
    return wss
