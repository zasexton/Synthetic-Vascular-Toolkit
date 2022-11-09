"""
File for loading spline functions.
"""
import tkinter as tk
from tkinter.filedialog import askopenfilename
import scipy as sp
import pickle

def load_splines(filename=None):
    """
    This file loads Spline functions that have
    been pickled into a binary file (*.pkl file)

    Parameters
    ----------
              filename: str, optional
                       the path to the spline file.
                       if this is not provided then
                       the function will prompt the
                       user to fetch the file with a
                       dialog window popup.

    Returns
    -------
              splines: List
                     a list of spline functions. each
                     list item is a different N-dimensional
                     spline which can be evaluated along the
                     interval [0,1]. Vessel data evaluations
                     are given in xyzr format. This means
                     the x-coordinate, y-coordinate, z-coordinate,
                     and radius value are returned in this
                     order.

    Example:

    >>> splines = load_splines()
    >>> splines[0](0)     # we are evaluating the first spline at zero
    [array(0.02465846), array(-1.3615156), array(1.86000645), array(0.06768623)]
    >>>
    >>> x,y,z,r = splines[0](0) # we can store the data into the relevant variables
    >>> print(x)
    array(0.02465846)
    >>>print(y)
    array(-1.3615156)
    >>>print(z)
    array(1.86000645)
    >>>print(r)
    array(0.06768623)
    
    """
    filename = askopenfilename()
    file     = open(filename,'rb')
    splines  = pickle.load(file)
    return splines
