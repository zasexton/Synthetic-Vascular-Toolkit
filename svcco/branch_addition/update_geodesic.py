import numpy as np

def update_geodesic(data,gamma,nu):
    sub_division_map = [[-1]]*data.shape[0]
    sub_division_index = []
    
