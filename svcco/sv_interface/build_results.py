
make_results = """import pyvista as pv
import numpy as np
import os
from tqdm import tqdm

geom_data = np.genfromtxt("geom.csv",delimiter=",")
data = np.load("solver_0d_branch_results.npy",allow_pickle=True).item()

total = None
timepoints = []
min_pressure = np.inf
max_pressure = -np.inf
min_flow = np.inf
max_flow = -np.inf
min_wss = np.inf
max_wss = -np.inf
for idx in tqdm(range(len(data['time'])),desc="Building Timeseries ",position=0):
    #time_merge = None
    time = data['time'][idx]
    tmp_vessels = []
    for jdx in tqdm(range(len(data['flow'])),desc="Building Vessel Data",position=1,leave=False):
        vessel = list(data['flow'].keys())[jdx]
        start = geom_data[vessel,0:3]
        end   = geom_data[vessel,3:6]
        direction = (geom_data[vessel,3:6] - geom_data[vessel,0:3])/np.linalg.norm(geom_data[vessel,3:6] - geom_data[vessel,0:3])
        length    = geom_data[vessel,6]
        radius    = geom_data[vessel,7]
        number_segments = len(data['flow'][jdx]) - 1 #assume only 1 segment right now
        number_points   = len(data['flow'][jdx])
        for kdx in range(number_segments):
            center = (1/2)*direction*length + start
            vessel = pv.Cylinder(center=center,direction=direction,height=length,radius=radius)
            vessel = vessel.elevation(low_point=end,high_point=start,scalar_range=[data['pressure'][jdx][kdx+1][idx]/1333.33, data['pressure'][jdx][kdx][idx]/1333.33])
            if data['pressure'][jdx][kdx][idx]/1333.33 > max_pressure:
                max_pressure = data['pressure'][jdx][kdx][idx]/1333.33
            if data['pressure'][jdx][kdx+1][idx]/1333.33 < min_pressure:
                min_pressure = data['pressure'][jdx][kdx+1][idx]/1333.33
            vessel.rename_array('Elevation','Pressure [mmHg]',preference='point')
            vessel.cell_data['Flow [mL/s]'] = data['flow'][jdx][kdx][idx]
            re = (1.06*2*radius*((data['flow'][jdx][kdx][idx]/(np.pi*radius**2))/2))/0.04
            fd = 64/re
            wss = ((data['flow'][jdx][kdx][idx]/(np.pi*radius**2))/2)*fd*1.06
            vessel.cell_data['WSS [dyne/cm^2]']  = wss
            if max_flow < data['flow'][jdx][kdx][idx]:
                max_flow = data['flow'][jdx][kdx][idx]
            if min_flow > data['flow'][jdx][kdx][idx]:
                min_flow = data['flow'][jdx][kdx][idx]
            if max_wss < wss:
                max_wss = wss
            if min_wss > wss:
                min_wss = wss
            tmp_vessels.append(vessel)
        #if time_merge is None:
        #    time_merge = vessel
        #else:
        #    time_merge = time_merge.merge(vessel)
    time_merge = tmp_vessels[0].merge(tmp_vessels[1:])
    time_merge.field_data['time'] = time
    timepoints.append(time_merge)
    #if total is None:
    #    total = time_merge
    #else:
    #    total = total.merge(time_merge)
    if not os.path.isdir("timeseries"):
        os.mkdir("timeseries")

if not os.path.isdir("timeseries_for_pressure_gif"):
    os.mkdir("timeseries_for_pressure_gif")
if not os.path.isdir("timeseries_for_flow_gif"):
    os.mkdir("timeseries_for_flow_gif")
if not os.path.isdir("timeseries_for_wss_gif"):
    os.mkdir("timeseries_for_wss_gif")
total = timepoints[0].merge(timepoints[1:])
for i in tqdm(range(len(timepoints)),desc="Saving Timeseries",position=1):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='Pressure [mmHg]',clim=[round(min_pressure,4),round(max_pressure,4)],cmap="coolwarm")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_pressure_gif"+os.sep+"time_point_{}.png".format(i))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='Flow [mL/s]',clim=[round(min_flow,4),round(max_flow,4)],cmap="GnBu")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_flow_gif"+os.sep+"time_point_{}.png".format(i))
    p = pv.Plotter(off_screen=True)
    p.add_mesh(timepoints[i],scalars='WSS [dyne/cm^2]',clim=[round(min_wss,2),round(max_wss,2)],cmap="coolwarm")
    p.show(auto_close=True,screenshot=os.getcwd()+os.sep+"timeseries_for_wss_gif"+os.sep+"time_point_{}.png".format(i))
    timepoints[i].save(os.getcwd()+os.sep+"timeseries"+os.sep+"time_point_{}.vtp".format(i))

total.save(os.getcwd()+os.sep+"timeseries"+os.sep+"total.vtp")
"""
