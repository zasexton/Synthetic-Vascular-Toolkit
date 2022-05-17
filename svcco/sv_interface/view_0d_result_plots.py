view_plots="""import numpy as np
import matplotlib.pyplot as plt
data = np.load("solver_0d_branch_results.npy",allow_pickle=True).item()
vessels = []
fig_flow = plt.figure()
ax_flow = fig_flow.add_subplot()
fig_pressure = plt.figure()
ax_pressure = fig_pressure.add_subplot()
fig_flow_outlets = plt.figure()
ax_flow_outlets = fig_flow_outlets.add_subplot()
fig_pressure_outlets = plt.figure()
ax_pressure_outlets = fig_pressure_outlets.add_subplot()
for vessel in data["flow"]:
    vessels.append(vessel)
for vessel in vessels:
    ax_flow.plot(data["time"],data["flow"][vessel][0],label="vessel_"+str(vessel))
    ax_pressure.plot(data["time"],data["pressure"][vessel][0]/1333.22,label="vessel_"+str(vessel))
import json
info = json.load(open("solver_0d.in"))
all_inlets = []
all_outlets = []
for i in info["junctions"]:
    for j in i["inlet_vessels"]:
        all_inlets.append(j)
    for j in i["outlet_vessels"]:
        all_outlets.append(j)
all_inlets = set(all_inlets)
all_outlets = set(all_outlets)
true_outlets = list(all_outlets.difference(all_inlets))
true_inlets = list(all_inlets.difference(all_outlets))
for vessel in true_outlets:
    ax_flow_outlets.plot(data["time"],data["flow"][vessel][-1],label="vessel_"+str(vessel))
    ax_pressure_outlets.plot(data["time"],data["pressure"][vessel][-1]/1333.22,label="vessel_"+str(vessel))
ax_flow.set_xlabel("Time (sec)")
ax_flow.set_ylabel("Flow (mL/s)")
ax_pressure.set_xlabel("Time (sec)")
ax_pressure.set_ylabel("Pressure (mmHg)")
ax_flow_outlets.set_xlabel("Time (sec)")
ax_pressure_outlets.set_xlabel("Time (sec)")
ax_flow_outlets.set_ylabel("Flow (mL/s)")
ax_flow_outlets.set_title("Outlets Only")
ax_pressure_outlets.set_ylabel("Pressure (mmHg)")
ax_pressure_outlets.set_title("Outlets Only")
#ax_pressure_outlets.set_ylim([-10*np.finfo(float).eps,10*np.finfo(float).eps])
plt.show()

"""
