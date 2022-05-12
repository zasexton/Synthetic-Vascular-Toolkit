rom="""if not terminating:
    rom_simulation = simulation.ROM()
    params = simulation.ROMParameters()
    model_params = params.ModelParameters()

    model_params.name = '{}'+'_'+'{}'
    model_params.inlet_face_names = [str(inlet)]
    model_params.outlet_face_names = [str(o) for o in outlets]
    model_params.centerlines_file_name = '{}' + os.sep +'centerlines.vtp'

    mesh_params = params.MeshParameters()

    fluid_params = params.FluidProperties()
    material = params.WallProperties.OlufsenMaterial()

    bcs = params.BoundaryConditions()
    bcs.add_velocities(face_name=str(inlet),file_name='{}'+os.sep+'{}')
    for face_name in outlets:
        bcs.add_resistance(face_name=str(face_name),resistance=((1/resistance[face_name])/total_res)*0+outlet_res_total)
    solution_params = params.Solution()
    solution_params.time_step = {}
    solution_params.num_time_steps = {}
    outdir = '{}'
    rom_simulation.write_input_file(model_order={},model=model_params,
                                    mesh=mesh_params,fluid=fluid_params,
                                    material=material,boundary_conditions=bcs,
                                    solution=solution_params,directory=outdir)
    if {} == 0:
        zeroD_folder = '{}'
        run_rom = open(zeroD_folder+os.sep+'run.py','w+')
        run_rom_file = ''
        path = sv.__path__[0].replace(os.sep+'sv','')
        run_rom_file += 'import os\\n'
        run_rom_file += 'import sys\\n'
        run_rom_file += "sys.path.append('{{}}')\\n".format(path)
        run_rom_file += 'from svZeroDSolver import svzerodsolver\\n'
        input_file = '{}'+os.sep+'solver_0d.in'
        if 'Windows' in platform.system():
            input_file = input_file.replace(os.sep,os.sep+os.sep)
        run_rom_file += "svzerodsolver.solver.set_up_and_run_0d_simulation('{{}}')\\n".format(input_file)
        run_rom.writelines([run_rom_file])
        run_rom.close()

        rom_view = open(zeroD_folder+os.sep+'view.py','w+')
        rom_view_file = ''
        rom_view_file += 'import numpy as np\\n'
        rom_view_file += 'import matplotlib.pyplot as plt\\n'
        rom_view_file += 'data = np.load("solver_0d_branch_results.npy",allow_pickle=True).item()\\n'
        rom_view_file += 'vessels = []\\n'
        rom_view_file += 'fig_flow = plt.figure()\\n'
        rom_view_file += 'ax_flow = fig_flow.add_subplot()\\n'
        rom_view_file += 'fig_pressure = plt.figure()\\n'
        rom_view_file += 'ax_pressure = fig_pressure.add_subplot()\\n'
        rom_view_file += 'fig_flow_outlets = plt.figure()\\n'
        rom_view_file += 'ax_flow_outlets = fig_flow_outlets.add_subplot()\\n'
        rom_view_file += 'fig_pressure_outlets = plt.figure()\\n'
        rom_view_file += 'ax_pressure_outlets = fig_pressure_outlets.add_subplot()\\n'
        rom_view_file += 'for vessel in data["flow"]:\\n'
        rom_view_file += '    vessels.append(vessel)\\n'
        rom_view_file += 'for vessel in vessels:\\n'
        rom_view_file += '    ax_flow.plot(data["time"],data["flow"][vessel][0],label="vessel_"+str(vessel))\\n'
        rom_view_file += '    ax_pressure.plot(data["time"],data["pressure"][vessel][0]/1333.22,label="vessel_"+str(vessel))\\n'
        rom_view_file += 'import json\\n'
        rom_view_file += 'info = json.load(open("solver_0d.in"))\\n'
        rom_view_file += 'all_inlets = []\\n'
        rom_view_file += 'all_outlets = []\\n'
        rom_view_file += 'for i in info["junctions"]:\\n'
        rom_view_file += '    for j in i["inlet_vessels"]:\\n'
        rom_view_file += '        all_inlets.append(j)\\n'
        rom_view_file += '    for j in i["outlet_vessels"]:\\n'
        rom_view_file += '        all_outlets.append(j)\\n'
        rom_view_file += 'all_inlets = set(all_inlets)\\n'
        rom_view_file += 'all_outlets = set(all_outlets)\\n'
        rom_view_file += 'true_outlets = list(all_outlets.difference(all_inlets))\\n'
        rom_view_file += 'true_inlets = list(all_inlets.difference(all_outlets))\\n'
        rom_view_file += 'for vessel in true_outlets:\\n'
        rom_view_file += '    ax_flow_outlets.plot(data["time"],data["flow"][vessel][-1],label="vessel_"+str(vessel))\\n'
        rom_view_file += '    ax_pressure_outlets.plot(data["time"],data["pressure"][vessel][-1]/1333.22,label="vessel_"+str(vessel))\\n'
        rom_view_file += 'ax_flow.set_xlabel("Time (sec)")\\n'
        rom_view_file += 'ax_flow.set_ylabel("Flow (mL/s)")\\n'
        rom_view_file += 'ax_pressure.set_xlabel("Time (sec)")\\n'
        rom_view_file += 'ax_pressure.set_ylabel("Pressure (mmHg)")\\n'
        rom_view_file += 'ax_flow_outlets.set_xlabel("Time (sec)")\\n'
        rom_view_file += 'ax_pressure_outlets.set_xlabel("Time (sec)")\\n'
        rom_view_file += 'ax_flow_outlets.set_ylabel("Flow (mL/s)")\\n'
        rom_view_file += 'ax_flow_outlets.set_title("Outlets Only")\\n'
        rom_view_file += 'ax_pressure_outlets.set_ylabel("Pressure (mmHg)")\\n'
        rom_view_file += 'ax_pressure_outlets.set_title("Outlets Only")\\n'
        rom_view_file += 'plt.show()\\n'
        rom_view.writelines([rom_view_file])
        rom_view.close()
    if {} == 1:
        oneD_folder = '{}'
        print('Locating OneDSovler executable...')
        import sv
        from glob import glob
        import shutil
        if platform.system() == 'Windows':
            #search_path = sv.__path__[0].split('Python3.5')[0]
            search_path = 'C:\\\\Program Files\\\\SimVascular'
            files = glob(search_path+'\\\\**\\\\svOneDSolver.exe',recursive=True)
            if len(files) > 0:
                isfile = [os.path.isfile(f) for f in files]
                only_files = [files[of] for of in isfile if of]
                print('OneDSolver Found!')
                shutil.copy(only_files[0],oneD_folder)
            else:
                print('OneDSolver not found (will require user-specified executable location)')
        elif platform.system() == 'Linux':
            if 'simvascular' not in sv.__path__[0]:
                print('WARNING: Searching non-release build. OneDSolver not guarunteed to be installed.')
                print('Searching parent directory to simvascular build')
                search_dir = '/usr/local/sv'
                files = glob(search_dir+'/**/OneDSolver',recursive=True)
                if len(files) > 0:
                    isfile = [os.path.isfile(f) for f in files]
                    only_files = [files[of] for of in isfile if of]
                    print('OneDSolver Found!')
                    shutil.copy(only_files[0],oneD_folder)
                else:
                    print('OneDSolver not found (will require user-specified executable location)')
            else:
                #search_dir = sv.__path__[0].split('simvascular')[0]
                search_dir = '/usr/local/sv'
                files = glob(search_dir+'/**/OneDSolver',recursive=True)
                if len(files) > 0:
                    isfile = [os.path.isfile(f) for f in files]
                    only_files = [files[of] for of in isfile if of]
                    print('OneDSolver Found!')
                    shutil.copy(only_files[0],oneD_folder)
                else:
                    print('OneDSolver not found (will require user-specified executable location)')
        elif platform.system() == 'Darwin':
            # We will just assume the search directory here
            search_dir = '/usr/local/sv'
            files = glob(search_dir+'/**/OneDSolver',recursive=True)
            if len(files) > 0:
                isfile = [os.path.isfile(f) for f in files]
                only_files = [files[of] for of in isfile if of]
                print('OneDSolver Found!')
                shutil.copy(only_files[0],oneD_folder)
            else:
                print('OneDSolver not found (will require user-specified executable location)')
"""
