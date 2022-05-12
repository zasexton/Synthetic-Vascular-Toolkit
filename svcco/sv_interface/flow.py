flow="""if not terminating:
    flow_file_3d = open('{}'+os.sep+'{}'+'_3d.flow','w+')
    flow_file_rom = open('{}'+os.sep+'{}'+'_rom.flow','w+')
    flow_file_lines_3d = []
    flow_file_lines_rom = []
    times = {}
    flows = {}
    for t,f in zip(times,flows):
        flow_file_lines_3d.append(str(t)+' -'+str(f)+'\\n')
        flow_file_lines_rom.append(str(t)+' '+str(f)+'\\n')
    flow_file_3d.writelines(flow_file_lines_3d)
    flow_file_rom.writelines(flow_file_lines_rom)
    flow_file_3d.close()
    flow_file_rom.close()
"""
