solver="""if not terminating:
    solver_file = open('{}'+'/{}.inp','w+')
    solver_construction = 'Density: {}\\n'
    solver_construction += 'Viscosity: {}\\n\\n'
    solver_construction += 'Number of Timesteps: {}\\n'
    solver_construction += 'Time Step Size: {}\\n\\n'
    solver_construction += 'Number of Timesteps between Restarts: {}\\n'
    solver_construction += 'Number of Force Surfaces: {}\\n'
    solver_construction += 'Surface ID\\'s for Force Calculation: {}\\n'
    solver_construction += 'Force Calculation Method: {}\\n'
    solver_construction += 'Print Average Solution: {}\\n'
    solver_construction += 'Print Error Indicators: {}\\n\\n'
    solver_construction += 'Time Varying Boundary Conditions From File: {}\\n\\n'
    solver_construction += 'Step Construction: {}\\n\\n'
    num_res = len(mesher.get_model_face_ids()) - 2
    solver_construction += 'Number of Resistance Surfaces: {{}}\\n'.format(num_res)
    solver_construction += 'List of Resistance Surfaces:'
    for j in range(2,len(mesher.get_model_face_ids())+1):
        if j == skip:
            continue
        solver_construction += ' {{}}'.format(j)
    solver_construction += '\\n'
    solver_construction += 'Resistance Values:'
    faces = mesher.get_model_face_ids()
    outlet_res_total = {}
    for j in range(2,len(mesher.get_model_face_ids())+1):
        if j == skip:
            continue
        solver_construction += ' {{}}'.format(round(((1/resistance[faces[j-1]])*total_res)*outlet_res_total,2))
    solver_construction += '\\n\\n'
    solver_construction += 'Pressure Coupling: {}\\n'
    solver_construction += 'Number of Coupled Surfaces: {{}}\\n\\n'.format(num_res)
    solver_construction += 'Backflow Stabilization Coefficient: {}\\n'
    solver_construction += 'Residual Control: {}\\n'
    solver_construction += 'Residual Criteria: {}\\n'
    solver_construction += 'Minimum Required Iterations: {}\\n'
    solver_construction += 'svLS Type: {}\\n'
    solver_construction += 'Number of Krylov Vectors per GMRES Sweep: {}\\n'
    solver_construction += 'Number of Solves per Left-hand-side Formation: {}\\n'
    solver_construction += 'Tolerance on Momemtum Equations: {}\\n'
    solver_construction += 'Tolerance on Continuity Equations: {}\\n'
    solver_construction += 'Tolerance on svLS NS Solver: {}\\n'
    solver_construction += 'Maximum Number of Iterations for svLS NS Solver: {}\\n'
    solver_construction += 'Maximum Number of Iterations for svLS Momentum Loop: {}\\n'
    solver_construction += 'Maximum Number of Iterations for svLS Continuity Loop: {}\\n'
    solver_construction += 'Time Integration Rule: {}\\n'
    solver_construction += 'Time Integration Rho Infinity: {}\\n'
    solver_construction += 'Flow Advection Form: {}\\n'
    solver_construction += 'Quadrature Rule on Interior: {}\\n'
    solver_construction += 'Quadrature Rule on Boundary: {}\\n'
    solver_file.writelines([solver_construction])
    solver_file.close()
    numstart = open('{}'+'/numstart.dat','w+')
    numstart.writelines(['0'])
    numstart.close()

"""
