import pdb
import numpy as np
import matplotlib.pyplot as plt

import sys
import svzerodsolver as run0d


def run_network_util(block_list, deltat, time, bc_type):
    connect_list, wdict = run0d.connections.connect_blocks_by_inblock_list(block_list)
    neq = run0d.connections.compute_neq(block_list, wdict)  # compute number of equations
    for b in block_list:  # run a consistency check
        run0d.connections.check_block_connection(b)
    var_name_list = run0d.connections.assign_global_ids(block_list, wdict)  # assign solution variables with global ID

    y0, ydot0 = run0d.connections.initialize_solution_structures(neq)  # initialize solution structures
    curr_y = y0.copy()
    curr_ydot = ydot0.copy()

    rho = 0.1
    args = {}
    args['Time step'] = deltat
    args['rho'] = rho
    args['Wire dictionary'] = wdict
    args['check_jacobian'] = False

    # create time integration
    t_int = run0d.time_integration.GenAlpha(rho, curr_y)

    ylist = [curr_y.copy()]
    for t in time[1:]:
        args['Solution'] = curr_y
        curr_y, curr_ydot = t_int.step(curr_y, curr_ydot, t, block_list, args, deltat)
        ylist.append(curr_y)

    ###############
    # these corrections are needed based on how i think the above for loop works: the "t" in the above for loop corresponds to t_current (t_n) and the output of gen_alpha_dae_integrator_NR is y_next (y_(n+1)). see https://github.com/StanfordCBCL/0D_LPN_Python_Solver/blob/master/test_nonlin_res.py for proof.
    ylist = ylist[:-1]
    ylist.insert(0, y0.copy())
    ###############

    ylist = np.array(ylist)
    for i in range(len(ylist[0, :])):
        if var_name_list[i] == "P_inflow_" + bc_type:
            inlet_pressure = ylist[:, i]
            return inlet_pressure


def run_rcr(Qfunc, time, p, distal_pressure):
    deltat = time[1] - time[0]

    inflow = run0d.blocks.UnsteadyFlowRef(connecting_block_list=['rcr'],
                                          name='inflow',
                                          Qfunc=Qfunc,
                                          flow_directions=[+1])
    rcr = run0d.blocks.UnsteadyRCRBlockWithDistalPressure(connecting_block_list=['inflow'],
                                                          name='rcr',
                                                          flow_directions=[-1],
                                                          Rp_func=lambda t: p['Rp'],
                                                          C_func=lambda t: p['C'],
                                                          Rd_func=lambda t: p['Rd'],
                                                          Pref_func=lambda t: distal_pressure)
    return run_network_util([inflow, rcr], deltat, time, bc_type="rcr")


def run_coronary(Qfunc, time, p, p_im_time, p_im_pres, cardiac_cycle_period, Pv=0.0):
    deltat = time[1] - time[0]

    Pim_matrix = np.zeros((len(p_im_time), 2))
    Pim_matrix[:, 0] = p_im_time
    Pim_matrix[:, 1] = p_im_pres
    distal_pressure = np.zeros((len(p_im_time), 2))
    distal_pressure[:, 0] = p_im_time
    distal_pressure[:, 1] = np.ones(len(p_im_time)) * Pv  # leaving Pv = 0.0 should be okay

    inflow = run0d.blocks.UnsteadyFlowRef(connecting_block_list=['coronary'], Qfunc=Qfunc, name='inflow',
                                          flow_directions=[+1])

    coronary = run0d.blocks.OpenLoopCoronaryWithDistalPressureBlock(Ra=p['R1'], Ca=p['C1'], Ram=p['R2'], Cim=p['C2'],
                                                                       Rv=p['R3'], Pim=Pim_matrix, Pv=distal_pressure,
                                                                       cardiac_cycle_period=cardiac_cycle_period,
                                                                       connecting_block_list=['inflow'],
                                                                       name='coronary', flow_directions=[-1])

    block_list = [inflow, coronary]

    return run_network_util(block_list, deltat, time, bc_type="coronary")
