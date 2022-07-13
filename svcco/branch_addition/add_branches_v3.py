import numpy as np
import os #remove before final
import time #not neccessary?
from .check import *
from .close import *
from .local_func_v7 import *
from ..collision.sphere_proximity import *
from ..collision.collision import *
from .add_bifurcation import *
from .sample_triad import *
from .triangle import * #might not need
from .basis import *
from scipy import interpolate
from scipy.spatial import KDTree
import matplotlib.pyplot as plt #remove before final
from .get_point import *
from mpl_toolkits.mplot3d import proj3d #remove before final
from .geodesic import extract_surface,geodesic
from ..implicit.visualize.visualize import show_mesh
from .finite_difference import finite_difference
from scipy.sparse.csgraph import shortest_path
from .add_geodesic_path import *
import pyvista as pv
import random
from scipy.interpolate import griddata
from time import perf_counter

def add_branch(tree,low,high,threshold_exponent=1.5,threshold_adjuster=0.75,
               all_max_attempts=40,max_attemps=10,sampling=20,max_skip=8,
               flow_ratio=None,radius_buffer=0,isforest=False,threshold=None,radius_scale=4,
               method='L-BFGS-B'):
    number_edges       = tree.parameters['edge_num']
    if threshold is None:
        threshold = ((tree.boundary.volume)**(1/3)/(number_edges**threshold_exponent))
    mu                 = tree.parameters['mu']
    lam                = tree.parameters['lambda']
    gamma              = tree.parameters['gamma']
    nu                 = tree.parameters['nu']
    Qterm              = tree.parameters['Qterm']
    Pperm              = tree.parameters['Pperm']
    Pterm              = tree.parameters['Pterm']
    new_branch_found   = False
    if tree.homogeneous:
        subb = None
        search_time = 0
        close_time = 0
        constraint_time = 0
        local_time = 0
        collision_time = 0
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        add_time = 0
        total_time = 0
        start_total = time.time()
        start_rng = time.time()
        nonconvex_solve = True
        if len(tree.rng_points) == 0:
            #print('repick')
            rng_points,_ = tree.boundary.pick(size=len(tree.boundary.tet_verts),homogeneous=True,replacement=False)
            rng_points = rng_points.tolist()
        else:
            rng_points = tree.rng_points
        total_attempts = 0
        attempt        = 0
        while not new_branch_found:
            start = time.time()
            #print(threshold_distance)
            #total_attempts = 0
            #attempt        = 0
            for att in range(all_max_attempts):
                #print(attempt)
                start1 = time.time()
                #point,_ = tree.boundary.pick(homogeneous=True)
                #time1 += time.time() - start1
                if len(rng_points) == 1:
                    #print('repick_low')
                    rng_points,_ = tree.boundary.pick(size=len(tree.boundary.tet_verts),homogeneous=True,replacement=False)
                    rng_points = rng_points.tolist()
                point = np.array(rng_points.pop(0))
                if number_edges < 200:
                    vessel, line_distances = close_exact(tree.data,point)
                else:
                    vessel, line_distances = close(tree.data,point)
                time1 += time.time() - start1
                start2 = time.time()
                line_distances_below_threshold = sum(line_distances < threshold)
                minimum_line_distance = min(line_distances)
                time2 += time.time() - start2
                start3 = time.time()
                ################
                # check to see the attempt number
                ################
                if attempt < max_attemps:
                    attempt += 1
                    total_attempts += 1
                else:
                    if not tree.convex:
                        cell_list_outter = set([])
                        ptss = []
                        for i in range(tree.data.shape[0]):
                            first  = tree.data[i,0:3]
                            third  = tree.data[i,3:6]
                            second = (first+third)/2
                            ptss.append(second.tolist())
                            #cell_list_outter.update(set(tree.boundary.cell_lookup.query(first,k=tree.boundary.tet_verts.shape[0]//2)))
                            #cell_list_outter.update(set(tree.boundary.cell_lookup.query(third,k=tree.boundary.tet_verts.shape[0]//2)))
                            #cell_list_outter.update(set(tree.boundary.cell_lookup.query(second,k=tree.boundary.tet_verts.shape[0]//2)))
                            #cell_list_outter.update(set(tree.boundary.cell_lookup.query_ball_point(first,threshold+threshold*0.5)))
                            #cell_list_outter.update(set(tree.boundary.cell_lookup.query_ball_point(third,threshold+threshold*0.5)))
                            #d,id = tree.boundary.cell_lookup.query(second,5000)
                            #id = id.tolist()
                            #id = id[::-1]
                            #cell_list_outter.update(set(id[:200]))
                            cell_list_outter.update(set(tree.boundary.cell_lookup.query_ball_point(second,threshold)))
                            if threshold > ((tree.boundary.volume)**(1/3)/(number_edges**threshold_exponent))*(threshold_adjuster**5):
                                #pass
                                #cell_list_outter.difference_update(set(tree.boundary.cell_lookup.query_ball_point(first,threshold)))
                                #cell_list_outter.difference_update(set(tree.boundary.cell_lookup.query_ball_point(third,threshold)))
                                cell_list_outter.difference_update(set(tree.boundary.cell_lookup.query_ball_point(second,threshold*threshold_adjuster)))
                            ball_size = 2*threshold
                            while len(cell_list_outter) < 1:
                                cell_list_outter.update(set(tree.boundary.cell_lookup.query_ball_point(second,ball_size)))
                                ball_size *= 2
                        cell_list = list(cell_list_outter)
                        #plotter = pv.Plotter()
                        subb = tree.boundary.tet.grid.extract_cells(cell_list)
                        #plotter.add_mesh(tree.boundary.tet.grid,opacity=0.5)
                        #plotter.add_mesh(subb)
                        #point_poly = pv.PolyData(np.array(ptss))
                        #plotter.add_mesh(point_poly,color='red')
                        #plotter.show()
                        rng_points = []
                        max_attemps = min(100,len(cell_list))
                        for j in range(len(cell_list)):
                            p,_ = tree.boundary.pick_in_cell(cell_list[j])
                            rng_points.append(p[0].tolist())
                        random.shuffle(rng_points)
                    threshold *= threshold_adjuster
                    attempt = 0
                    #print(threshold)
                ##################
                ##################
                #print("Below Threshold: {}".format(line_distances_below_threshold))
                #print("Minimum Distances: {}  Threshold: {}".format(minimum_line_distance,radius_scale*tree.data[vessel[0],21]))
                if (line_distances_below_threshold == 0 and
                    minimum_line_distance > radius_scale*tree.data[vessel[0],21]):
                    escape = False
                    start3 = time.time()
                    for i in range(max_skip):
                        if flow_ratio is not None:
                            if tree.data[vessel[i],22] < flow_ratio*Qterm:
                                vessel = vessel[i]
                                escape = True
                                break

                        else:
                            vessel = vessel[i]
                            escape = True
                            break
                    time3 += time.time() - start3
                    if escape:
                        #print('viable')
                        break
                    """
                    if attempt < max_attemps:
                        attempt += 1
                        total_attempts += 1
                    else:
                        #print('adjusting threshold')
                        if not tree.convex:
                            p0 = (tree.data[vessel[i],0:3]+tree.data[vessel[i],3:6])/2
                            p1 = point
                            path,lengths,res,pf,f,rm = boundary.find_best_path(p0,p1,niter=10)
                            path = np.array(path)
                            data = np.vstack((t.data,np.zeros((path,tree.data.shape[1]))))
                            data,sub_div_ind,sub_div_map = add_geodesic_path(data,path,lengths,vessel[i],
                                                                             sub_division_index,sub_division_map)
                            return vessel[i],data,sub_div_ind,sub_div_map
                        threshold_distance *= threshold_adjuster
                        attempt = 0
                    """
                else:
                    #print('line distance')
                    start4 = time.time()
                    if attempt < max_attemps:
                        attempt += 1
                        total_attempts += 1
                    else:
                        #print('adjusting threshold')
                        attempt = 0
                        threshold *= threshold_adjuster
                        continue
                        #attempt = 0
                    time4 += time.time() - start4
            search_time += time.time()-start
            start = time.time()
            if not isinstance(vessel,np.int64):
                if len(vessel) > 1:
                    vessel = vessel[0]
            proximal = tree.data[vessel,0:3]
            distal   = tree.data[vessel,3:6]
            terminal = point
            #print(distal)
            #print(type(distal))
            #print(terminal)
            #print(type(terminal))
            if np.all(terminal.shape != distal.shape):
                terminal = terminal.flatten()
            points   = get_local_points(tree.data,vessel,terminal,sampling,tree.clamped_root)
            #high_res_points = get_local_points(tree.data,vessel,terminal,10*sampling,tree.clamped_root)
            #points   = np.array(relative_length_constraint(points,proximal,distal,terminal,0.25))
            #P = forest.show()
            #P.add_points(points,render_points_as_spheres=True,point_size=20)
            #P.add_points(terminal,render_points_as_spheres=True,point_size=20,color='g')
            #P.show()
            points   = np.array(relative_length_constraint(points,proximal,distal,terminal,0.25))
            #high_res_points = np.array(relative_length_constraint(high_res_points,proximal,distal,terminal,0.25))
            #P = forest.show()
            #P.add_points(points,render_points_as_spheres=True,point_size=20)
            #P.add_points(terminal,render_points_as_spheres=True,point_size=20,color='g')
            #P.show()
            if not tree.convex:
                points = boundary_constraint(points,tree.boundary,2)
                #high_res_points = boundary_constraint(high_res_points,tree.boundary,2)
            if len(points) == 0:
                attempt += 1
                #print('constraint 1')
                continue
            if vessel != 0 and not tree.clamped_root:
                points = np.array(angle_constraint(points,terminal,distal,-0.4,True))
                #high_res_points = np.array(angle_constraint(high_res_points,terminal,distal,-0.4,True))
            if len(points) == 0:
                attempt += 1
                #print('constraint 2')
                continue
            if vessel != 0 and not tree.clamped_root:
               points = np.array(angle_constraint(points,terminal,distal,0.75,False))
               #high_res_points = np.array(angle_constraint(high_res_points,terminal,distal,0.75,False))
            if len(points) == 0:
                attempt += 1
                #print('constraint 3')
                continue
            if vessel != 0 and not tree.clamped_root:
                points = np.array(angle_constraint(points,terminal,proximal,0.2,False))
                #high_res_points = np.array(angle_constraint(high_res_points,terminal,proximal,0.2,False))
            if len(points) == 0:
                attempt += 1
                #print('constraint 4')
                continue
            if vessel != 0 and not tree.clamped_root:
                points = np.array(angle_constraint(points,distal,proximal,0.2,False))
                #high_res_points = np.array(angle_constraint(high_res_points,distal,proximal,0.2,False))
            if len(points) == 0:
                attempt += 1
                #print('constraint 5')
                continue
            if tree.data[vessel,17] >= 0:
                p_vessel = int(tree.data[vessel,17])
                vector_1 = -tree.data[p_vessel,12:15]
                vector_2 = (points - proximal)/np.linalg.norm(points - proximal,axis=1).reshape(-1,1)
                #vector_3 = (high_res_points - proximal)/np.linalg.norm(high_res_points - proximal,axis=1).reshape(-1,1)
                angle = np.array([np.dot(vector_1,vector_2[i]) for i in range(len(vector_2))])
                #high_res_angle = np.array([np.dot(vector_1,vector_3[i]) for i in range(len(vector_3))])
                points = points[angle<0]
                #high_res_points = high_res_points[high_res_angle<0]
                if len(points) == 0:
                    attempt += 1
                    #print('constraint 6')
                    continue


            tmp_points = []
            #high_res_tmp_points = []
            if not tree.convex:
                for pt in range(points.shape[0]):
                    if tree.boundary.within(points[pt,0],points[pt,1],points[pt,2],2):
                        tmp_points.append(points[pt,:])
                points = np.array(tmp_points)
                #for pt in range(high_res_points.shape[0]):
                #    if tree.boundary.within(high_res_points[pt,0],high_res_points[pt,1],high_res_points[pt,2],2):
                #        high_res_tmp_points.append(high_res_points[pt,:])
                #high_res_points = np.array(high_res_tmp_points)
                if len(points) == 0:
                    attempt += 1
                    #print('constraint 7')
                    continue
            tmp_points = []
            subdivision = 5
            #plotter = pv.Plotter()
            #polys = pv.PolyData(points)
            #term_poly = pv.PolyData(terminal)
            #plotter.add_mesh(tree.boundary.tet.grid,opacity=0.25)
            #plotter.add_mesh(polys,color='red')
            #plotter.add_mesh(term_poly,color='green')
            #if subb is not None:
            #    plotter.add_mesh(subb,color='blue',opacity=0.25)
            #plotter.show()
            if not tree.convex:
                for pt in range(points.shape[0]):
                    include = True
                    for sub in range(1,2*subdivision):
                        mid_proximal = points[pt,:]*(sub/(2*subdivision))+proximal*(1-sub/(2*subdivision))
                        mid_distal = points[pt,:]*(sub/(2*subdivision))+distal*(1-sub/(2*subdivision))
                        mid_terminal = points[pt,:]*(sub/(2*subdivision))+terminal*(1-sub/(2*subdivision))
                        mid_proximal = mid_proximal.flatten()
                        mid_distal = mid_distal.flatten()
                        mid_terminal = mid_terminal.flatten()
                        if vessel != 0:
                            if not tree.boundary.DD[0]((mid_proximal[0],mid_proximal[1],mid_proximal[2],len(tree.boundary.patches)//10)) < 0.1:
                                #print('proximal')
                                include = False
                                break
                        val = tree.boundary.DD[0]((mid_distal[0],mid_distal[1],mid_distal[2],len(tree.boundary.patches)//10))
                        if not val < 0.01:
                            #print(val)
                            #plotter = pv.Plotter()
                            #plotter.add_mesh(tree.boundary.tet.grid,opacity=0.5)
                            #val_poly = pv.PolyData(mid_distal)
                            #plotter.add_mesh(sub)
                            #plotter.add_mesh(val_poly,color='yellow')
                            #plotter.show()
                            #print('distal')
                            include = False
                            break
                        if not tree.boundary.DD[0]((mid_terminal[0],mid_terminal[1],mid_terminal[2],len(tree.boundary.patches)//10)) < 0.01:
                            #print('terminal')
                            include = False
                            break
                    if include:
                        tmp_points.append(points[pt,:])
                points = np.array(tmp_points)
            #plotter.show()
            if len(points) == 0:
                attempt += 1
                nonconvex_solve=False
                #print('constraint 8')
                continue
            #print('passed all constraints')

            constraint_time += time.time()-start
            start = perf_counter()
            #construct_results = constructor(tree.data,terminal,
            #                      vessel,gamma,nu,Qterm,Pperm,Pterm,lam,mu,sampling,method=method)
            #print('Constructor Time: {}'.format(perf_counter()-start))
            construct_time=perf_counter()-start
            #print("Minimize x: {} Value: {}".format(results[5],np.pi*results[0]**lam*results[1]**mu))
            #start = perf_counter()
            #fd_point,fd_idx,fd_volume,fd_trial = finite_difference(tree.data,points,terminal,
            #                                                       vessel,gamma,nu,Qterm,Pperm,Pterm)
            #fd_time = perf_counter()
            #print('Finite Difference Time: {}'.format(perf_counter()-start))
            #print("Finite difference: {} Value: {}".format(fd_point,fd_volume[fd_idx]))
            #start = time.time()
            start = perf_counter()
            results = fast_local_function(tree.data,points,terminal,
                                          vessel,gamma,nu,Qterm,Pperm,Pterm)
            brute_time = perf_counter()-start
            #truth_results = fast_local_function(tree.data,high_res_points,terminal,
            #                                    vessel,gamma,nu,Qterm,Pperm,Pterm)
            #print('Brute Time: {}'.format(perf_counter()-start))
            local_time += time.time()-start
            volume  = np.pi*(results[0]**lam)*(results[1]**mu)
            brute = volume
            idx     = np.argmin(volume)
            bif     = results[5][idx]
            #truth_volume = np.pi*(truth_results[0]**lam)*(truth_results[1]**mu)
            #truth_idx = np.argmin(truth_volume)
            #truth_bif = truth_results[5][truth_idx]
            #truth_best_vol = truth_volume[truth_idx]
            #print("Brute x: {} Value: {}".format(bif,min(volume)))
            start = time.time()
            no_collision = collision_free(tree.data,results,idx,terminal,
                                          vessel,radius_buffer)
            collision_time += time.time()-start

            if no_collision:
                new_branch_found = True
                start = time.time()
                data,sub_division_map,sub_division_index = add_bifurcation(tree,vessel,terminal,
                                                                           results,idx,isforest=isforest)
                #brute = np.sum(np.pi*data[:,21]**lam*data[:,20]**mu)
                #print("Brute x: {} Value: {}".format(bif,brute[idx]))
                #fig = plt.figure()
                #ax1 = fig.add_subplot(121)
                #x = np.linspace(min(points[:,0]),max(points[:,0]),100)
                #y =  np.linspace(min(points[:,1]),max(points[:,1]),100)
                #X, Y = np.meshgrid(x,y)
                #fd_Ti = griddata((points[:,0],points[:,1]),fd_volume,(X,Y),method='linear')
                #ax1.contour(X,Y,fd_Ti,linewidths=0.5,colors='k')
                #ax1.pcolormesh(X,Y,fd_Ti,shading='auto',cmap=plt.get_cmap('rainbow'))
                #ax1.colorbar()
                #brute_Ti = griddata((points[:,0],points[:,1]),brute,(X,Y),method='linear')
                #ax2 = fig.add_subplot(122)
                #ax2.contour(X,Y,brute_Ti,linewidths=0.5,colors='k')
                #ax2.pcolormesh(X,Y,brute_Ti,shading='auto',cmap=plt.get_cmap('rainbow'))
                #ax2.colorbar()
                #plt.colorbar()
                #plt.show()
                add_time += time.time()-start
                total_time += time.time()-start_total
                tree.time['search'].append(search_time)
                tree.time['constraints'].append(constraint_time)
                tree.time['local_optimize'].append(local_time)
                tree.time['collision'].append(collision_time)
                tree.time['close_time'].append(close_time)
                tree.time['search_1'].append(time1)
                tree.time['search_2'].append(time2)
                tree.time['search_3'].append(time3)
                tree.time['search_4'].append(time4)
                tree.time['add_time'].append(add_time)
                tree.time['total'].append(total_time)
                #tree.time['brute_time'].append(brute_time)
                #tree.time['method_time'].append(construct_time)
                #tree.time['depth'].append(data[vessel,26])
                #tree.time['method_x_value'].append(construct_results[5].flatten())
                #tree.time['method_value'].append(np.pi*construct_results[0]**lam*construct_results[1]**mu)
                #tree.time['brute_x_value'].append(bif)
                #tree.time['brute_value'].append(brute[idx])
                #tree.time['truth_x_value'].append(truth_bif)
                #tree.time['truth_value'].append(truth_best_vol)
                if nonconvex_solve:
                    tree.nonconvex_counter += 1
                else:
                    tree.nonconvex_counter = 0
                if tree.nonconvex_counter > 100 or tree.convex:
                    tree.convex =True
                else:
                    tree.convex =False
                #print("returning")
                return vessel,data,sub_division_map,sub_division_index,threshold
            else:
                #print('collision')
                attempt += 1
                continue
    else:
        reduced_data  = tree.data[tree.data[:,-1]>-1]
        segment_data  = tree.data[tree.data[:,-1]==-1]
        vessel        = np.random.choice(list(range(reduced_data.shape[0])))
        vessel_path   = segment_data[segment_data[:,29].astype(int)==vessel]
        other_vessels = segment_data[segment_data[:,29].astype(int)!=vessel]
        if reduced_data.shape[0] > 1:
            other_KDTree = KDTree((other_vessels[:,0:3]+other_vessels[:,3:6])/2)
        else:
            other_KDTree = None
        mesh,pa,cp,cd = tree.boundary.mesh(vessel_path[1:,0:3],threshold,threshold//fraction,dive=0,others=other_KDTree)
        D,PR = shortest_path(graph,directed=False,method="D",return_predecessors=True)
        bif_idx = set(list(range(mesh.shape[0])))
