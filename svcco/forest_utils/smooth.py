# smooth connections between trees in a network
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
import numpy as np

def smooth(forest,curve_sample_size_min=5,curve_sample_size_max=11,curve_degree=2):
    connections = []
    for network in range(forest.number_of_networks):
        if not np.any(forest.connections[network][0] is None):
            network_connections = []
            for tree in range(forest.trees_per_network[network]):
                tree_assignment = forest.assignments[network][tree]
                idx = max(forest.networks[network][tree].data[:,-1]) + 1
                #tree_connection_edges = np.zeros((len(forest.connections[network][0])*(curve_sample_size-1),forest.networks[network][tree].data.shape[1]))
                tree_connection_edges = []
                for edge in range(len(forest.connections[network][0])):
                    #print('Tree {}: Edge: {}'.format(tree,edge))
                    success = False
                    samples = curve_sample_size_min
                    ctr_1 = ((forest.networks[network][tree].data[tree_assignment[edge],0:3]+forest.networks[network][tree].data[tree_assignment[edge],3:6])/2).flatten().tolist()
                    ctr_2 = forest.networks[network][tree].data[tree_assignment[edge],3:6].flatten().tolist()
                    ctr_3 = forest.connections[network][0][edge].tolist()
                    curve = BSpline.Curve()
                    curve.degree = curve_degree
                    curve.ctrlpts = [ctr_1,ctr_2,ctr_3]
                    curve.knotvector = utilities.generate_knot_vector(curve.degree,len(curve.ctrlpts))
                    while samples <= curve_sample_size_max and not success:
                        curve.sample_size = samples
                        curve.evaluate()
                        pts = curve.evalpts
                        d0  = forest.networks[network][tree].data[tree_assignment[edge],12:15]
                        for p in range(len(pts)-1):
                            p0 = np.array(pts[p])
                            p1 = np.array(pts[p+1])
                            l  = np.linalg.norm(p1-p0)
                            d1 = (p1-p0)/l
                            A  = (abs(np.arccos(np.dot(-d0,d1)))/np.pi)*180
                            if A < 90:
                                success = False
                                samples += 2
                                break
                            else:
                                success = True
                            d0 = d1
                    if samples > curve_sample_size_max:
                        samples -= 2
                    for i in range(samples-1):
                        tmp = np.zeros(forest.networks[network][tree].data.shape[1])
                        tmp[0:3] = np.array(pts[i])
                        tmp[3:6] = np.array(pts[i+1])
                        tmp[21]  = forest.networks[network][tree].data[tree_assignment[edge],21]
                        tmp[20]  = np.linalg.norm(np.array(pts[i+1])-np.array(pts[i]))
                        tmp[12:15] = (np.array(pts[i+1])-np.array(pts[i]))/tmp[20]
                        tmp[26] = forest.networks[network][tree].data[tree_assignment[edge],26]
                        tmp[-1] = idx
                        tmp[15] = -1
                        tmp[16] = -1
                        idx += 1
                        if i == 0:
                            tmp[17] = forest.networks[network][tree].data[tree_assignment[edge],-1]
                        elif i > 0:
                            tree_connection_edges[-1][15] = tmp[-1]
                            tree_connection_edges[-1][16] = -1
                            tmp[17] = tree_connection_edges[-1][-1]
                        #else:
                        #    tree_connection_edges[-1][15] = -1
                        #    tree_connection_edges[-1][16] = -1
                        #    tmp[17] = tree_connection_edges[-1][-1]
                        tree_connection_edges.append(tmp.tolist())
                network_connections.append(np.array(tree_connection_edges))
            connections.append(network_connections)
        else:
            connections.append([None])
    return connections
