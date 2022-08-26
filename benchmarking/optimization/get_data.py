import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

wd = os.getcwd()

nelder_mead = []
powell      = []
bfgs        = []
newton_cg   = []
lbfsgb      = []
tnc         = []
cobyla      = []
slsqp       = []
dogleg      = []
trust_ncg   = []

data = {'Nelder-Mead':{},
        'Powell':{},
        'BFGS':{},
        'Newton-CG':{},
        'L-BFGS-B':{},
        'TNC':{},
        'COBYLA':{},
        'SLSQP':{}}

solvers = ['Nelder-Mead','Powell','BFGS','Newton-CG','L-BFGS-B','TNC',
           'COBYLA','SLSQP']

results = []
for sol in solvers:
    tmp_results = []
    for res in glob.glob(wd+os.sep+sol+os.sep+"*"):
        file = open(res,"rb")
        tmp_results.append(pickle.load(file))
        file.close()
    results.append(tmp_results)

# Get max depth ranges for data
for i in range(len(results)):
    max_depth = 0
    for j in range(len(results[i])):
        max_val = max(results[i][j]['depth'])
        if max_val > max_depth:
            max_depth = max_val
    max_depth = int(max_depth)
    data[solvers[i]]['depth'] = list(range(0,max_depth+1))
    data[solvers[i]]['method_time_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
    data[solvers[i]]['brute_time_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
    data[solvers[i]]['method_error_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
    data[solvers[i]]['brute_error_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
    data[solvers[i]]['method_align_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
    data[solvers[i]]['brute_align_bin']   = [[] for k in range(len(data[solvers[i]]['depth']))]
# allocate data into correct bins
for i in range(len(results)):
    for j in range(len(results[i])):
        for k in range(len(results[i][j]['depth'])):
            data[solvers[i]]['method_time_bin'][int(results[i][j]['depth'][k])].append(results[i][j]['method_time'][k])
            data[solvers[i]]['brute_time_bin'][int(results[i][j]['depth'][k])].append(results[i][j]['brute_time'][k])
            data[solvers[i]]['method_error_bin'][int(results[i][j]['depth'][k])].append(abs(results[i][j]['truth_value'][k].item() - results[i][j]['method_value'][k])/abs(results[i][j]['truth_value'][k].item()))
            data[solvers[i]]['brute_error_bin'][int(results[i][j]['depth'][k])].append(abs(results[i][j]['truth_value'][k].item() - results[i][j]['brute_value'][k])/abs(results[i][j]['truth_value'][k].item()))
            data[solvers[i]]['method_align_bin'][int(results[i][j]['depth'][k])].append(np.linalg.norm(results[i][j]['truth_x_value'][k] - results[i][j]['method_x_value'][k]))
            data[solvers[i]]['brute_align_bin'][int(results[i][j]['depth'][k])].append(np.linalg.norm(results[i][j]['truth_x_value'][k] - results[i][j]['brute_x_value'][k]))

    data[solvers[i]]['method_time_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['method_time_bin']]).flatten()
    data[solvers[i]]['brute_time_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['brute_time_bin']]).flatten()
    data[solvers[i]]['method_error_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['method_error_bin']]).flatten()
    data[solvers[i]]['brute_error_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['brute_error_bin']]).flatten()
    data[solvers[i]]['method_time_sd'] = np.array([np.std(times) for times in data[solvers[i]]['method_time_bin']]).flatten()
    data[solvers[i]]['brute_time_sd'] = np.array([np.std(times) for times in data[solvers[i]]['brute_time_bin']]).flatten()
    data[solvers[i]]['method_error_sd'] = np.array([np.std(times) for times in data[solvers[i]]['method_error_bin']]).flatten()
    data[solvers[i]]['brute_error_sd'] = np.array([np.std(times) for times in data[solvers[i]]['brute_error_bin']]).flatten()
    data[solvers[i]]['method_align_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['method_align_bin']]).flatten()
    data[solvers[i]]['brute_align_mean'] = np.array([np.mean(times) for times in data[solvers[i]]['brute_align_bin']]).flatten()
    data[solvers[i]]['method_align_sd'] = np.array([np.std(times) for times in data[solvers[i]]['method_align_bin']]).flatten()
    data[solvers[i]]['brute_align_sd'] = np.array([np.std(times) for times in data[solvers[i]]['brute_align_bin']]).flatten()
fig = plt.figure(figsize=[11,9])
nrows = len(solvers)
ncols = 3
idx   = 1
for i in range(len(results)):
    depth = data[solvers[i]]['depth']
    method_times = data[solvers[i]]['method_time_mean']
    brute_times = data[solvers[i]]['brute_time_mean']
    method_times_sd = data[solvers[i]]['method_time_sd']
    brute_times_sd = data[solvers[i]]['brute_time_sd']
    method_error = data[solvers[i]]['method_error_mean']
    brute_error = data[solvers[i]]['brute_error_mean']
    method_error_sd = data[solvers[i]]['method_error_sd']
    brute_error_sd = data[solvers[i]]['brute_error_sd']
    method_align = data[solvers[i]]['method_align_mean']
    brute_align = data[solvers[i]]['brute_align_mean']
    method_align_sd = data[solvers[i]]['method_align_sd']
    brute_align_sd = data[solvers[i]]['brute_align_sd']
    plt.subplot(nrows,ncols,idx)
    plt.plot(depth,method_times,label=solvers[i])
    plt.fill_between(depth,method_times-method_times_sd,method_times+method_times_sd,alpha=0.5)
    plt.plot(depth,brute_times,label="brute")
    plt.fill_between(depth,brute_times-brute_times_sd,brute_times+brute_times_sd,alpha=0.5)
    plt.legend(loc='upper right',prop={'size':6})
    plt.ylim([0,None])
    idx += 1
    plt.subplot(nrows,ncols,idx)
    plt.plot(depth,method_error,label=solvers[i])
    plt.fill_between(depth,method_error-method_error_sd,method_error+method_error_sd,alpha=0.5)
    plt.plot(depth,brute_error,label="brute")
    plt.fill_between(depth,brute_error-brute_error_sd,brute_error+brute_error_sd,alpha=0.5)
    plt.legend(loc='upper right',prop={'size':6})
    plt.ylim([0,None])
    idx += 1
    plt.subplot(nrows,ncols,idx)
    plt.plot(depth,method_align,label=solvers[i])
    plt.fill_between(depth,method_align-method_align_sd,method_align+method_align_sd,alpha=0.5)
    plt.plot(depth,brute_align,label="brute")
    plt.fill_between(depth,brute_align-brute_align_sd,brute_align+brute_align_sd,alpha=0.5)
    plt.legend(loc='upper right',prop={'size':6})
    plt.ylim([0,None])
    idx += 1
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
fig.text(0.5,0.02,"Bifurcation Depth",ha="center")
fig.text(0.02,0.5,"Time (seconds)",va="center",rotation="vertical")
fig.text(0.328,0.5,"Relative Volume Error",va="center",rotation="vertical")
fig.text(0.625,0.5,"Absolute Alignment Error",va="center",rotation="vertical")
plt.show()
