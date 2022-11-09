# Code: Build Time Figure Between Convex and Nonconvex geometries

import numpy as np
from pickle import load
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import os

###############################################
# Parameters to change for figure display
###############################################

tree_sizes = [100,1000,4000,7999]

#convex_directories    = [os.getcwd()+os.sep+'cube_times',os.getcwd()+os.sep+'sphere_times']
convex_directories = os.walk(os.getcwd()+os.sep+'convex_data')
convex_directories = [x[0] for x in convex_directories]
convex_directories = list(filter(lambda n: 'times' in n,convex_directories))

nonconvex_directories = os.walk(os.getcwd()+os.sep+'brain_data')
nonconvex_directories = [x[0] for x in nonconvex_directories]
nonconvex_directories = list(filter(lambda n: 'times' in n,nonconvex_directories))

linecolors = []
red = cm.Reds
for l in range(len(convex_directories)):
    linecolors.append(red(256-l))

blue = cm.Blues
for l in range(len(nonconvex_directories)):
    linecolors.append(blue(256-l))

convex_directories.extend(nonconvex_directories)
directories  = convex_directories


linestyles = [(0,())]*len(directories)

#for l in range(len(nonconvex_directories)):
#    if l == 0:
#        linestyles.append(tuple([0,tuple([])]))
#    else:
#        linestyles.append(tuple([0,tuple([3,l])]))

capsize = 5

###############################################
# Below is just code to unpack data
# there is no need to touch this.
###############################################

DATA = {}

for dirname in directories:
    files = os.listdir(dirname)
    DATA[dirname] = np.zeros((len(files),len(tree_sizes)))
    for i, fname in tqdm(enumerate(files),desc='loading {} directory'.format(dirname)):
        f = open(dirname+os.sep+fname,'rb')
        time_dat = load(f)
        for j, size in enumerate(tree_sizes):
            DATA[dirname][i,j] = sum(time_dat['total'][:size])
    DATA[dirname+'_mean'] = np.mean(DATA[dirname],axis=0)
    DATA[dirname+'_std']  = np.std(DATA[dirname],axis=0)

fig = plt.figure()
ax = fig.add_subplot(111,xscale='log',yscale='log')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_ylabel('Build Time (s)')
ax.set_xlabel('Tree Size (# Terminals)')
ax.grid(which='both')
for i, dirname in enumerate(directories):
    #ax.plot(tree_sizes,DATA[dirname+'_mean'],color=linecolors[i],linestyle=linestyles[i])
    ax.errorbar(tree_sizes,DATA[dirname+'_mean'],yerr=DATA[dirname+'_std'],
                capsize=capsize,ecolor=linecolors[i],linestyle=linestyles[i],
                color=linecolors[i])

plt.show()
