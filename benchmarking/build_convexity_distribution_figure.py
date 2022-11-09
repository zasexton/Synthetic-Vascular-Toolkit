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
nonconvex_directories = list(filter(lambda n: 'data' in n,nonconvex_directories))

#nonconvex_directories = os.walk(os.getcwd()+os.sep+'brain_data')
#nonconvex_directories = [x[0] for x in nonconvex_directories]
#nonconvex_directories = list(filter(lambda n: 'times' in n,nonconvex_directories))

#linecolors = []
#red = cm.Reds
#for l in range(len(convex_directories)):
#    linecolors.append(red(256-l))

#blue = cm.Blues
#for l in range(len(nonconvex_directories)):
#    linecolors.append(blue(256-l))

#convex_directories.extend(nonconvex_directories)
#directories  = convex_directories

directories = nonconvex_directories

#linestyles = [(0,())]*len(directories)

#for l in range(len(nonconvex_directories)):
#    if l == 0:
#        linestyles.append(tuple([0,tuple([])]))
#    else:
#        linestyles.append(tuple([0,tuple([3,l])]))

#capsize = 5

###############################################
# Below is just code to unpack data
# there is no need to touch this.
###############################################

DATA = []
data = []

for dirname in directories:
    if 'stats.pkl' not in os.listdir(dirname):
        continue
    file = dirname + os.sep + 'stats.pkl'
    f = open(file,'rb')
    conn = load(f)
    DATA.append((conn[3])/(conn[1]))
    #if DATA[-1] > 2:
    #    print(dirname)
    dir_adjust = dirname.split(os.sep)[:-1]
    name = ''
    for n in dir_adjust:
        name += n + os.sep
    dir_adjust = name + 'times'
    files = os.listdir(dir_adjust)
    #print(dir_adjust)
    tmp = np.zeros((len(files),len(tree_sizes)))
    for i, fname in tqdm(enumerate(files),desc='loading {} directory'.format(dir_adjust)):
        f = open(dir_adjust+os.sep+fname,'rb')
        time_dat = load(f)
        for j, size in enumerate(tree_sizes):
            tmp[i,j] = sum(time_dat['total'][:size])
    data.append(np.mean(tmp[:,-1]))
    #data[dirname+'_std']  = np.std(data[dirname],axis=0)
    #print('Time')
    #print(data[dirname+'_mean'][-1]/60)
    #print('Convexity')
    #print(DATA[-1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(DATA)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_ylabel('Number of Regions')
ax.set_xlabel('Convexity')
#ax.grid(which='both')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(DATA,data)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_ylabel('Build Time (s)')
ax.set_xlabel('Convexity')
plt.show()
