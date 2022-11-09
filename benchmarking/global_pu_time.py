import numpy as np
import svcco
from time import perf_counter as time
import matplotlib.pyplot as plt
from tqdm import tqdm

def global_pu_time(start,stop):
    sizes = np.arange(start,stop)
    num_points = []
    pu_time = []
    gu_time = []
    pur_time = []
    gur_time = []
    #warmup the code for jit compilation
    sph = svcco.implicit.tests.bumpy_sphere.bumpy_sphere(10)
    s = svcco.surface()
    s.set_data(sph)
    s.solve()
    s.solve(PU=False)
    for si in tqdm(sizes):
        sph = svcco.implicit.tests.bumpy_sphere.bumpy_sphere(samples=si)
        num_points.append(sph.shape[0])
        s = svcco.surface()
        s.set_data(sph)
        begin = time()
        s.solve()
        pu_time.append(time() - begin)
        #begin = time()
        #s.solve(regularize=True)
        #pur_time.append(time() - begin)
        begin = time()
        s.solve(PU=False)
        gu_time.append(time() - begin)
        #begin = time()
        #s.solve(PU=False,regularize=True)
        #gur_time.append(time() - begin)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(num_points,pu_time,label='Partition of Unity')
    #ax.plot(num_points,pur_time,label='Regularized Partition of Unity')
    ax.loglog(num_points,gu_time,label='Global')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Time (sec)')
    #ax.plot(num_points,gur_time,label='Regularize Global')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('PU_vs_Global_Time_plot.svg', format="svg", bbox_inches='tight')
    return num_points,pu_time,pur_time,gu_time,gur_time,
