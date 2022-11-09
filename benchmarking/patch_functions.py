# Patch functions
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator
import matplotlib.ticker as mticker

def gen_fig(h=0,n_lines=9):
    r = np.linspace(-2,2,num=1000)
    fig,ax = plt.subplots()
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    for q in range(n_lines):
        w = (1-np.tanh(abs(r)))/(abs(r)+h)**q
        ax.plot(r,w,c=cmap.to_rgba(q+1),label="q={}".format(q))
    ax.set_xlabel(r'Normalized Distance, $r$')
    ax.set_ylabel('Weight')
    ax.set_yscale('log')
    fig.colorbar(cmap, ticks=c,label=r"Patch Degree, $q$")
    name = "Patch_functions_q{}_h{}.svg".format(n_lines,h)
    fig.savefig(name,format="svg")
    plt.show()

def log_tick_formatter(val, pos=None):
    return f"10$^{{{int(val)}}}$"

def gen_3d_fig(h=0,q=1,n_lines=9):
    x_interval = (-1, 1)
    y_interval = (-1, 1)
    x_points = np.linspace(x_interval[0], x_interval[1], 1000)
    y_points = np.linspace(y_interval[0], y_interval[1], 1000)
    X, Y = np.meshgrid(x_points, y_points)
    #r = (X**2 + Y**2)**(1/2)
    #Z = (1-np.tanh(r))/(r+h)**q
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #for q in range(n_lines):
    r = (X**2 + Y**2)**(1/2)
    Z = (1-np.tanh(r))/(r+h)**q
    ax.plot_surface(X,Y,np.log10(Z),color=cmap.to_rgba(q+1),label="q={}".format(q))
    ax.grid(False)
    Zmin = np.where(Z > 0, Z, np.inf).min()
    #print(np.ceil(np.log10(Z.max())))
    #print(np.floor(np.log10(Zmin))-1)
    ax.set_zlim3d([np.floor(np.log10(Zmin)),np.ceil(np.log10(Z.max()))])
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(MultipleLocator(1))
    zminorticks = []
    zaxmin, zaxmax = ax.get_zlim()
    for zorder in np.arange(np.floor(zaxmin),np.ceil(zaxmax)):
        zminorticks.extend(np.log10(np.linspace(2,9,8)) + zorder)
    ax.zaxis.set_minor_locator(FixedLocator(zminorticks))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    name = "3D_Patch_functions_q{}_h{}.svg".format(q,h)
    fig.savefig(name,format="svg")
    plt.show()

gen_fig(h=0)
gen_fig(h=1)
gen_3d_fig(h=0,q=1)
gen_3d_fig(h=0,q=8)
gen_3d_fig(h=1,q=1)
gen_3d_fig(h=1,q=8)
