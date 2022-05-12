# take nth order derivatives
from sympy import symbols,Function,tanh
from sympy import diff as D
import numpy as np
#(xyzk,KDTree=None,patch_points=None,b_coef=None,h=None,
# q=None,sol_mat=None,patch_x=None,patch_y=None,patch_z=None):
def outter_parentheses(string):
    open_p = [i for i, ch in enumerate(string) if ch == "("]
    close_p = [i for i, ch in enumerate(string) if ch == ")"]
    pairs = []
    while len(close_p) > 0:
        j = close_p[0]
        i = None
        jj = j
        while i == None:
            i = string[:jj].rfind("(")
            if i in open_p:
                break
            else:
                jj = i
                i = None
        if i == open_p[0]:
            pairs.append((i,j+1))
        close_p.remove(j)
        open_p.remove(i)
    return pairs

def global_d(order):
    x,y,z    = symbols('x y z')
    xi,yi,zi = symbols('xi yi zi')
    A,B1,B2,B3,C1,C2,C3,C4 = symbols('A B1 B2 B3 C1 C2 C3 C4')
    r  = ((x-xi)**2+(y-yi)**2+(z-zi)**2)**(1/2)
    rr = ((x-xi)**2+(y-yi)**2+(z-zi)**2)
    dfix = D(r**3,x)
    dfiy = D(r**3,y)
    dfiz = D(r**3,z)
    fi = -tanh(A*r**3+B1*dfix+B2*dfiy+B3*dfiz+C1*x+C2*y+C3*z+C4)
    fi_inner = A*r**3+B1*dfix+B2*dfiy+B3*dfiz+C1*x+C2*y+C3*z+C4
    fi_inner = str(fi_inner)
    fi_inner = fi_inner.replace(str(rr),'rr')
    fs = str(fi_inner)
    vars = [x,y,z]
    del_orders_f = [[fi]]
    del_notation = [['f']]
    for i in range(order):
        tmp_f = []
        tmp_n = []
        for idx,p in enumerate(del_orders_f[-1]):
            for jdx,v in enumerate(vars):
                tmp_f.append(D(p,v))
                if i == 0:
                   tmp_n.append('d'+del_notation[-1][idx]+str(v))
                else:
                   tmp_n.append(del_notation[-1][idx]+str(v))
        del_orders_f.append(tmp_f)
        del_notation.append(tmp_n)
    key = {str(rr) :'rr',
           'tanh'  :'np.tanh',
           fi_inner:'f'}
    var = {'A' :'sol_mat[0,:,0]',
           'B1':'sol_mat[0,:,1]',
           'B2':'sol_mat[0,:,2]',
           'B3':'sol_mat[0,:,3]',
           'C1':'b_coef[0,:,1]',
           'C2':'b_coef[0,:,2]',
           'C3':'b_coef[0,:,3]',
           'C4':'b_coef[0,:,0]',
           'xi':'patch_points[0,:,0]',
           'yi':'patch_points[0,:,1]',
           'zi':'patch_points[0,:,2]',
           'x' :'pt_3[0,:,0]',
           'y' :'pt_3[0,:,1]',
           'z' :'pt_3[0,:,2]',}
    for i in range(len(del_orders_f)):
        for j in range(len(del_orders_f[i])):
            del_orders_f[i][j] = str(del_orders_f[i][j])
            for item in key.keys():
                if item in del_orders_f[i][j]:
                    del_orders_f[i][j] = del_orders_f[i][j].replace(item,key[item])
            for item in var.keys():
                if item in del_orders_f[i][j]:
                    del_orders_f[i][j] = del_orders_f[i][j].replace(item,var[item])
                if item in fs:
                    fs = fs.replace(item,var[item])
    return del_orders_f,del_notation,fs

def construct_global(order,dim=3):
    funcs,nots,fs = global_d(order)
    functions = []
    func_strings = []
    for n in range(len(funcs)):
        d_func = 'def d_{}(xyz,'.format(n)+\
                           'patch_points=None,b_coef=None,'+\
                           'sol_mat=None):\n'
        d_func += '    x,y,z = xyz\n'
        d_func += '    pt = np.ones((1,b_coef.shape[-1]))\n'
        d_func += '    pt[:,1] *= x\n'
        d_func += '    pt[:,2] *= y\n'
        d_func += '    pt[:,3] *= z\n'
        d_func += '    pt_3 = np.array([[[x,y,z]]])\n'
        d_func += '    rr = np.sum(np.square(pt_3-patch_points[:,:]),axis=2)\n'
        if n == 0:
            d_func += '    {} = {}\n'.format(nots[0][0],fs)
            d_func += '    {} = np.sum({},axis=1)\n'.format(nots[0][0],nots[0][0])
        elif n > 0:
            funcs[0][0] = fs
        for i in range(n+1):
            for j in range(len(funcs[i])):
                tmp_func = funcs[i][j]
                d_func += '    {} = {}\n'.format(nots[i][j],funcs[i][j])
                if i == 0 and n > 0:
                    d_func += '    {} = np.sum({},axis=1)\n'.format(nots[i][j],funcs[i][j])
        d_func += '    res = np.zeros({})\n'.format(len(funcs[n]))
        if n > 0:
            d_func += '    {} = -np.tanh({})\n'.format(nots[0][0],nots[0][0])
        for j in range(len(funcs[n])):
            d_func += '    res_{} = {}\n'.format(nots[n][j],nots[n][j])
            d_func += '    res_{} = np.sum(res_{})\n'.format(nots[n][j],nots[n][j])
            d_func += '    res[{}] = res_{}\n'.format(j,nots[n][j])
        if n > 1:
            d_func += '    res = res.reshape({}'.format(dim)
            for c in range(n-1):
                d_func += ',{}'.format(dim)
            d_func += ')\n'
        d_func += '    return res\n'
        #d_func += 'functions.append(d_{})'.format(n)
        exec(d_func)
        func_strings.append(d_func)
    return functions,func_strings

def d(order):
    # Symbolically express nth order derivatives
    x,y,z    = symbols('x y z')
    xi,yi,zi = symbols('xi yi zi')
    xc,yc,zc = symbols('xc yc zc')
    A,B1,B2,B3,C1,C2,C3,C4 = symbols('A B1 B2 B3 C1 C2 C3 C4')
    h,q = symbols('h q')
    r  = ((x-xi)**2+(y-yi)**2+(z-zi)**2)**(1/2)
    rr = ((x-xi)**2+(y-yi)**2+(z-zi)**2)
    rc = ((x-xc)**2+(y-yc)**2+(z-zc)**2)**(1/2)
    rrc = ((x-xc)**2+(y-yc)**2+(z-zc)**2)
    dfix = D(r**3,x)
    dfiy = D(r**3,y)
    dfiz = D(r**3,z)
    fi = -tanh(A*r**3+B1*dfix+B2*dfiy+B3*dfiz+C1*x+C2*y+C3*z+C4)
    fi_inner = A*r**3+B1*dfix+B2*dfiy+B3*dfiz+C1*x+C2*y+C3*z+C4
    fi_inner = str(fi_inner)
    fi_inner = fi_inner.replace(str(rr),'rr')
    fs = str(fi_inner)
    wi = (1-tanh(rc))/(h+rc)**q
    Fi = Function('Fi')(x,y,z)
    Wi = Function('Wi')(x,y,z)
    Ki  = Function('K')(x,y,z)
    F  = (Fi*Wi)/Ki
    vars = [x,y,z]
    if order == 0:
        del_orders_fi = [[fi]]
        del_orders_wi = [[wi]]
        del_orders_F  = [[F]]
        del_orders_Fi = [[Fi]]
        del_orders_Wi = [[Wi]]
        del_orders_Ki = [[Ki]]
        notation_f    = [['f']]
        notation_w    = [['w']]
        notation_k    = [['k']]
    else:
        del_orders_fi = []
        del_orders_wi = []
        del_orders_F  = []
        del_orders_Fi = []
        del_orders_Wi = []
        del_orders_Ki = []
        notation_f    = []
        notation_w    = []
        notation_k    = []
    for i in range(order):
        tmp_del_order_fi = []
        tmp_del_order_wi = []
        tmp_del_order_F  = []
        tmp_del_order_Fi = []
        tmp_del_order_Wi = []
        tmp_del_order_Ki = []
        tmp_notation_f   = []
        tmp_notation_w   = []
        tmp_notation_k   = []
        if i == 0:
            for j in range(len(vars)):
                tmp_del_order_fi.append(D(fi,vars[j]))
                tmp_del_order_wi.append(D(wi,vars[j]))
                tmp_del_order_F.append(D(F,vars[j]))
                tmp_del_order_Fi.append(D(Fi,vars[j]))
                tmp_del_order_Wi.append(D(Wi,vars[j]))
                tmp_del_order_Ki.append(D(Ki,vars[j]))
                tmp_notation_f.append('df'+str(vars[j]))
                tmp_notation_w.append('dw'+str(vars[j]))
                tmp_notation_k.append('dk'+str(vars[j]))
        else:
            for p in range(len(del_orders_F[-1])):
                for j in range(len(vars)):
                    tmp_del_order_fi.append(D(del_orders_fi[-1][p],vars[j]))
                    tmp_del_order_wi.append(D(del_orders_wi[-1][p],vars[j]))
                    tmp_del_order_F.append(D(del_orders_F[-1][p],vars[j]))
                    tmp_del_order_Fi.append(D(del_orders_Fi[-1][p],vars[j]))
                    tmp_del_order_Wi.append(D(del_orders_Wi[-1][p],vars[j]))
                    tmp_del_order_Ki.append(D(del_orders_Ki[-1][p],vars[j]))
                    tmp_notation_f.append(notation_f[-1][p]+str(vars[j]))
                    tmp_notation_w.append(notation_w[-1][p]+str(vars[j]))
                    tmp_notation_k.append(notation_k[-1][p]+str(vars[j]))
        del_orders_fi.append(tmp_del_order_fi)
        del_orders_wi.append(tmp_del_order_wi)
        del_orders_F.append(tmp_del_order_F)
        del_orders_Fi.append(tmp_del_order_Fi)
        del_orders_Wi.append(tmp_del_order_Wi)
        del_orders_Ki.append(tmp_del_order_Ki)
        notation_f.append(tmp_notation_f)
        notation_w.append(tmp_notation_w)
        notation_k.append(tmp_notation_k)
    if order > 0:
        del_orders_fi.insert(0,[str(fi)])
        del_orders_wi.insert(0,[str(wi)])
        del_orders_F.insert(0,[str(F)])
        notation_f.insert(0,['f'])
        notation_w.insert(0,['w'])
        notation_k.insert(0,['k'])
    # Translate Derivatives into Code
    key = {str(rr) :'rr',
           str(rc):'rc',
           'tanh'  :'np.tanh',
           fi_inner:'f'}
    var = {'A' :'sol_mat[active,:,0]',
           'B1':'sol_mat[active,:,1]',
           'B2':'sol_mat[active,:,2]',
           'B3':'sol_mat[active,:,3]',
           'C1':'b_coef[active,:,1]',
           'C2':'b_coef[active,:,2]',
           'C3':'b_coef[active,:,3]',
           'C4':'b_coef[active,:,0]',
           'xc':'patch_1[active]',
           'yc':'patch_2[active]',
           'zc':'patch_3[active]',
           'xi':'patch_points[active,:,0]',
           'yi':'patch_points[active,:,1]',
           'zi':'patch_points[active,:,2]',
           'x' :'pt_3[:,:,0]',
           'y' :'pt_3[:,:,1]',
           'z' :'pt_3[:,:,2]',}
    funcs = {str(Fi):'f',
             str(Wi):'w',
             str(Ki):'k'}
    for i in range(len(del_orders_Fi)):
        for j in range(len(del_orders_Fi[i])):
            if order > 0:
                key[str(del_orders_Fi[i][j])] = notation_f[i+1][j]
                key[str(del_orders_Wi[i][j])] = notation_w[i+1][j]
                key[str(del_orders_Ki[i][j])] = notation_k[i+1][j]
            else:
                key[str(del_orders_Fi[i][j])] = notation_f[i][j]
                key[str(del_orders_Wi[i][j])] = notation_w[i][j]
                key[str(del_orders_Ki[i][j])] = notation_k[i][j]
    for i in range(len(del_orders_F)):
        for j in range(len(del_orders_F[i])):
            del_orders_fi[i][j] = str(del_orders_fi[i][j])
            del_orders_wi[i][j] = str(del_orders_wi[i][j])
            del_orders_F[i][j]  = str(del_orders_F[i][j])
            for item in key.keys():
                if item in del_orders_fi[i][j]:
                    #if key[item] == 'f' and i == 0:
                    #    pass
                    #else:
                    del_orders_fi[i][j] = del_orders_fi[i][j].replace(item,key[item])
                if item in del_orders_wi[i][j]:
                    del_orders_wi[i][j] = del_orders_wi[i][j].replace(item,key[item])
                if item in del_orders_F[i][j]:
                    del_orders_F[i][j] = del_orders_F[i][j].replace(item,key[item])
            for item in var.keys():
                if item in del_orders_fi[i][j]:
                    del_orders_fi[i][j] = del_orders_fi[i][j].replace(item,var[item])
                if item in del_orders_wi[i][j]:
                    del_orders_wi[i][j] = del_orders_wi[i][j].replace(item,var[item])
                if item in fs:
                    fs = fs.replace(item,var[item])
    for i in range(len(del_orders_F)):
        for j in range(len(del_orders_F[i])):
            for item in funcs.keys():
                if item in del_orders_F[i][j]:
                    del_orders_F[i][j] = del_orders_F[i][j].replace(item,funcs[item])
    for i in range(len(del_orders_fi)):
        if i == 0:
            continue
        for j in range(len(del_orders_fi[i])):
            pairs = outter_parentheses(del_orders_fi[i][j])
            old_fi = del_orders_fi[i][j]
            fixed_func = ''+old_fi[:pairs[0][0]]
            for ii,p in enumerate(pairs):
                if old_fi[p[1]-4:p[1]-2] == "**" and 'tanh' not in old_fi[p[0]:p[1]]:
                    p_square_start = old_fi.find("(",p[0]+1,p[1]-1)
                    p_square_end   = old_fi.rfind(")",p[0]+1,p[1]-1)
                    fixed_func += old_fi[p[0]:p_square_start]+'np.sum(np.nan_to_num'
                    fixed_func += old_fi[p_square_start:p_square_end+1]+',axis=1)'
                    fixed_func += old_fi[p_square_end+1:p[1]]
                    #fixed_func += 'np.sum(np.nan_to_num'+old_fi[p[0]:p[1]-4]+'),axis=1)'+old_fi[p[1]-4:p[1]]
                elif 'tanh' not in old_fi[p[0]:p[1]] and (p[1]-p[0]) > 3:
                    fixed_func += 'np.sum(np.nan_to_num'+old_fi[p[0]:p[1]]+',axis=1)'
                else:
                    fixed_func += old_fi[p[0]:p[1]]
                if ii < len(pairs)-1:
                    fixed_func += old_fi[p[1]:pairs[ii+1][0]]
                else:
                    if p[1] < len(old_fi):
                        fixed_func += old_fi[p[1]:]
            del_orders_fi[i][j] = fixed_func
    return del_orders_fi,del_orders_wi,del_orders_F,notation_f,notation_w,notation_k,fs

def construct(order,dim=3):
    funcs,weights,F,f_note,w_note,k_note,fs = d(order)
    functions = []
    func_strings = []
    for n in range(len(funcs)):
        d_func = 'def d_{}(xyzk,KDTree=None,'.format(n)+\
                           'patch_points=None,b_coef=None,'+\
                           'h=None,q=None,sol_mat=None,patch_1=None,'+\
                           'patch_2=None,patch_3=None,noskip=True):\n'
        d_func += '    x,y,z,k = xyzk\n'
        d_func += '    if noskip:\n'
        d_func += '        _,active = KDTree.query([x,y,z],k=k)\n'
        d_func += '    else:\n'
        d_func += '        active = list(range(len(patch_1)))\n'
        d_func += '    pt = np.ones((len(patch_1[active]),b_coef.shape[-1]))\n'
        d_func += '    pt[:,1] *= x\n'
        d_func += '    pt[:,2] *= y\n'
        d_func += '    pt[:,3] *= z\n'
        d_func += '    rc = ((x-patch_1[active])**2+(y-patch_2[active])**2+(z-patch_3[active])**2)**(1/2)\n'
        d_func += '    pt_3 = np.array([[[x,y,z]]])\n'
        d_func += '    if len(pt_3.shape) > 3:\n'
        d_func += '        pt_3 = pt_3.reshape(1,1,3)\n'
        #d_func += '    pt_3.flatten()
        d_func += '    rr = np.sum(np.square(pt_3-patch_points[active,:]),axis=2)\n'
        if n == 0:
            d_func += '    {} = {}\n'.format(f_note[0][0],fs)
            d_func += '    {} = np.nan_to_num({})\n'.format(f_note[0][0],f_note[0][0])
            d_func += '    {} = np.sum({},axis=1)\n'.format(f_note[0][0],f_note[0][0])
        elif n > 0:
            funcs[0][0] = fs
        for i in range(n+1):
            for j in range(len(weights[i])):
                d_func += '    {} = {}\n'.format(w_note[i][j],weights[i][j])
                d_func += '    {} = np.sum({})\n'.format(k_note[i][j],w_note[i][j])
        for i in range(n+1):
            for j in range(len(funcs[i])):
                tmp_func = funcs[i][j]
                d_func += '    {} = {}\n'.format(f_note[i][j],funcs[i][j])
                if i == 0 and n > 0:
                    d_func += '    {} = np.nan_to_num({})\n'.format(f_note[i][j],f_note[i][j])
                    d_func += '    {} = np.sum({},axis=1)\n'.format(f_note[i][j],f_note[i][j])
        d_func += '    res = np.zeros({})\n'.format(len(funcs[n]))
        if n > 0:
            d_func += '    {} = -np.tanh({})\n'.format(f_note[0][0],f_note[0][0])
        for j in range(len(funcs[n])):
            d_func += '    res_{} = {}\n'.format(f_note[n][j],F[n][j])
            d_func += '    res_{} = np.sum(res_{})\n'.format(f_note[n][j],f_note[n][j])
            d_func += '    res[{}] = res_{}\n'.format(j,f_note[n][j])
        if n > 1:
            d_func += '    res = res.reshape({}'.format(dim)
            for c in range(n-1):
                d_func += ',{}'.format(dim)
            d_func += ')\n'
        d_func += '    return res\n'
        #d_func += 'functions.append(d_{})'.format(n)
        exec(d_func)
        func_strings.append(d_func)
    return functions,func_strings

def gradient(xyzk,KDTree=None,patch_points=None,b_coef=None,h=None,q=None,sol_mat=None,patch_x=None,patch_y=None,patch_z=None):
    x,y,z,k = xyzk
    _,active = KDTree.query([x,y,z],k=k)
    pt = np.ones((len(patch_x[active]),len(b_coef[0])))
    pt[:,1] *= x
    pt[:,2] *= y
    pt[:,3] *= z
    r = ((x-patch_x[active])**2+(y-patch_y[active])**2+(z-patch_z[active])**2)**(1/2)
    weights = (1-np.tanh(r))/(r+h)**q
    dwx  = (-q*(1 - np.tanh(r))*(x-patch_x[active])*(h+r)**(-q-1))/r - ((1/np.cosh(r)**2)*(x-patch_x[active])*((h + r)**(-q)))/r
    dwy  = (-q*(1 - np.tanh(r))*(y-patch_y[active])*(h+r)**(-q-1))/r - ((1/np.cosh(r)**2)*(y-patch_y[active])*((h + r)**(-q)))/r
    dwz  = (-q*(1 - np.tanh(r))*(z-patch_z[active])*(h+r)**(-q-1))/r - ((1/np.cosh(r)**2)*(z-patch_z[active])*((h + r)**(-q)))/r
    pt_3 = np.array([[[x,y,z]]])
    scale = np.sqrt(np.sum(np.square(pt_3 - patch_points[active,:]),axis=2))
    fa = scale**3
    xcom = 3*scale*(pt_3[:,:,0] - patch_points[active,:,0])
    ycom = 3*scale*(pt_3[:,:,1] - patch_points[active,:,1])
    zcom = 3*scale*(pt_3[:,:,2] - patch_points[active,:,2])
    dx_1 = 3*scale*(pt_3[:,:,0] - patch_points[active,:,0])
    dx_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])**2)/scale + scale)
    dx_3 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,1] - patch_points[active,:,1]))/scale)
    dx_4 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dy_1 = 3*scale*(pt_3[:,:,1] - patch_points[active,:,1])
    dy_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,1] - patch_points[active,:,1]))/scale)
    dy_3 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])**2)/scale+scale)
    dy_4 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_1 = 3*scale*(pt_3[:,:,2] - patch_points[active,:,2])
    dz_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_3 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_4 = 3*(((pt_3[:,:,2] - patch_points[active,:,2])**2)/scale+scale)
    b = np.sum(pt*b_coef[active],axis=1)
    mat = np.zeros(sol_mat[active,:,:].shape)
    dx_mat = np.zeros(sol_mat[active,:,:].shape)
    dy_mat = np.zeros(sol_mat[active,:,:].shape)
    dz_mat = np.zeros(sol_mat[active,:,:].shape)
    mat[:,:,0] = sol_mat[active,:,0]*fa
    mat[:,:,1] = sol_mat[active,:,1]*xcom
    mat[:,:,2] = sol_mat[active,:,2]*ycom
    mat[:,:,3] = sol_mat[active,:,3]*zcom
    dx_mat[:,:,0] = sol_mat[active,:,0]*dx_1
    dx_mat[:,:,1] = sol_mat[active,:,1]*dx_2
    dx_mat[:,:,2] = sol_mat[active,:,2]*dx_3
    dx_mat[:,:,3] = sol_mat[active,:,3]*dx_4
    dy_mat[:,:,0] = sol_mat[active,:,0]*dy_1
    dy_mat[:,:,1] = sol_mat[active,:,1]*dy_2
    dy_mat[:,:,2] = sol_mat[active,:,2]*dy_3
    dy_mat[:,:,3] = sol_mat[active,:,3]*dy_4
    dz_mat[:,:,0] = sol_mat[active,:,0]*dz_1
    dz_mat[:,:,1] = sol_mat[active,:,1]*dz_2
    dz_mat[:,:,2] = sol_mat[active,:,2]*dz_3
    dz_mat[:,:,3] = sol_mat[active,:,3]*dz_4
    mat = np.nan_to_num(mat)
    dx_mat = np.nan_to_num(dx_mat)
    dy_mat = np.nan_to_num(dy_mat)
    dz_mat = np.nan_to_num(dz_mat)
    fsg = np.sum(np.sum(mat,axis=2),axis=1)+b
    total = np.sum(weights)
    dwx_total = np.sum(dwx)
    dwy_total = np.sum(dwy)
    dwz_total = np.sum(dwz)
    dx = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dx_mat,axis=2),axis=1)+b_coef[active,1])
    dy = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dy_mat,axis=2),axis=1)+b_coef[active,2])
    dz = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dz_mat,axis=2),axis=1)+b_coef[active,3])
    dx2 = ((-dwx*np.tanh(fsg)+dx*weights)*total+(dwx_total*weights*np.tanh(fsg)))/total**2
    dy2 = ((-dwy*np.tanh(fsg)+dy*weights)*total+(dwy_total*weights*np.tanh(fsg)))/total**2
    dz2 = ((-dwz*np.tanh(fsg)+dz*weights)*total+(dwz_total*weights*np.tanh(fsg)))/total**2
    #dx2 = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dx_mat,axis=2),axis=1)+b_coef[active,1])*(weights/total)#-(dwx/dwx_total)*np.tanh(fsg)
    #dy2 = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dy_mat,axis=2),axis=1)+b_coef[active,2])*(weights/total)#-(dwy/dwy_total)*np.tanh(fsg)
    #dz2 = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dz_mat,axis=2),axis=1)+b_coef[active,3])*(weights/total)#-(dwz/dwz_total)*np.tanh(fsg)
    #res_x = np.sum(weights*dx)/total
    #res_y = np.sum(weights*dy)/total
    #res_z = np.sum(weights*dz)/total
    res2_x = np.sum(dx2)
    res2_y = np.sum(dy2)
    res2_z = np.sum(dz2)
    return res2_x,res2_y,res2_z
def function_marching(x,y,z,patch_points=None,a_coef=None,b_coef=None,patch_x=None,patch_y=None,patch_z=None,q=None,h=None):
    interps = []
    weights = []
    total = 0
    for i in range(len(patch_x)):
        a = a_coef[i]
        b = b_coef[i]
        n = len(patch_points[i])
        xi = patch_points[i]
        fsg = b[0]+x*b[1]+y*b[2]+z*b[3]
        for j in range(n):
            scale = ((x-xi[j,0])**2+(y-xi[j,1])**2+(z-xi[j,2])**2)**(1/2)
            fsg += a[j]*(((x-xi[j,0])**2+(y-xi[j,1])**2+(z-xi[j,2])**2)**(1/2))**3
            fsg += a[n+j]*3*scale*(x-xi[j,0])
            fsg += a[n+j+n]*3*scale*(y-xi[j,1])
            fsg += a[n+j+2*n]*3*scale*(z-xi[j,2])
        fsg = -np.tanh(fsg)
        interps.append(fsg)
        r = ((x-patch_x[i])**2+(y-patch_y[i])**2+(z-patch_z[i])**2)**(1/2)
        value = (1-np.tanh(r))/(r+h)**q
        weights.append(value)
    total = sum(weights)
    res = 0
    for i in range(len(patch_x)):
        res += (weights[i]/total)*interps[i]
    return res

def function(xyzk,KDTree=None,patch_points=None,b_coef=None, h=None,q=None,sol_mat=None,patch_x=None,patch_y=None,patch_z=None):
    x,y,z,k = xyzk
    _,active = KDTree.query([x,y,z],k=k)
    pt = np.ones((len(patch_x[active]),len(b_coef[0])))
    pt[:,1] *= x
    pt[:,2] *= y
    pt[:,3] *= z
    r = ((x-patch_x[active])**2+(y-patch_y[active])**2+(z-patch_z[active])**2)**(1/2)
    weights = (1-np.tanh(r))/(r+h)**q
    pt_3 = np.array([[[x,y,z]]])
    scale = np.sqrt(np.sum(np.square(pt_3 - patch_points[active,:]),axis=2))
    fa = scale**3
    dx = 3*scale*(pt_3[:,:,0] - patch_points[active,:,0])
    dy = 3*scale*(pt_3[:,:,1] - patch_points[active,:,1])
    dz = 3*scale*(pt_3[:,:,2] - patch_points[active,:,2])
    b = np.sum(pt*b_coef[active],axis=1)
    mat = np.zeros(sol_mat[active,:,:].shape)
    mat[:,:,0] = sol_mat[active,:,0]*fa
    mat[:,:,1] = sol_mat[active,:,1]*dx
    mat[:,:,2] = sol_mat[active,:,2]*dy
    mat[:,:,3] = sol_mat[active,:,3]*dz
    mat = np.nan_to_num(mat)
    interps = -np.tanh(np.sum(np.sum(mat,axis=2),axis=1)+b)
    total = np.sum(weights)
    weights = weights/total
    res = np.sum(weights*interps)
    return res

def hessian(xyzk,KDTree=None,patch_points=None,b_coef=None,h=None,q=None,sol_mat=None,patch_x=None,patch_y=None,patch_z=None):
    x,y,z,k = xyzk
    _,active = KDTree.query([x,y,z],k=k)
    pt = np.ones((len(patch_x[active]),len(b_coef[0])))
    pt[:,1] *= x
    pt[:,2] *= y
    pt[:,3] *= z
    r = ((x-patch_x[active])**2+(y-patch_y[active])**2+(z-patch_z[active])**2)**(1/2)
    weights = (1-np.tanh(r))/(r+h)**q
    dwx  = (-q*(1 - np.tanh(r))*(x-patch_x[active])*(h+r)**(-q-1))/r - \
           ((1/np.cosh(r)**2)*(x-patch_x[active])*((h + r)**(-q)))/r
    dwy  = (-q*(1 - np.tanh(r))*(y-patch_y[active])*(h+r)**(-q-1))/r - \
           ((1/np.cosh(r)**2)*(y-patch_y[active])*((h + r)**(-q)))/r
    dwz  = (-q*(1 - np.tanh(r))*(z-patch_z[active])*(h+r)**(-q-1))/r - \
           ((1/np.cosh(r)**2)*(z-patch_z[active])*((h + r)**(-q)))/r
    dwxx = q**2*(1-np.tanh(r))*(-patch_x[active]+x)**2/((h+r)**2*(h+ r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_x[active] + x)**2/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_x[active] + x)*(patch_x[active]-x)/((h + r)*(h + r)**q*r**3) -\
           1.0*q*(1 -np.tanh(r))/((h + r)*(h + r)**q*r) + \
           2*q*(1 -np.tanh(r)**2)*(-patch_x[active] + x)**2/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_x[active] + x)**2*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_x[active] + x)*(patch_x[active]-x)/((h + r)**q*r**3) -\
           1.0*(1 -np.tanh(r)**2)/((h + r)**q*r)
    #dwxx_2 = -((-q - 1)*q*(x - patch_x[active])**2*(1 - np.tanh(r))*(r + h)**(-q - 2))/r**2 -\
    #          (q*(1 - np.tanh(r))*(r + h)**(-q - 1))/r +\
    #          (q*(x - patch_x[active])**2*(1 - np.tanh(r))*(r + h)**(-q - 1))/r**3 +\
    #          (2*q*(x - patch_x[active])**2*(1/np.cosh(r)**2)*(r + h)**(-q - 1))/r**2 -\
    #          ((1/np.cosh(r)**2)*(r + h)**(-q))/r + ((x - patch_x[active])**2*(1/np.cosh(r)**2)*(r + h)**(-q))/r**3 +\
    #          (2*(x - patch_x[active])**2*np.tanh(r)*(1/np.cosh(r)**2)*(r + h)**(-q))/r**2
    #print('dwxx: {}'.format(dwxx[1]))
    #print('dwxx_2: {}'.format(dwxx_2[1]))
    dwxy = q**2*(1 -np.tanh(r))*(-patch_x[active]+x)*(-patch_y[active]+y)/((h + r)**2*(h + r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_x[active]+x)*(-patch_y[active]+y)/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_x[active]+x)*(patch_y[active]-y)/((h + r)*(h + r)**q*r**3) +\
           2*q*(1 -np.tanh(r)**2)*(-patch_x[active]+x)*(-patch_y[active]+y)/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_x[active]+x)*(-patch_y[active]+y)*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_x[active]+x)*(patch_y[active] - y)/((h + r)**q*r**3)
    dwxz = q**2*(1 -np.tanh(r))*(-patch_x[active]+x)*(-patch_z[active]+z)/((h + r)**2*(h + r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_x[active]+x)*(-patch_z[active]+z)/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_x[active]+x)*(patch_z[active]-z)/((h + r)*(h + r)**q*r**3) +\
           2*q*(1 -np.tanh(r)**2)*(-patch_x[active]+x)*(-patch_z[active]+z)/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_x[active]+x)*(-patch_z[active]+z)*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_x[active]+x)*(patch_z[active]-z)/((h + r)**q*r**3)
    dwyx = dwxy
    dwyy = q**2*(1 -np.tanh(r))*(-patch_y[active] + y)**2/((h + r)**2*(h + r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_y[active] + y)**2/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_y[active] + y)*(patch_y[active] - y)/((h + r)*(h + r)**q*r**3) -\
           1.0*q*(1 -np.tanh(r))/((h + r)*(h + r)**q*r) +\
           2*q*(1 -np.tanh(r)**2)*(-patch_y[active] + y)**2/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_y[active] + y)**2*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_y[active] + y)*(patch_y[active] - y)/((h + r)**q*r**3) -\
           1.0*(1 -np.tanh(r)**2)/((h + r)**q*r)
    dwyz = q**2*(1 -np.tanh(r))*(-patch_y[active] + y)*(-patch_z[active] + z)/((h + r)**2*(h + r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_y[active] + y)*(-patch_z[active] + z)/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_y[active] + y)*(patch_z[active] - z)/((h + r)*(h + r)**q*r**3) +\
           2*q*(1 -np.tanh(r)**2)*(-patch_y[active] + y)*(-patch_z[active] + z)/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_y[active] + y)*(-patch_z[active] + z)*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_y[active] + y)*(patch_z[active] - z)/((h + r)**q*r**3)
    dwzx = dwxz
    dwzy = dwyz
    dwzz = q**2*(1 -np.tanh(r))*(-patch_z[active] + z)**2/((h + r)**2*(h + r)**q*r**2) +\
           q*(1 -np.tanh(r))*(-patch_z[active] + z)**2/((h + r)**2*(h + r)**q*r**2) -\
           q*(1 -np.tanh(r))*(-patch_z[active] + z)*(patch_z[active] - z)/((h + r)*(h + r)**q*r**3) -\
           1.0*q*(1 -np.tanh(r))/((h + r)*(h + r)**q*r) +\
           2*q*(1 -np.tanh(r)**2)*(-patch_z[active] + z)**2/((h + r)*(h + r)**q*r**2) +\
           2*(1 -np.tanh(r)**2)*(-patch_z[active] + z)**2*np.tanh(r)/((h + r)**q*r**2) -\
           (1 -np.tanh(r)**2)*(-patch_z[active] + z)*(patch_z[active] - z)/((h + r)**q*r**3) -\
           1.0*(1 -np.tanh(r)**2)/((h + r)**q*r)
    pt_3 = np.array([[[x,y,z]]])
    scale = np.sqrt(np.sum(np.square(pt_3 - patch_points[active,:]),axis=2))
    fa = scale**3
    xcom = 3*scale*(pt_3[:,:,0] - patch_points[active,:,0])
    ycom = 3*scale*(pt_3[:,:,1] - patch_points[active,:,1])
    zcom = 3*scale*(pt_3[:,:,2] - patch_points[active,:,2])
    dx_1 = 3*scale*(pt_3[:,:,0] - patch_points[active,:,0])
    dx_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])**2)/scale + scale)
    dx_3 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,1] - patch_points[active,:,1]))/scale)
    dx_4 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dy_1 = 3*scale*(pt_3[:,:,1] - patch_points[active,:,1])
    dy_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,1] - patch_points[active,:,1]))/scale)
    dy_3 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])**2)/scale+scale)
    dy_4 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_1 = 3*scale*(pt_3[:,:,2] - patch_points[active,:,2])
    dz_2 = 3*(((pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_3 = 3*(((pt_3[:,:,1] - patch_points[active,:,1])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale)
    dz_4 = 3*(((pt_3[:,:,2] - patch_points[active,:,2])**2)/scale+scale)
    b = np.sum(pt*b_coef[active],axis=1)
    mat = np.zeros(sol_mat[active,:,:].shape)
    dx_mat = np.zeros(sol_mat[active,:,:].shape)
    dy_mat = np.zeros(sol_mat[active,:,:].shape)
    dz_mat = np.zeros(sol_mat[active,:,:].shape)
    mat[:,:,0] = sol_mat[active,:,0]*fa
    mat[:,:,1] = sol_mat[active,:,1]*xcom
    mat[:,:,2] = sol_mat[active,:,2]*ycom
    mat[:,:,3] = sol_mat[active,:,3]*zcom
    dx_mat[:,:,0] = sol_mat[active,:,0]*dx_1
    dx_mat[:,:,1] = sol_mat[active,:,1]*dx_2
    dx_mat[:,:,2] = sol_mat[active,:,2]*dx_3
    dx_mat[:,:,3] = sol_mat[active,:,3]*dx_4
    dy_mat[:,:,0] = sol_mat[active,:,0]*dy_1
    dy_mat[:,:,1] = sol_mat[active,:,1]*dy_2
    dy_mat[:,:,2] = sol_mat[active,:,2]*dy_3
    dy_mat[:,:,3] = sol_mat[active,:,3]*dy_4
    dz_mat[:,:,0] = sol_mat[active,:,0]*dz_1
    dz_mat[:,:,1] = sol_mat[active,:,1]*dz_2
    dz_mat[:,:,2] = sol_mat[active,:,2]*dz_3
    dz_mat[:,:,3] = sol_mat[active,:,3]*dz_4
    mat = np.nan_to_num(mat)
    dx_mat = np.nan_to_num(dx_mat)
    dy_mat = np.nan_to_num(dy_mat)
    dz_mat = np.nan_to_num(dz_mat)
    #dxx
    dxx_1 = (pt_3[:,:,0] - patch_points[active,:,0])*(3*(pt_3[:,:,0] - patch_points[active,:,0])*(patch_points[active,:,0]-pt_3[:,:,0]))/scale**3 #B1
    dxx_2 = (pt_3[:,:,1] - patch_points[active,:,1])*(3*(pt_3[:,:,0] - patch_points[active,:,0])*(patch_points[active,:,0]-pt_3[:,:,0]))/scale**3 #B2
    dxx_3 = (pt_3[:,:,2] - patch_points[active,:,2])*(3*(pt_3[:,:,0] - patch_points[active,:,0])*(patch_points[active,:,0]-pt_3[:,:,0]))/scale**3 #B3
    dxx_4 = 6*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B1
    dxx_5 = (3*(pt_3[:,:,0] - patch_points[active,:,0])**2)/scale #A
    dxx_6 = 3*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B1
    dxx_7 = 3*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B2
    dxx_8 = 3*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B3
    dxx_9 = 3*scale #A
    #dxy
    dxy_1 = (pt_3[:,:,0]-patch_points[active,:,0])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B1
    dxy_2 = (pt_3[:,:,1]-patch_points[active,:,1])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B2
    dxy_3 = (pt_3[:,:,2]-patch_points[active,:,2])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B3
    dxy_4 = 3*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B2
    dxy_5 = 3*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B1
    dxy_6 = (3*(pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,1] - patch_points[active,:,1]))/scale #A
    # dxz
    dxz_1 = (pt_3[:,:,0]-patch_points[active,:,0])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B1
    dxz_2 = (pt_3[:,:,1]-patch_points[active,:,1])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B2
    dxz_3 = (pt_3[:,:,2]-patch_points[active,:,2])*(3*(pt_3[:,:,0]-patch_points[active,:,0])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B3
    dxz_4 = 3*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B3
    dxz_5 = 3*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B1
    dxz_6 = (3*(pt_3[:,:,0] - patch_points[active,:,0])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale #A
    #dyy
    dyy_1 = (pt_3[:,:,0] - patch_points[active,:,0])*(3*(pt_3[:,:,1] - patch_points[active,:,1])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B1
    dyy_2 = (pt_3[:,:,1] - patch_points[active,:,1])*(3*(pt_3[:,:,1] - patch_points[active,:,1])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B2
    dyy_3 = (pt_3[:,:,2] - patch_points[active,:,2])*(3*(pt_3[:,:,1] - patch_points[active,:,1])*(patch_points[active,:,1] - pt_3[:,:,1]))/scale**3 #B3
    dyy_4 = 6*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B2
    dyy_5 = (3*(pt_3[:,:,1] - patch_points[active,:,1])**2)/scale #A
    dyy_6 = 3*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B1
    dyy_7 = 3*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B2
    dyy_8 = 3*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B3
    dyy_9 = 3*scale #A
    #dyz
    dyz_1 = (pt_3[:,:,0]-patch_points[active,:,0])*(3*(pt_3[:,:,1]-patch_points[active,:,1])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B1
    dyz_2 = (pt_3[:,:,1]-patch_points[active,:,1])*(3*(pt_3[:,:,1]-patch_points[active,:,1])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B2
    dyz_3 = (pt_3[:,:,2]-patch_points[active,:,2])*(3*(pt_3[:,:,1]-patch_points[active,:,1])*(patch_points[active,:,2]-pt_3[:,:,2]))/scale**3 #B3
    dyz_4 = 3*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B3
    dyz_5 = 3*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B2
    dyz_6 = (3*(pt_3[:,:,1] - patch_points[active,:,1])*(pt_3[:,:,2] - patch_points[active,:,2]))/scale #A
    #dzz
    dzz_1 = (pt_3[:,:,0] - patch_points[active,:,0])*(3*(pt_3[:,:,2] - patch_points[active,:,2])*(patch_points[active,:,2] - pt_3[:,:,2]))/scale**3 #B1
    dzz_2 = (pt_3[:,:,1] - patch_points[active,:,1])*(3*(pt_3[:,:,2] - patch_points[active,:,2])*(patch_points[active,:,2] - pt_3[:,:,2]))/scale**3 #B2
    dzz_3 = (pt_3[:,:,2] - patch_points[active,:,2])*(3*(pt_3[:,:,2] - patch_points[active,:,2])*(patch_points[active,:,2] - pt_3[:,:,2]))/scale**3 #B3
    dzz_4 = 6*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B3
    dzz_5 = (3*(pt_3[:,:,2] - patch_points[active,:,2])**2)/scale #A
    dzz_6 = 3*(pt_3[:,:,0] - patch_points[active,:,0])/scale #B1
    dzz_7 = 3*(pt_3[:,:,1] - patch_points[active,:,1])/scale #B2
    dzz_8 = 3*(pt_3[:,:,2] - patch_points[active,:,2])/scale #B3
    dzz_9 = 3*scale #A
    dxx_mat = np.zeros(sol_mat[active,:,:].shape)
    dxy_mat = np.zeros(sol_mat[active,:,:].shape)
    dxz_mat = np.zeros(sol_mat[active,:,:].shape)
    dyy_mat = np.zeros(sol_mat[active,:,:].shape)
    dyz_mat = np.zeros(sol_mat[active,:,:].shape)
    dzz_mat = np.zeros(sol_mat[active,:,:].shape)
    #multiply solution coefficients to variable values
    dxx_mat[:,:,0] = sol_mat[active,:,0]*(dxx_5+dxx_9)
    dxx_mat[:,:,1] = sol_mat[active,:,1]*(dxx_1+dxx_4+dxx_6)
    dxx_mat[:,:,2] = sol_mat[active,:,2]*(dxx_2+dxx_7)
    dxx_mat[:,:,3] = sol_mat[active,:,3]*(dxx_3+dxx_8)

    dxy_mat[:,:,0] = sol_mat[active,:,0]*(dxy_6)
    dxy_mat[:,:,1] = sol_mat[active,:,1]*(dxy_1+dxy_5)
    dxy_mat[:,:,2] = sol_mat[active,:,2]*(dxy_2+dxy_4)
    dxy_mat[:,:,3] = sol_mat[active,:,3]*(dxy_3)

    dxz_mat[:,:,0] = sol_mat[active,:,0]*(dxz_6)
    dxz_mat[:,:,1] = sol_mat[active,:,1]*(dxz_1+dxz_5)
    dxz_mat[:,:,2] = sol_mat[active,:,2]*(dxz_2)
    dxz_mat[:,:,3] = sol_mat[active,:,3]*(dxz_3+dxz_4)

    dyy_mat[:,:,0] = sol_mat[active,:,0]*(dyy_5+dyy_9)
    dyy_mat[:,:,1] = sol_mat[active,:,1]*(dyy_1+dyy_6)
    dyy_mat[:,:,2] = sol_mat[active,:,2]*(dyy_2+dyy_4+dyy_7)
    dyy_mat[:,:,3] = sol_mat[active,:,3]*(dyy_3+dyy_8)

    dyz_mat[:,:,0] = sol_mat[active,:,0]*(dyz_6)
    dyz_mat[:,:,1] = sol_mat[active,:,1]*(dyz_1)
    dyz_mat[:,:,2] = sol_mat[active,:,2]*(dyz_2+dyz_5)
    dyz_mat[:,:,3] = sol_mat[active,:,3]*(dyz_3+dyz_4)

    dzz_mat[:,:,0] = sol_mat[active,:,0]*(dzz_5+dzz_9)
    dzz_mat[:,:,1] = sol_mat[active,:,1]*(dzz_1+dzz_6)
    dzz_mat[:,:,2] = sol_mat[active,:,2]*(dzz_2+dzz_7)
    dzz_mat[:,:,3] = sol_mat[active,:,3]*(dzz_3+dzz_4+dzz_8)
    dxx_mat = np.nan_to_num(dxx_mat)
    dxy_mat = np.nan_to_num(dxy_mat)
    dxz_mat = np.nan_to_num(dxz_mat)
    dyy_mat = np.nan_to_num(dyy_mat)
    dyz_mat = np.nan_to_num(dyz_mat)
    dzz_mat = np.nan_to_num(dzz_mat)
    #combine inner components to arrive at gradient, function, and partial
    #hessian values
    dxx_mat = np.sum(np.sum(dxx_mat,axis=2),axis=1)
    dxy_mat = np.sum(np.sum(dxy_mat,axis=2),axis=1)
    dxz_mat = np.sum(np.sum(dxz_mat,axis=2),axis=1)
    dyy_mat = np.sum(np.sum(dyy_mat,axis=2),axis=1)
    dyz_mat = np.sum(np.sum(dyz_mat,axis=2),axis=1)
    dzz_mat = np.sum(np.sum(dzz_mat,axis=2),axis=1)

    fsg = np.sum(np.sum(mat,axis=2),axis=1)+b
    dx2 = (np.sum(np.sum(dx_mat,axis=2),axis=1)+b_coef[active,1])
    dy2 = (np.sum(np.sum(dy_mat,axis=2),axis=1)+b_coef[active,2])
    dz2 = (np.sum(np.sum(dz_mat,axis=2),axis=1)+b_coef[active,3])
    dx = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dx_mat,axis=2),axis=1)+b_coef[active,1])
    dy = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dy_mat,axis=2),axis=1)+b_coef[active,2])
    dz = -(1/np.cosh(fsg)**2)*(np.sum(np.sum(dz_mat,axis=2),axis=1)+b_coef[active,3])
    dwx_total = np.sum(dwx)
    dwy_total = np.sum(dwy)
    dwz_total = np.sum(dwz)
    dwxx_total = np.sum(dwxx)
    dwxy_total = np.sum(dwxy)
    dwxz_total = np.sum(dwxz)
    dwyx_total = np.sum(dwyx)
    dwyy_total = np.sum(dwyy)
    dwyz_total = np.sum(dwyz)
    dwzx_total = np.sum(dwzx)
    dwzy_total = np.sum(dwzy)
    dwzz_total = np.sum(dwzz)
    total = np.sum(weights)
    dx22 = ((-dwx*np.tanh(fsg)+dx*weights)*total+(dwx_total*weights*np.tanh(fsg)))/total**2
    dy22 = ((-dwy*np.tanh(fsg)+dy*weights)*total+(dwy_total*weights*np.tanh(fsg)))/total**2
    dz22 = ((-dwz*np.tanh(fsg)+dz*weights)*total+(dwz_total*weights*np.tanh(fsg)))/total**2
    #print('dx: {}'.format(np.sum(dx22)))
    #print('dy: {}'.format(np.sum(dy22)))
    #print('dz: {}'.format(np.sum(dz22)))
    #dx2 = ((-dwx*np.tanh(fsg)+dx*weights)*total+(dwx_total*weights*np.tanh(fsg)))/total**2
    #dy2 = ((-dwy*np.tanh(fsg)+dy*weights)*total+(dwy_total*weights*np.tanh(fsg)))/total**2
    #dz2 = ((-dwz*np.tanh(fsg)+dz*weights)*total+(dwz_total*weights*np.tanh(fsg)))/total**2

    dxx = -dxx_mat*(1/np.cosh(fsg))**2+2*dx2**2*(1/np.cosh(fsg))**2*np.tanh(fsg)
    dxy = -dxy_mat*(1/np.cosh(fsg))**2+2*dx2*dy2*(1/np.cosh(fsg))**2*np.tanh(fsg)
    dxz = -dxz_mat*(1/np.cosh(fsg))**2+2*dx2*dz2*(1/np.cosh(fsg))**2*np.tanh(fsg)
    dyy = -dyy_mat*(1/np.cosh(fsg))**2+2*dy2**2*(1/np.cosh(fsg))**2*np.tanh(fsg)
    dyz = -dyz_mat*(1/np.cosh(fsg))**2+2*dy2*dz2*(1/np.cosh(fsg))**2*np.tanh(fsg)
    dzz = -dzz_mat*(1/np.cosh(fsg))**2+2*dz2**2*(1/np.cosh(fsg))**2*np.tanh(fsg)

    #res_dxx = (((-dwxx*np.tanh(fsg)+dx*dwx+dxx*weights+dx*dwx)*total+\
    #           dwx_total*(-dwx*np.tanh(fsg)+dx*weights)-\
    #           (dwxx_total*(-weights*np.tanh(fsg))+\
    #            dwx_total*(-dwx*np.tanh(fsg)+dx*weights)))*total**2 -\
    #           2*dwx_total*((-dwx*np.tanh(fsg)+dx*weights)*total - \
    #                         dwx_total*(-weights*np.tanh(fsg))))/total**4
    res_dxx = (-total*(weights*(2*dx*dwx_total + -np.tanh(fsg)*dwxx_total) +
               2*-np.tanh(fsg)*dwx_total*dwx) + total**2*(weights*dxx + 2*dx*dwx +
               -np.tanh(fsg)*dwxx) + 2*-np.tanh(fsg)*weights*dwx_total**2)/total**3
    #res_dxy = (((-dwxy*np.tanh(fsg)+dwx*dy+dxy*weights+dwy*dx)*total+\
    #           dwy_total*(-dwx*np.tanh(fsg)+dx*weights)-\
    #           (dwxy_total*(-weights*np.tanh(fsg))+\
    #            dwx_total*(-dwy*np.tanh(fsg)+dy*weights)))*total**2 -\
    #            2*dwy_total*((-dwx*np.tanh(fsg)+dx*weights)*total - \
    #                          dwx_total*(-weights*np.tanh(fsg))))/total**4
    res_dxy = (-total*(weights*(dx*dwy_total + dy*dwx*total + -np.tanh(fsg)*dwxy_total) +
               -np.tanh(fsg)*(dwx_total*dwy + dwy_total*dwy)) + total**2*(dx*dwy + dy*dwy +
               dxy*weights + -np.tanh(fsg)*dwxy) +
               2*-np.tanh(fsg)*dwy_total*dwx_total*weights)/total**3

    #res_dxz = (((-dwxz*np.tanh(fsg)+dwx*dz+dxz*weights+dwz*dx)*total+\
    #           dwz_total*(-dwx*np.tanh(fsg)+dx*weights)-\
    #           (dwxz_total*(-weights*np.tanh(fsg))+\
    #            dwx_total*(-dwz*np.tanh(fsg)+dz*weights)))*total**2 -\
    #            2*dwz_total*((-dwx*np.tanh(fsg)+dx*weights)*total - \
    #                          dwx_total*(-weights*np.tanh(fsg))))/total**4
    res_dxz = (-total*(weights*(dx*dwz_total + dz*dwx*total + -np.tanh(fsg)*dwxz_total) +
               -np.tanh(fsg)*(dwx_total*dwz + dwz_total*dwz)) + total**2*(dx*dwz + dz*dwz +
               dxz*weights + -np.tanh(fsg)*dwxz) +
               2*-np.tanh(fsg)*dwz_total*dwx_total*weights)/total**3
    #res_dyy = (((-dwyy*np.tanh(fsg)+dy*dwy+dyy*weights+dy*dwy)*total+\
    #           dwy_total*(-dwy*np.tanh(fsg)+dy*weights)-\
    #           (dwyy_total*(-weights*np.tanh(fsg))+\
    #            dwy_total*(-dwy*np.tanh(fsg)+dy*weights)))*total**2 -\
    #           2*dwy_total*((-dwy*np.tanh(fsg)+dy*weights)*total - \
    #                         dwy_total*(-weights*np.tanh(fsg))))/total**4
    res_dyy = (-total*(weights*(2*dy*dwy_total-np.tanh(fsg)*dwyy_total)-
               2*np.tanh(fsg)*dwy_total*dwy)+total**2*(weights*dyy+2*dy*dwy-
               np.tanh(fsg)*dwyy)-2*np.tanh(fsg)*weights*dwy_total**2)/total**3
    #res_dyz = (((-dwyz*np.tanh(fsg)+dwy*dz+dyz*weights+dwz*dy)*total+\
    #           dwz_total*(-dwy*np.tanh(fsg)+dy*weights)-\
    #           (dwyz_total*(-weights*np.tanh(fsg))+\
    #            dwy_total*(-dwz*np.tanh(fsg)+dz*weights)))*total**2 -\
    #            2*dwz_total*((-dwy*np.tanh(fsg)+dy*weights)*total - \
    #                          dwy_total*(-weights*np.tanh(fsg))))/total**4
    res_dyz = (-total*(weights*(dy*dwz_total + dz*dwy*total + -np.tanh(fsg)*dwyz_total) +
               -np.tanh(fsg)*(dwy_total*dwz + dwz_total*dwz)) + total**2*(dy*dwz + dz*dwz +
               dxz*weights + -np.tanh(fsg)*dwyz) +
               2*-np.tanh(fsg)*dwz_total*dwy_total*weights)/total**3
    #res_dzz = (((-dwzz*np.tanh(fsg)+dz*dwz+dzz*weights+dz*dwz)*total+\
    #            dwz_total*(-dwz*np.tanh(fsg)+dz*weights)-\
    #            (dwzz_total*(-weights*np.tanh(fsg))+\
    #             dwz_total*(-dwz*np.tanh(fsg)+dz*weights)))*total**2 -\
    #             2*dwz_total*((-dwz*np.tanh(fsg)+dz*weights)*total - \
    #                           dwz_total*(-weights*np.tanh(fsg))))/total**4
    res_dzz = (-total*(weights*(2*dz*dwz_total-np.tanh(fsg)*dwzz_total)-
               2*np.tanh(fsg)*dwz_total*dwz)+total**2*(weights*dzz+2*dz*dwz-
               np.tanh(fsg)*dwzz)-2*np.tanh(fsg)*weights*dwz_total**2)/total**3
    #total = np.sum(weights)
    #res_dxx = np.sum(weights*dxx)/total
    #res_dxy = np.sum(weights*dxy)/total
    #res_dxz = np.sum(weights*dxz)/total
    #res_dyy = np.sum(weights*dyy)/total
    #res_dyz = np.sum(weights*dyz)/total
    #res_dzz = np.sum(weights*dzz)/total
    res_dxx = np.sum(res_dxx)
    res_dxy = np.sum(res_dxy)
    res_dxz = np.sum(res_dxz)
    res_dyy = np.sum(res_dyy)
    res_dyz = np.sum(res_dyz)
    res_dzz = np.sum(res_dzz)
    return np.array([[res_dxx,res_dxy,res_dxz],[res_dxy,res_dyy,res_dyz],[res_dxz,res_dyz,res_dzz]])

def core_hessian(x,y,z,patch_points=None,a_coef=None,b_coef=None,patch_x=None,patch_y=None,patch_z=None,q=None,h=None):
    fsg_dxx = []
    fsg_dxy = []
    fsg_dxz = []
    fsg_dyy = []
    fsg_dyz = []
    fsg_dzz = []
    weights = []
    for i in range(len(patch_x)):
        a = a_coef[i]
        b = b_coef[i]
        n = len(patch_points[i])
        xi = patch_points[i]
        #patch = patches[i]
        dx = b[1]
        dy = b[2]
        dz = b[3]
        fsg = b[0]+x*b[1]+y*b[2]+z*b[3]
        dxx = 0
        dxy = 0
        dxz = 0
        dyy = 0
        dyz = 0
        dzz = 0
        for j in range(n):
            scale = ((x-xi[j,0])**2+(y-xi[j,1])**2+(z-xi[j,2])**2)**(1/2)
            #f_value = np.sech(a[j]*scale**3+3*scale*(a[n+j]*(x-xi[j,0])+(y-xi[j,1])+(z-xi[j,2]))+x*b[1]+y*b[2]+z*b[3]+b[0])**2
            # function value
            fsg += a[j]*(((x-xi[j,0])**2+(y-xi[j,1])**2+(z-xi[j,2])**2)**(1/2))**3
            fsg += a[n+j]*3*scale*(x-xi[j,0])
            fsg += a[n+j+n]*3*scale*(y-xi[j,1])
            fsg += a[n+j+2*n]*3*scale*(z-xi[j,2])
            # derivative values
            dx += a[j]*3*scale*(x-xi[j,0])
            dy += a[j]*3*scale*(y-xi[j,1])
            dz += a[j]*3*scale*(z-xi[j,2])
            # hessian x-components
            dxx += (-((3*(x-xi[j,0])**2*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(6*a[n+j]*(x-xi[j,0]))/scale+(3*a[j]*(x-xi[j,0])**2)/scale+
                    (3*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/scale+
                    3*a[j]*scale)
            dxy += (-((3*(x-xi[j,0])*(y-xi[j,1])*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(3*a[n+j+n]*(x-xi[j,0]))/scale+(3*a[n+j]*(y-xi[j,1]))/scale+
                    (3*(a[j]*(x-xi[j,0])*(y-xi[j,1])))/scale)
            dxz += (-((3*(x-xi[j,0])*(z-xi[j,2])*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(3*a[n+j+2*n]*(x-xi[j,0]))/scale+(3*a[n+j]*(z-xi[j,2]))/scale+
                    (3*(a[j]*(x-xi[j,0])*(z-xi[j,2])))/scale)
            dyy += (-((3*(y-xi[j,1])**2*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(6*a[n+j+n]*(y-xi[j,1]))/scale+(3*a[j]*(y-xi[j,1])**2)/scale+
                    (3*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/scale+
                    3*a[j]*scale)
            dyz += (-((3*(y-xi[j,1])*(z-xi[j,2])*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(3*a[n+j+2*n]*(y-xi[j,1]))/scale+(3*a[n+j+n]*(z-xi[j,2]))/scale+
                    (3*(a[j]*(y-xi[j,1])*(z-xi[j,2])))/scale)
            dzz += (-((3*(z-xi[j,2])**2*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/
                      (scale**3))+(6*a[n+j+2*n]*(z-xi[j,2]))/scale+(3*a[j]*(z-xi[j,2])**2)/scale+
                    (3*(a[n+j]*(x-xi[j,0])+a[n+j+n]*(y-xi[j,1])+a[n+j+2*n]*(z-xi[j,2])))/scale+
                    3*a[j]*scale)
            #x-derivatives components numerator
            dx += a[n+j]*3*((((x-xi[j,0])**2)/scale)+scale)
            dx += a[n+j+n]*3*(((x-xi[j,0])*(y-xi[j,1]))/scale)
            dx += a[n+j+2*n]*3*(((x-xi[j,0])*(z-xi[j,2]))/scale)
            #y-derativate components numerator
            dy += a[n+j]*3*(((x-xi[j,0])*(y-xi[j,1]))/scale)
            dy += a[n+j+n]*3*((((y-xi[j,1])**2)/scale)+scale)
            dy += a[n+j+2*n]*3*(((y-xi[j,1])*(z-xi[j,2]))/scale)
            #z-derivative components numerator
            dz += a[n+j]*3*(((x-xi[j,0])*(z-xi[j,2]))/scale)
            dz += a[n+j+n]*3*(((z-xi[j,2])*(y-xi[j,1]))/scale)
            dz += a[n+j+2*n]*3*((((z-xi[j,2])**2)/scale)+scale)
            #x-derivative components denominator
            #fsg += a[n+j]*3*scale*(x-xi[j,0])
            #fsg += a[n+j+n]*3*scale*(y-xi[j,1])
            #fsg += a[n+j+2*n]*3*scale*(z-xi[j,2])
        #dx = -(1/np.cosh(fsg))**2*dx
        #dy = -(1/np.cosh(fsg))**2*dy
        #dz = -(1/np.cosh(fsg))**2*dz
        dxx = -dxx*(1/np.cosh(fsg))**2+2*dx**2*(1/np.cosh(fsg))**2*np.tanh(fsg)
        dxy = -dxy*(1/np.cosh(fsg))**2+2*dx*dy*(1/np.cosh(fsg))**2*np.tanh(fsg)
        dxz = -dxz*(1/np.cosh(fsg))**2+2*dx*dz*(1/np.cosh(fsg))**2*np.tanh(fsg)
        dyy = -dyy*(1/np.cosh(fsg))**2+2*dy**2*(1/np.cosh(fsg))**2*np.tanh(fsg)
        dyz = -dyz*(1/np.cosh(fsg))**2+2*dy*dz*(1/np.cosh(fsg))**2*np.tanh(fsg)
        dzz = -dzz*(1/np.cosh(fsg))**2+2*dz**2*(1/np.cosh(fsg))**2*np.tanh(fsg)
        fsg_dxx.append(dxx)
        fsg_dxy.append(dxy)
        fsg_dxz.append(dxz)
        fsg_dyy.append(dyy)
        fsg_dyz.append(dyz)
        fsg_dzz.append(dzz)
        #p0 = patch[0,:]
        #p1 = patch[-1,:]
        #d = ((p1[0]-p0[0])**2+(p1[1]-p0[1])**2+(p1[2]-p0[2])**2)**(1/2)
        r = ((x-patch_x[i])**2+(y-patch_y[i])**2+(z-patch_z[i])**2)**(1/2)
        #k = r/d
        #direc = np.array([x-p0[0],y-p0[1],z-p0[2]])/r
        value = (1-np.tanh(r))/(r+h)**q
        weights.append(value)
    weights = np.array(weights)
    total = np.sum(weights)
    res_xx = 0
    res_xy = 0
    res_xz = 0
    res_yy = 0
    res_yz = 0
    res_zz = 0
    for i in range(len(patch_x)):
        res_xx += (weights[i]/total)*fsg_dxx[i]
        res_xy += (weights[i]/total)*fsg_dxy[i]
        res_xz += (weights[i]/total)*fsg_dxz[i]
        res_yy += (weights[i]/total)*fsg_dyy[i]
        res_yz += (weights[i]/total)*fsg_dyz[i]
        res_zz += (weights[i]/total)*fsg_dzz[i]
    return np.array([[res_xx,res_xy,res_xz],[res_xy,res_yy,res_yz],[res_xz,res_yz,res_zz]])

def linesearch_d0(t,point=None,normal=None,patch_points=None,
                  b_coef=None,h=None,q=None,sol_mat=None,
                  patch_1=None,patch_2=None,patch_3=None):
    x  = point[0]
    y  = point[1]
    z  = point[2]
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
    xi = patch_points[:,:,0]
    yi = patch_points[:,:,1]
    zi = patch_points[:,:,2]
    xc = patch_1
    yc = patch_2
    zc = patch_3
    dx = (nx*t + x - xi)
    dy = (ny*t + y - yi)
    dz = (nz*t + z - zi)
    dxc = (nx*t + x - xc)
    dyc = (ny*t + y - yc)
    dzc = (nz*t + z - zc)
    r  = (dx**2 + dy**2 + dz**2)
    rc = (dxc**2 + dyc**2 + dzc**2)
    A  = sol_mat[:,:,0]
    B1 = sol_mat[:,:,1]
    B2 = sol_mat[:,:,2]
    B3 = sol_mat[:,:,3]
    C4 = b_coef[:,:,0]
    C1 = b_coef[:,:,1]
    C2 = b_coef[:,:,2]
    C3 = b_coef[:,:,3]
    finner = A*r**1.5 +\
             B1*(3.0*nx*t + 3.0*x - 3.0*xi)*r**0.5 +\
             B2*(3.0*ny*t + 3.0*y - 3.0*yi)*r**0.5 +\
             B3*(3.0*nz*t + 3.0*z - 3.0*zi)*r**0.5 +\
             C1*(nx*t + x) + C2*(ny*t + y) + C3*(nz*t + z) + C4
    finner = np.sum(finner,axis=1)
    f = -np.tanh(finner)
    w = (1 - np.tanh(rc**0.5))/(h + rc**0.5)**q
    f = (f*w)/np.sum(w)
    return np.sum(f)

def linesearch_d1(t,point=None,normal=None,patch_points=None,
                  b_coef=None,h=None,q=None,sol_mat=None,
                  patch_1=None,patch_2=None,patch_3=None):
    x  = point[0]
    y  = point[1]
    z  = point[2]
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
    xi = patch_points[:,:,0]
    yi = patch_points[:,:,1]
    zi = patch_points[:,:,2]
    xc = patch_1
    yc = patch_2
    zc = patch_3
    dx = (nx*t + x - xi)
    dy = (ny*t + y - yi)
    dz = (nz*t + z - zi)
    dxc = (nx*t + x - xc)
    dyc = (ny*t + y - yc)
    dzc = (nz*t + z - zc)
    r  = (dx**2 + dy**2 + dz**2)
    rc = (dxc**2 + dyc**2 + dzc**2)
    A  = sol_mat[:,:,0]
    B1 = sol_mat[:,:,1]
    B2 = sol_mat[:,:,2]
    B3 = sol_mat[:,:,3]
    C4 = b_coef[:,:,0]
    C1 = b_coef[:,:,1]
    C2 = b_coef[:,:,2]
    C3 = b_coef[:,:,3]
    finner = A*r**1.5 +\
             B1*(3.0*nx*t + 3.0*x - 3.0*xi)*r**0.5 +\
             B2*(3.0*ny*t + 3.0*y - 3.0*yi)*r**0.5 +\
             B3*(3.0*nz*t + 3.0*z - 3.0*zi)*r**0.5 +\
             C1*(nx*t + x) + C2*(ny*t + y) + C3*(nz*t + z) + C4
    finner = np.sum(finner,axis=1)
    f = -np.tanh(finner)
    w = (1 - np.tanh(rc**0.5))/(h + rc**0.5)**q

    wp = -q*(1 - np.tanh(rc**0.5))*(1.0*nx*dxc + 1.0*ny*dyc + 1.0*nz*dzc)/((h + rc**0.5)*(h + rc**0.5)**q*rc**0.5) -\
         (1 - np.tanh(rc**0.5)**2)*(1.0*nx*dxc + 1.0*ny*dyc + 1.0*nz*dzc)/((h + rc**0.5)**q*rc**0.5)

    fp = -(1 - np.tanh(finner)**2)*\
         np.sum(np.nan_to_num(A*(3.0*nx*dx + 3.0*ny*dy + 3.0*nz*dz)*r**0.5 +
                3.0*B1*nx*r**0.5 +
                B1*(3.0*nx*t + 3.0*x - 3.0*xi)*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)/r**0.5 +
                3.0*B2*ny*r**0.5 +
                B2*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)*(3.0*ny*t + 3.0*y - 3.0*yi)/r**0.5 +
                3.0*B3*nz*r**0.5 +
                B3*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)*(3.0*nz*t + 3.0*z - 3.0*zi)/r**0.5 +
                C1*nx + C2*ny + C3*nz),axis=1)

    DFT = f*wp/np.sum(w) - f*w*np.sum(wp)/np.sum(w)**2 + w*fp/np.sum(w)
    DFT = np.sum(DFT)
    return DFT

def linesearch_d2(t,point=None,normal=None,patch_points=None,b_coef=None,h=None,q=None,sol_mat=None,patch_1=None,patch_2=None,patch_3=None):
    x  = point[0]
    y  = point[1]
    z  = point[2]
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]
    xi = patch_points[:,:,0]
    yi = patch_points[:,:,1]
    zi = patch_points[:,:,2]
    xc = patch_1
    yc = patch_2
    zc = patch_3
    dx = (nx*t + x - xi)
    dy = (ny*t + y - yi)
    dz = (nz*t + z - zi)
    dxc = (nx*t + x - xc)
    dyc = (ny*t + y - yc)
    dzc = (nz*t + z - zc)
    r  = (dx**2 + dy**2 + dz**2)
    rc = (dxc**2 + dyc**2 + dzc**2)
    A  = sol_mat[:,:,0]
    B1 = sol_mat[:,:,1]
    B2 = sol_mat[:,:,2]
    B3 = sol_mat[:,:,3]
    C4 = b_coef[:,:,0]
    C1 = b_coef[:,:,1]
    C2 = b_coef[:,:,2]
    C3 = b_coef[:,:,3]
    finner = A*r**1.5 +\
             B1*(3.0*nx*t + 3.0*x - 3.0*xi)*r**0.5 +\
             B2*(3.0*ny*t + 3.0*y - 3.0*yi)*r**0.5 +\
             B3*(3.0*nz*t + 3.0*z - 3.0*zi)*r**0.5 +\
             C1*(nx*t + x) + C2*(ny*t + y) + C3*(nz*t + z) + C4
    finner = np.sum(np.nan_to_num(finner),axis=1)
    f = -np.tanh(finner)

    w = (1 - np.tanh(rc**0.5))/(h + rc**0.5)**q

    wp = -q*(1 - np.tanh(rc**0.5))*(1.0*nx*dxc + 1.0*ny*dyc + 1.0*nz*dzc)/((h + rc**0.5)*(h + rc**0.5)**q*rc**0.5) -\
         (1 - np.tanh(rc**0.5)**2)*(1.0*nx*dxc + 1.0*ny*dyc + 1.0*nz*dzc)/((h + rc**0.5)**q*rc**0.5)

    wpp = -(1.0*q*(np.tanh(rc**0.5) - 1)*(q*(nx*dxc + ny*dyc + nz*dzc)**2/((h + rc**0.5)*rc**1.0) -
           (nx**2 + ny**2 + nz**2)/rc**0.5 + (nx*dxc + ny*dyc + nz*dzc)**2/rc**1.5 +
           (nx*dxc + ny*dyc + nz*dzc)**2/((h + rc**0.5)*rc**1.0))/(h + rc**0.5) +
           2.0*q*(np.tanh(rc**0.5)**2 - 1)*(nx*dxc + ny*dyc + nz*dzc)**2/((h + rc**0.5)*rc**1.0) +
           (np.tanh(rc**0.5)**2 - 1)*(-1.0*(nx**2 + ny**2 + nz**2)/rc**0.5 +
           1.0*(nx*dxc + ny*dyc + nz*dzc)**2/rc**1.5 +
           2.0*(nx*dxc + ny*dyc + nz*dzc)**2*np.tanh(rc**0.5)/rc**1.0))/(h + rc**0.5)**q


    part = np.sum(np.nan_to_num(A*r**1.5 + 3.0*B1*dx*r**0.5 + 3.0*B2*dy*r**0.5 +
                  3.0*B3*dz*r**0.5 + C1*(nx*t + x) + C2*(ny*t + y) +
                  C3*(nz*t + z) + C4),axis=1)

    fp = -(1 - np.tanh(finner)**2)*\
         np.sum(np.nan_to_num(A*(3.0*nx*dx + 3.0*ny*dy + 3.0*nz*dz)*r**0.5 +
                3.0*B1*nx*r**0.5 +
                B1*(3.0*nx*t + 3.0*x - 3.0*xi)*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)/r**0.5 +
                3.0*B2*ny*r**0.5 +
                B2*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)*(3.0*ny*t + 3.0*y - 3.0*yi)/r**0.5 +
                3.0*B3*nz*r**0.5 +
                B3*(1.0*nx*dx + 1.0*ny*dy + 1.0*nz*dz)*(3.0*nz*t + 3.0*z - 3.0*zi)/r**0.5 +
                C1*nx + C2*ny + C3*nz),axis=1)

    fpp = (np.tanh(part)**2 - 1)*\
          (np.sum(np.nan_to_num(3.0*A*(nx**2 + ny**2 + nz**2)*r**0.5 +
                  3.0*A*(nx*dx + ny*dy + nz*dz)**2/r**0.5 +
                  6.0*B1*nx*(nx*dx + ny*dy + nz*dz)/r**0.5 +
                  3.0*B1*(nx**2 + ny**2 + nz**2)*dx/r**0.5 -
                  3.0*B1*dx*(nx*dx + ny*dy + nz*dz)**2/r**1.5 +
                  6.0*B2*ny*(nx*dx + ny*dy + nz*dz)/r**0.5 +
                  3.0*B2*(nx**2 + ny**2 + nz**2)*dy/r**0.5 -
                  3.0*B2*(nx*dx + ny*dy + nz*dz)**2*dy/r**1.5 +
                  6.0*B3*nz*(nx*dx + ny*dy + nz*dz)/r**0.5 +
                  3.0*B3*(nx**2 + ny**2 + nz**2)*dz/r**0.5 -
                  3.0*B3*(nx*dx + ny*dy + nz*dz)**2*dz/r**1.5 -
                  18.0*(1.0*A*(nx*dx + ny*dy + nz*dz)*r**0.5 +
                  B1*nx*r**0.5 +
                  1.0*B1*dx*(nx*dx + ny*dy + nz*dz)/r**0.5 +
                  B2*ny*r**0.5 +
                  1.0*B2*(nx*dx + ny*dy + nz*dz)*dy/r**0.5 +
                  B3*nz*r**0.5 +
                  1.0*B3*(nx*dx + ny*dy + nz*dz)*dz/r**0.5 +
                  0.333333333333333*C1*nx +
                  0.333333333333333*C2*ny +
                  0.333333333333333*C3*nz)**2),axis=1)*\
           np.tanh(part))
    DFT2 = (-(np.sum(wpp) - 2*np.sum(wp)**2/np.sum(w))*f*w/np.sum(w) + f*wpp - 2*f*np.sum(wp)*wp/np.sum(w) + w*fpp + 2*fp*wp - 2*w*fp*np.sum(wp)/np.sum(w))/np.sum(w)
    return np.sum(DFT2)
