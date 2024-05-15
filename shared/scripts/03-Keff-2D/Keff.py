import dolfin as dlf
import numpy as np
import os
import glob
import sys
from mpi4py import MPI

# mpi communicator
comm = dlf.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# loglevel
dlf.set_log_level(dlf.LogLevel.INFO)
# CRITICAL
# ERROR
# WARNING
# INFO
# PROGRESS
# TRACE
# DEBUG

# filename
filename = 'output/Keff'

# mesh file
meshfile = 'mesh/Keff_mesh'

# write xdmf and vtu output for Paraview
results = {'xdmf': True, 'vtk': False}

# time stepping
dt = 1.0e-3
dt_max = 0.1
Tend = 4.0
max_iters = 8
min_iters = 4
t = 0.0
dt_red = 0.5
dt_inc = 1.5

# maximal crack length
crack_max = np.Inf   # stop due to crack size

# crack tip tolerance
crack_tol = 0.75

# elastic constants
lam_matr = dlf.Constant(40)
mu_matr = dlf.Constant(40)
lam_incl = dlf.Constant(65)
mu_incl = dlf.Constant(65)

lam_values = [lam_matr, lam_incl]
mu_values = [mu_matr, mu_incl]

kappa_matr = lam_matr.values()[0]/(2*(lam_matr.values()[0]+mu_matr.values()[0]))

# displacement amplitude ('type' = 'tanh' / 'Kfield')
surfing = {'type': 'tanh',        
           'A': 0.1,
           'd': 0.1,
           'v': 1.0,
           'K1': 10.0}

# degradation function
eta = dlf.Constant(0.001)
a_para = dlf.Constant(0.01)
degrad_func = {'type': 'standard'} # standard/advanded

# locking tolerance
locking = {'tol': 0.02, # dlf.DOLFIN_EPS_LARGE 
           'val': 0.0,
           'init': True,
           'crack': True}

# phase field parameters (given)
Gc = dlf.Constant(1.0)         # KIc = 10 (t)
epsilon = dlf.Constant(0.05)    # 0.1 0.05 0.01
Mob = dlf.Constant(1000.0)
iMob = dlf.Constant(1.0/Mob)

# initial values
u0 = dlf.Constant((0.0, 0.0))
s0 = dlf.Constant(1.0)

# crack geometry, CAREFUL that this is correct for loaded meshes
h0 = 1                  # hight
l0 = 4                  # length
a0 = l0/32              # intial crack

# read mesh file and material distribution into mesh function
meshmatfile = meshfile+'.xdmf'
with dlf.XDMFFile(meshmatfile) as infile:
    mesh = dlf.Mesh(comm)
    infile.read(mesh)
    material = dlf.MeshFunction('size_t', mesh, mesh.topology().dim())
    infile.read(material, 'name_to_read')
if rank == 0:
    print('')
    print('Reading mesh and material:', meshmatfile)
    sys.stdout.flush()
print(dlf.info(mesh, False))
sys.stdout.flush()


# define displacement boundary condition on upper/lower surface
def outer(x, on_boundary):
    val = False
    if dlf.near(np.abs(x[1]), h0/2):
        val = True
    return val


def newcrack(x, on_boundary):
    val = False
    for i in locked_nodes:
        x_lock = mesh.coordinates()[i]
        if (dlf.near(x[0], x_lock[0]) and dlf.near(x[1], x_lock[1])):
            val = True
    return val

# surfing bc
if surfing['type'] == 'tanh':
    Uexpr = dlf.Expression(('0',
                            '0.5*A*(1-tanh((x[0]-v*t)/d))*(x[1]/abs(x[1]))'),
                            A=surfing['A'], v=surfing['v'], d=surfing['d'], t=0.0, degree=1)
if surfing['type'] == 'Kfield':
    Uexpr = dlf.Expression(('K1/(2*mu)*sqrt(sqrt((x[0]-v*t)*(x[0]-v*t)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1],x[0]-v*t)))*cos(atan2(x[1],x[0]-v*t)/2.0)',
                            'K1/(2*mu)*sqrt(sqrt((x[0]-v*t)*(x[0]-v*t)+x[1]*x[1])/(2*pi))*(kappa-cos(atan2(x[1],x[0]-v*t)))*sin(atan2(x[1],x[0]-v*t)/2.0)'),
                            K1=surfing['K1'], v=surfing['v'], t=0.0, kappa=kappa_matr, mu=mu_matr, degree=1)

# function space using mesh and degree
V = dlf.VectorElement('CG', mesh.cell_name(), 1)  # displacements 2nd order slight improvement
T = dlf.FiniteElement('CG', mesh.cell_name(), 1)  # fracture field
W = dlf.FunctionSpace(mesh, V*T)                  # combined space
M = dlf.FunctionSpace(mesh, 'DG', 0)              # cell wise (DG0) space for Lame constants

# copy material distribution mesh function to lam_x and mu_x as DG0 function
lam_x = dlf.Function(M)
mu_x = dlf.Function(M)
help = np.asarray(material.array(), dtype=np.int32)
lam_x.vector()[:] = np.choose(help, lam_values)
mu_x.vector()[:] = np.choose(help, mu_values)

# incl_expr = dlf.Expression('(sqrt((x[0]-xcen)*(x[0]-xcen)+(x[1]-ycen)*(x[1]-ycen))<r0)? val_incl : val_matr', r0=r0, xcen=2, ycen=0, val_incl=lam_incl, val_matr=lam_matr, degree=0)
# lam_x = dlf.interpolate(incl_expr, M)
# incl_expr = dlf.Expression('(sqrt((x[0]-xcen)*(x[0]-xcen)+(x[1]-ycen)*(x[1]-ycen))<r0)? val_incl : val_matr', r0=r0, xcen=2, ycen=0, val_incl=mu_incl, val_matr=mu_matr, degree=0)
# mu_x = dlf.interpolate(incl_expr, M)

# normals
nor = dlf.FacetNormal(mesh)

# define positive/negative function
def pos(x):
    val = 0.5*(x+np.abs(x))
    return val


def neg(x):
    val = 0.5*(x-np.abs(x))
    return val


# define degradation function
def degrad(s):
    if degrad_func['type'] == 'standard':
        degrad = s**2+eta
    if degrad_func['type'] == 'advanced':
        degrad = a_para*(s**3-s**2)+3*s**2-2*s**3+eta
    return degrad


# compute strain energy
def psiel(u, lam, mu):
    eps = dlf.sym(dlf.grad(u))
    # val = 0.5*lam*(dlf.tr(eps))**2+mu*dlf.inner(eps, eps)
    # return val
    # epsvol = dlf.tr(eps)
    # deveps = dlf.dev(eps)
    #
    # extend to 3d

    Kmod = lam + 2/3*mu
    Gmod = mu

    eps3d = dlf.as_tensor([[eps[0, 0], eps[0, 1], 0.0],
                           [eps[1, 0], eps[1, 1], 0.0],
                           [0.0, 0.0, 0.0]])
    epsvol = dlf.tr(eps3d)
    deveps = dlf.dev(eps3d)
    psielpos = 0.5*Kmod*pos(epsvol)**2+Gmod*dlf.inner(deveps, deveps)
    psielneg = 0.5*Kmod*neg(epsvol)**2
    # #
    # direct computation
    # epsvol = eps[0, 0] + eps[1 ,1] 
    # deveps2 = (2/3*eps[0, 0] - 1/3*eps[1, 1])**2 + (2/3*eps[1, 1]-1/3*eps[0, 0])**2 + 2*eps[0, 1]**2 + (1/3*(eps[0, 0] + eps[1, 1]))**2
    # psielpos = 0.5*Kmod*pos(epsvol)**2+mu*deveps2
    # psielneg = 0.5*Kmod*neg(epsvol)**2
    #
    # without tension/compression 
    # psielpos = 0.5*lam*(dlf.tr(eps))**2+mu*dlf.inner(eps, eps)
    # psielneg = 0.0
    return psielpos, psielneg

# undegraded stress
def sig(u, lam, mu):
    eps = dlf.sym(dlf.grad(u))
    val = lam*dlf.tr(eps)*dlf.Identity(2)+2*mu*eps
    return val


# compute crack length and surface energy
def crack(s):
    crack = ((1-s)**2)/(4*epsilon)+epsilon*(dlf.dot(dlf.grad(s), dlf.grad(s)))
    return crack


def psisurf(s):
    psisurf = Gc*crack(s)
    return psisurf


# define solution, restart, trial and test space
w = dlf.Function(W)
wrestart = dlf.Function(W)
wm1 = dlf.Function(W)
dw = dlf.TestFunction(W)
ddw = dlf.TrialFunction(W)

# initialize u field to u0 and s field to s0
w0 = [dlf.interpolate(u0, W.sub(0).collapse()), dlf.interpolate(s0, W.sub(1).collapse())]
dlf.assign(wm1, w0)


# clear outputs
if rank == 0:
    for file in glob.glob(filename+'*.*'):
        os.remove(file)

# prepare newton-log-file
if rank == 0:
    for file in glob.glob(filename+'_log.txt'):
        os.remove(filename+'_log.txt')
    logfile = open(filename+'_log.txt', 'w')
    logfile.write('# Parameters used \n')
    logfile.write('# lambda_m = '+str(lam_matr.values()[0])+'\n')
    logfile.write('# mu_m =     '+str(mu_matr.values()[0])+'\n')
    logfile.write('# lambda_i = '+str(lam_incl.values()[0])+'\n')
    logfile.write('# mu_i =     '+str(mu_incl.values()[0])+'\n')
    logfile.write('# Gc =       '+str(Gc.values()[0])+'\n')
    logfile.write('# epsilon =  '+str(epsilon.values()[0])+'\n')
    logfile.write('# M =        '+str(Mob.values()[0])+'\n')
    logfile.write('# eta =      '+str(eta.values()[0])+'\n')
    logfile.write('# \n')
    logfile.write('# Locking setup \n')
    logfile.write('# lock:init  '+str(locking['init'])+'\n')
    logfile.write('# lock:crack '+str(locking['init'])+'\n')
    logfile.write('# lock:tol = '+str(locking['tol'])+'\n')
    logfile.write('# lock:val = '+str(locking['val'])+'\n')
    logfile.write('# \n')
    logfile.write('# Degradation function \n')
    logfile.write('# type:      '+degrad_func['type']+'\n')
    logfile.write('# eta =      '+str(eta.values()[0])+'\n')
    logfile.write('# a_param =  '+str(a_para.values()[0])+'\n')
    logfile.write('# \n')
    logfile.write('# meshfile:  '+meshfile+'\n')
    logfile.write('# \n')
    logfile.write('# time, dt, no. iterations (for convergence), crack length, psiel, psisurf, crack tip x, crack tip y, Jint x, Jint y \n')
    logfile.close()

# prepare outputs
if results['vtk']:
    vtkout_u = dlf.File(filename+'_u.pvd')
    vtkout_s = dlf.File(filename+'_s.pvd')
    vtkout_lam = dlf.File(filename+'_lam.pvd')
    vtkout_lam << lam_x
    vtkout_mu = dlf.File(filename+'_mu.pvd')
    vtkout_mu << mu_x
if results['xdmf']:
    xdmfout = dlf.XDMFFile(filename+'.xdmf')
    xdmfout.parameters['functions_share_mesh'] = True
    xdmfout.parameters['rewrite_function_mesh'] = False
    xdmfout.parameters['flush_output'] = True
    lam_x.rename('lambda', 'lambda')
    mu_x.rename('mu', 'mu')
    xdmfout.write(lam_x, 0.0)
    xdmfout.write(mu_x, 0.0)

#
# main computation loop
#
t = 0.0
trestart = 0
crack_length = 0
firstloop = True
locked_nodes = np.array([], dtype=int)

# find inital crack
tol = mesh.hmin()/2
if locking['init']:
    for idx, point in enumerate(mesh.coordinates()):
        if point[0]<a0+tol and dlf.near(point[1], 0.0):
            locked_nodes = np.append(locked_nodes, idx)  
   
while t < Tend and crack_length < crack_max:
    # report solution status
    if rank == 0:
        print(' ')
        print('==================================================')
        print('Computing solution at time = {0:.4e}'.format(t))
        print('==================================================')
        print('Current time step dt = {0:.4e}'.format(dt))
        print('==================================================')
        print(' ')
        sys.stdout.flush()

    # update displacment
    Uexpr.t = t

    # split to displacement and crack field
    u, s = dlf.split(w)
    um1, sm1 = dlf.split(wm1)
    du, ds = dlf.split(dw)

    # potential, equilibrium, rate and dr
    psielpos, psielneg = psiel(u, lam_x, mu_x)
    pot = (degrad(s)*psielpos+psielneg+psisurf(s))*dlf.dx
    equi = dlf.derivative(pot, u, du)
    sdrive = dlf.derivative(pot, s, ds)
    rate = (s-sm1)/dt*ds*dlf.dx
    Res = iMob*rate+sdrive+equi
    dResdw = dlf.derivative(Res, w, ddw)
    
    # Displacemnt boundary conditions
    bc_surf = dlf.DirichletBC(W.sub(0), Uexpr, outer, method='pointwise')
    bc = [bc_surf]

    if locking['init'] or locking['crack']:
        bc_newcrack = dlf.DirichletBC(W.sub(1), locking['val'], newcrack, method='pointwise')
        bc.append(bc_newcrack)
    
    # define problem and solver
    problem = dlf.NonlinearVariationalProblem(Res, w, bc, dResdw)
    solver = dlf.NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['nonlinear_solver'] = 'snes'  # 'newton' or 'snes'
    prm['symmetric'] = True           # use of iMob before rate
    prm['newton_solver']['maximum_iterations'] = max_iters
    prm['snes_solver']['maximum_iterations'] = max_iters
    # prm['newton_solver']['linear_solver'] = 'mumps'
    # prm['newton_solver']['preconditioner'] = 'amg'
    prm['snes_solver']['linear_solver'] = 'mumps'
    prm['snes_solver']['preconditioner'] = 'petsc_amg'
    prm['snes_solver']['krylov_solver']['monitor_convergence'] = False
    prm['snes_solver']['krylov_solver']['maximum_iterations'] = 500
    prm['snes_solver']['krylov_solver']['report'] = True
    prm['snes_solver']['lu_solver']['report'] = True
    prm['snes_solver']['lu_solver']['symmetric'] = True
    prm['snes_solver']['lu_solver']['verbose'] = True

    if firstloop:
        if rank == 0:
            print(dlf.info(prm, True))
            sys.stdout.flush()
        firstloop = False

    restart_solution = False
    converged = False
    iters = max_iters

    try:
        (iters, converged) = solver.solve()
    except RuntimeError:
        dt = dt_red*dt
        restart_solution = True
        if rank == 0:
            print('-----------------------------')
            print('NO CONVERGENCE :( ')
            print('Decreasing dt to: {0:.4e}'.format(dt))
            print('-----------------------------')
            sys.stdout.flush()
    if converged and iters < min_iters and t > dlf.DOLFIN_EPS:
        dt = dt_inc*dt
        if dt > dt_max:
            dt = dt_max
        if rank == 0:
            print('-----------------------------')
            print('FAST CONVERGENCE :) ')
            print('Setting dt to: {0:.4e}'.format(dt))
            print('-----------------------------')
            sys.stdout.flush()
    if rank == 0:
        print('-----------------------------')
        print(' No. of iterations: ', iters)
        print(' Converged:         ', converged)
        print(' Restarting:        ', restart_solution)
        print('-----------------------------')
        sys.stdout.flush()

    if not(restart_solution):

        # split solution to displacement and crack field
        u, s = w.split()

        # set name for fields
        u.rename('u', 'u')
        s.rename('s', 's')
        w.rename('w', 'w')

        # compute crack_length
        crack_length = dlf.assemble(crack(s)*dlf.dx)
        if rank == 0:
            print('-------------------------')
            print(' New crack length:', crack_length)
            print('-------------------------')
            sys.stdout.flush()

        # write current solution
        if results['vtk']:
            vtkout_u << (u, t)
            vtkout_s << (s, t)
        if results['xdmf']:
            xdmfout.write(s, t)
            xdmfout.write(u, t)

        # find and update locked points (s<lock_tol)
        s_values = s.compute_vertex_values(mesh)
        
        # find and update locked points (s<locking['tol'])
        if locking['crack']:
            found = np.where(s_values < locking['tol'])
            locked_nodes = np.append(locked_nodes, found)
            locked_nodes = np.unique(locked_nodes)
            # print('proc:', rank, ' locked nodes: ', locked_nodes)
            print('proc:', rank, ' no. locked nodes', locked_nodes.size)
            sys.stdout.flush()

        # find crack tip
        crack_candidates = mesh.coordinates()[np.where(s_values < crack_tol)]
        try:
            crack_tip_idx = np.argmax(crack_candidates[:, 0])
            crack_tip = crack_candidates[crack_tip_idx]
        except ValueError:
            crack_tip = np.array([-np.Inf, -np.Inf])
        print('proc:', rank, 'crack tip:', crack_tip)
        sys.stdout.flush()
        
        (crack_tip_global_x, crack_tip_global_y) = comm.allreduce(sendobj=(crack_tip[0], crack_tip[1]), op=MPI.MAXLOC)    

        if rank == 0:
            print('crack tip coordinates (global):', np.array([crack_tip_global_x, crack_tip_global_y]))
            sys.stdout.flush() 


        # compute eleastic energy
        psielpos, psielneg = psiel(u, lam_x, mu_x)
        enel = dlf.assemble((degrad(s)*psielpos+psielneg)*dlf.dx)
        ensurf = dlf.assemble(psisurf(s)*dlf.dx)

        # compute Jint (ganz außen rum)
        Wen = degrad(s)*psielpos+psielneg
        sigma = degrad(s)*sig(u, lam_x, mu_x)
        eshelby = Wen*dlf.Identity(2) - dlf.grad(u).T*sigma
        Jintx = dlf.assemble((eshelby[0,0]*nor[0]+eshelby[0,1]*nor[1])*dlf.ds)
        Jinty = dlf.assemble((eshelby[1,0]*nor[0]+eshelby[1,1]*nor[1])*dlf.ds)

        # write to newton-log-file
        if rank == 0:
            logfile = open(filename+'_log.txt', 'a')
            logfile.write(str(t)+'  '+str(dt)+'  '+str(iters)+'  '+
                          str(crack_length)+'  '+
                          str(enel)+'  '+str(ensurf)+'  '+
                          str(crack_tip_global_x)+'  '+str(crack_tip_global_y)+'  '+
                          str(Jintx)+'  '+str(Jinty)+'\n')
            logfile.close()
        
        # update
        dlf.assign(wm1, w)
        dlf.assign(wrestart, w)
        trestart = t
        t = t+dt
    else:
        t = trestart+dt
        dlf.assign(w, wrestart)
