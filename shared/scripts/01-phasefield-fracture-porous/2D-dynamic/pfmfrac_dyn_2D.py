import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc

import ufl 
import numpy as np
import os 
import glob
import sys
import time


from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# set FEniCSX log level
# log.set_log_level(log.LogLevel.INFO)

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# set output-path
out_path = "output"

# set output-filename base (removing path and suffix): here python file-name
out_file = os.path.basename(__file__).removesuffix('.py')

if rank == 0:
    print(out_file)
    sys.stdout.flush()


# set plot frequency
plot_freq = 5

# set verose level
verbose = False   # True: lengthy output during calculation / False tabular output similar to log-file

# mesh 
Nx = 200
Ny = int(0.4*Nx)
# Nz = int(0.2*Nx)

# generate domain
# domain = dlfx.mesh.create_box(comm,[[0, 0, 0], [1, 0.4, 0.2]],[Nx, Ny, Nz], cell_type=dlfx.mesh.CellType.hexahedron)
domain = dlfx.mesh.create_rectangle(comm, [[0,0],[1,0.8]], [Nx, Ny], cell_type=dlfx.mesh.CellType.quadrilateral)
#domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
hsize = 1.0/Nx

# get mesh dimension
dim = domain.topology.dim
fdim = dim-1

# time stepping
# Tsteps = 100
t_end = 6.0
t_ini = 1
dt = 1.e-3
max_iters = 12
min_iters = 6   # avoid speed up by setting to 1
dt_scale_down = 0.5
dt_scale_up = 1.75

# Newmark parameter
beta = 0.5
gamma = 0.5

# dynamic constants & report
rho = 1.0
lam = 0.5
mu = 0.5
Gmod = mu
Kmod = lam + 2/dim*mu
#
cd = np.sqrt((lam+2*mu)/rho)
cs = np.sqrt(Gmod/rho)
cmax = max(cs, cd)
cfl = cmax*dt/hsize
dt_max = 2.0*hsize/cmax  # factor 2 unclear but from FEAP by Alex    

# phase field parameter
eta = 1.e-9
Gc = 1.0
epsilon = hsize*2.0

# maximum traction
trac_max = 1.5

# function space using mesh and degree
Ve = ufl.VectorElement('Lagrange', domain.ufl_cell(), 1)  # displacements
Te = ufl.FiniteElement('Lagrange', domain.ufl_cell(), 1)  # fracture fiels
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te])) # mixed function space
V, mapu = W.sub(0).collapse()                             # sub-function space u
T, maps = W.sub(1).collapse()                             # sub-function space s

# define top boundary
def top(x):
    return np.isclose(x[1], 0.8)


# define bottom boundary
def bottom(x):
    return np.isclose(x[1], 0.0)


# define crack boundary 
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.4), x[0]<0.25) 


# define boundary condition on top and bottom
topfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, top)
bottomfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, bottom)
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)

marked_facets = np.hstack([topfacets, bottomfacets])
marked_values = np.hstack([np.full_like(topfacets, 1), np.full_like(bottomfacets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = dlfx.mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

ds_top = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)(1)
ds_bottom = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)(2) 
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)

bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))


# compute strain in Voigt notation
def eps(u):
    # eps = ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1)+u[1].dx(0)])
    eps = ufl.sym(ufl.grad(u))
    return eps


# define positive and ngative Macaulay bracket
def pos(x):
    return 0.5*(x+np.abs(x))


def neg(x):
    return 0.5*(x-np.abs(x))


# degradation function and drivative
def degrad(s):
    # a=0.1
    # g = a*(s**3-s**2)+3*s**2-2*s**3+eta
    g = s**2+eta
    return g


# strain energy 
def psi(eps):
    # eps3d = ufl.as_tensor([[eps[0], 0.5*eps[2], 0.0],
    #                        [0.5*eps[2], eps[1], 0.0],
    #                        [0.0, 0.0, 0.0]])
    # epsvol = ufl.tr(eps3d)
    # deveps = ufl.dev(eps3d)
    epsvol = ufl.tr(eps)
    deveps = ufl.dev(eps)
    psiel_pos = 0.5*Kmod*pos(epsvol)**2+Gmod*ufl.inner(deveps, deveps)
    psiel_neg = 0.5*Kmod*neg(epsvol)**2
    return psiel_pos, psiel_neg


def psiel(eps, s):
    psiel_pos, psiel_neg = psi(eps)
    psiel = degrad(s)*psiel_pos+psiel_neg
    return psiel


def kin(v):
    kin = 0.5*rho*ufl.dot(v,v)
    return kin


def sig(eps, s):                   # better to implement vie diff(psiel,eps) 
    # epsvol = ufl.tr(eps)
    # deveps = ufl.dev(eps)
    # sig= Kmod*(degrad(s)*pos(epsvol)+neg(epsvol))*ufl.Identity(dim)+2*Gmod*deveps
    eps_var = ufl.variable(eps)                     # ufl variabel for derivative
    s_var = ufl.variable(s)                         # ufl variable for derivative
    sig = ufl.diff(psiel(eps_var, s_var), eps_var)  # sig = dpsi/deps
    return sig


# crack surcface
def crack(s):
    crack = ((1-s)**2)/(4*epsilon)+epsilon*(ufl.dot(ufl.grad(s), ufl.grad(s)))
    return crack


# fracture surface energy
def psisurf(s):
    psi_surf = Gc*crack(s)
    return psi_surf


# compute time dependent traction
def traction(t, t_ini, trac_max):
    traction = min(t/t_ini*trac_max, trac_max)
    traction_top = np.zeros(dim)
    traction_bottom = np.zeros(dim)
    traction_top[1] = traction        # y-direction
    traction_bottom[1] = -traction    # y-direction
    return traction_top, traction_bottom
    

def update_newmark(beta, gamma, dt, u, um1, vm1, am1, is_ufl=True):
    if is_ufl:
        acc = 1.0/(beta*dt*dt)*(u-um1) - 1.0/(beta*dt)*vm1-(0.5-beta)/beta*am1
        vel = gamma/(beta*dt)*(u-um1)+(1.0-gamma/beta)*vm1+dt*(beta-0.5*gamma)/beta*am1
    else: 
        acc = 1.0/(beta*dt*dt)*(u.x.array[:]-um1.x.array[:])-1.0/(beta*dt)*vm1.x.array[:]-(0.5-beta)/beta*am1.x.array[:]
        vel = gamma/(beta*dt)*(u.x.array[:]-um1.x.array[:])+(1.0-gamma/beta)*vm1.x.array[:]+dt*(beta-0.5*gamma)/beta*am1.x.array[:]
    return acc, vel 


# write log-file
def write_log(time, dt, iters, converged, elapsed_time, verbose=False, initialize=True):
    if initialize:
        log_file = open(out_path+'/'+out_file+'_log.txt', 'w')
        log_file.write('# time      |    dt        | iters  | converged |  elasped time \n')
        log_file.close()
        if not(verbose):
            print('time        |  dt          | iters  | converged |  elasped time')
            print('------------+--------------+--------+-----------+-------------------')
            sys.stdout.flush()
    else:
        log_file = open(out_path+'/'+out_file+'_log.txt', 'a')
        log_file.write('{0:.4e}     {1:.4e}     {2:4d}      {3:5s}      {4:.4e} \n'.format(time, dt, iters, str(converged), elapsed_time))
        log_file.close()
        if not(verbose):
            print('{0:.4e}  |  {1:.4e}  |  {2:4d}  |   {3:5s}   |  {4:.4e} '.format(time, dt, iters, str(converged), elapsed_time))
            sys.stdout.flush()
    return


# write results-file
def write_res(time, dt, iters, pot_el, pot_fr, pot_kin, initialize=True):
    if initialize:
        res_file = open(out_path+'/'+out_file+'_res.txt', 'w')
        res_file.write('# time      |  dt          | iters  |  pot_el        |  pot_fr       | pot_kin \n')
        res_file.close()
    else:
        res_file = open(out_path+'/'+out_file+'_res.txt', 'a')
        res_file.write('{0:.4e}     {1:.4e}     {2:4d}     {3:4e}     {4:.4e}     {5:.4e} \n'.format(time, dt, iters, pot_el, pot_fr, pot_kin))
        res_file.close()
    return


# define combined functions
w = dlfx.fem.Function(W)
dw = ufl.TestFunction(W)

# define velocity and acceleration
um1 = dlfx.fem.Function(V)
v = dlfx.fem.Function(V)
vm1 = dlfx.fem.Function(V)
a = dlfx.fem.Function(V)
am1 = dlfx.fem.Function(V)

# restart fields
wrestart = dlfx.fem.Function(W)
um1restart = dlfx.fem.Function(V)
vm1restart = dlfx.fem.Function(V)
am1restart = dlfx.fem.Function(V)
t_restart = 0.0

# initialize s=1 fields to zero
w.sub(1).x.array[:] = np.ones_like(w.sub(1).x.array[:])

# initialize tractions
trac_top = dlfx.fem.Constant(domain, np.zeros(dim))
trac_bottom = dlfx.fem.Constant(domain, np.zeros(dim))

# prepare xdmf output
xdmfout = dlfx.io.XDMFFile(comm, out_path+'/'+out_file+'.xdmf', 'w')
xdmfout.write_mesh(domain)
#xdmfout.write_meshtags(facet_tag)
xdmfout.close()

# tensor space for postprocessing
TENe = ufl.TensorElement('DG', domain.ufl_cell(), 0)      # constant in element
TEN = dlfx.fem.FunctionSpace(domain, TENe)
SCALe = ufl.FiniteElement('DG', domain.ufl_cell(), 0)     # constant in element
SCAL = dlfx.fem.FunctionSpace(domain, SCALe)

# prepare postprocessing (tensor space)
strain = dlfx.fem.Function(TEN)
stress = dlfx.fem.Function(TEN)
psielastic = dlfx.fem.Function(SCAL)
psifracture = dlfx.fem.Function(SCAL)
kinetic = dlfx.fem.Function(SCAL)

# report show for 3 secs
if rank == 0:
    print('')
    print('NEWMARK PARAMETER')
    print('')
    print('beta:  ', beta)
    print('gamma: ', gamma)
    print('')
    print('ELASTIC CONSTANTS')
    print('lambda:', lam)
    print('mu:    ', mu)
    print('G:     ', Gmod)
    print('K:     ', Kmod)
    print('')
    print('rho:   ', rho)
    print('')
    print('WAVE SPEEDS')
    print('c_D = ', cd)
    print('c_S = ', cs)
    print('')
    print('CFL-CONDIITON (CFL=c_max dt/h < 1)')
    print('h:     ', hsize)
    print('dt:    ', dt)
    print('cmax:  ', cmax)
    print('CFL:   ', cfl)
    if cfl > 1:
        print('WARNING CFL TOO LARGE')
    print('dt_max:', dt_max)
    print('')
    print('MESH-CONDITION eps./h > 2')
    print('h:     ', hsize)
    print('eps.:  ', epsilon)
    if epsilon/hsize<2:
        print('')
        print('WARNING MESH TOO COARSE')
    print('')
    sys.stdout.flush()
time.sleep(1)

# init log-file
if rank == 0:
    write_log(0.0, 0.0, 0, False, 0.0, verbose, initialize=True)

# init res_file
if rank == 0:
    write_res(0.0, 0.0, 0, 0.0, 0.0, 0.0, initialize=True)

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

t = 0
t_restart = 0
plot_counter = 0
iters_counter = 0
while t < t_end+0.5*dt:
    
    # update time
    t = t + dt

    # update tractions 
    tr_top, tr_bottom = traction(t, t_ini, trac_max)
    trac_top.value[:] = tr_top[:]
    trac_bottom.value[:] = tr_bottom[:]
    
    u, s = ufl.split(w)
    du, ds = ufl.split(dw)

    a_ufl, v_ufl = update_newmark(beta, gamma, dt, u, um1, vm1, am1, True)

    # psiel_pos, psiel_neg = psi(eps(u))
    # psi_surf = psisurf(s)
    # pot_in = (degrad(s)*psiel_pos+psiel_neg+psi_surf)*ufl.dx
    pot_in = (psiel(eps(u), s) +psisurf(s))*ufl.dx
    pot_ext = -ufl.dot(trac_top, u)*ds_top - ufl.dot(trac_bottom, u)*ds_bottom
    pot = pot_in + pot_ext
    inertia = rho*ufl.dot(a_ufl, du)*ufl.dx
    variation_u = ufl.derivative(pot, u, du)
    variation_s = ufl.derivative(pot, s, ds)
    Res = inertia + variation_u + variation_s

    bcs = [bccrack]

    problem = dlfx.fem.petsc.NonlinearProblem(Res, w, bcs)
    solver = dlfx.nls.petsc.NewtonSolver(comm, problem)
    solver.report = True
    solver.max_it = max_iters

    # configure linear solver (direct solver mumps)
    lin_solver = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = lin_solver.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg" 
    opts[f"{option_prefix}pc_type"] = "jacobi" 
    # opts[f"{option_prefix}ksp_type"] = "preonly" 
    # opts[f"{option_prefix}pc_type"] = "lu" 
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    lin_solver.setFromOptions()

    # prepare solver behavior
    restart = False
    iters = max_iters
    converged = False

    # report solution status     
    if rank == 0:
        if verbose:
            print(' ')
            print('============================================================')
            print('Computing solution at time = {0:.4e}/{1:.4e}'.format(t, t_end))
            print('                   with dt = {0:.4e}'.format(dt))
            print('============================================================') 
    try:
        (iters, converged) = solver.solve(w)    
    except RuntimeError:
        restart = True
    
    # update iters counter
    iters_counter = iters_counter + iters

    # forward solution to mpi processes (might be necessary)
    w.x.scatter_forward()

    # write log and/or tabular
    if rank == 0:
        elapsed_time = timer.elapsed()[0]
        write_log(t, dt, iters, converged, elapsed_time, verbose, initialize=False)

    if rank == 0:
        if verbose:
            print('Iterations / converged:  ', iters,'/', converged )
            sys.stdout.flush()

    if not(restart):   # no restart
        # split solution
        u = w.sub(0).collapse()
        s = w.sub(1).collapse()    
        
        # compute update variables at t_n
        a.x.array[:], v.x.array[:] = update_newmark(beta, gamma, dt, u, um1, vm1, am1, False)

        # copy for next iteration
        um1.x.array[:] = u.x.array[:]
        vm1.x.array[:] = v.x.array[:]
        am1.x.array[:] = a.x.array[:]

        # set restart 
        t_restart = t
        wrestart.x.array[:] = w.x.array[:]
        um1restart.x.array[:] = um1.x.array[:]
        vm1restart.x.array[:] = vm1.x.array[:]
        am1restart.x.array[:] = am1.x.array[:]

        # check for speedup        
        if iters < min_iters and t > np.finfo(float).eps:
            dt = float(min(dt*dt_scale_up, dt_max))
            if rank == 0:
                if verbose:
                    print('increasing dt to {0:.4e}'.format(dt))
                    sys.stdout.flush()

        # postprocessing of fields
        strain_expr = dlfx.fem.Expression(eps(u), TEN.element.interpolation_points())
        strain.interpolate(strain_expr)
        stress_expr = dlfx.fem.Expression(sig(eps(u), s), TEN.element.interpolation_points())
        stress.interpolate(stress_expr)
        psielastic_expr = dlfx.fem.Expression(psiel(eps(u), s), SCAL.element.interpolation_points())
        psielastic.interpolate(psielastic_expr)
        psifracture_expr = dlfx.fem.Expression(psisurf(s), SCAL.element.interpolation_points())
        psifracture.interpolate(psifracture_expr)
        kinetic_expr = dlfx.fem.Expression(kin(v), SCAL.element.interpolation_points())
        kinetic.interpolate(kinetic_expr)

        # postprocessing of global variables
        potel_form = dlfx.fem.form(psiel(eps(u), s)*ufl.dx)
        potel = dlfx.fem.assemble_scalar(potel_form)
        potfr_form = dlfx.fem.form(psisurf(s)*ufl.dx)
        potfr = dlfx.fem.assemble_scalar(potfr_form)
        potkin_form = dlfx.fem.form(kin(v)*ufl.dx)
        potkin = dlfx.fem.assemble_scalar(potkin_form)
        # collect results from processors
        comm.Barrier()
        potel_all = comm.allreduce(potel, op=MPI.SUM)
        potfr_all = comm.allreduce(potfr, op=MPI.SUM)
        potkin_all = comm.allreduce(potkin, op=MPI.SUM)
        comm.Barrier()

        # write results file
        if rank == 0:
            write_res(t, dt, iters, potel_all, potfr_all, potkin_all, initialize=False)

        # write xdmf plots
        if plot_counter % plot_freq == 0:
            u.name = 'u'
            s.name = 's'
            strain.name = 'eps'
            stress.name = 'sig'
            psielastic.name = 'psi_el'
            psifracture.name = 'psi_fr'
            kinetic.name = 'psi_kin'
            xdmfout = dlfx.io.XDMFFile(comm, out_path+'/'+out_file+'.xdmf', 'a')
            try:
                xdmfout.write_function(u, t)
                xdmfout.write_function(s, t)
                xdmfout.write_function(strain, t)
                xdmfout.write_function(stress, t)
                xdmfout.write_function(psielastic, t)
                xdmfout.write_function(psifracture, t)
                xdmfout.write_function(kinetic,t )
                xdmfout.close()
            except RuntimeError:
                xdmfout.close()
        plot_counter = plot_counter + 1
    
    else:   # restart
        t = t_restart
        dt = dt*dt_scale_down
        w.x.array[:] = wrestart.x.array[:]
        um1.x.array[:] = um1restart.x.array[:]
        vm1.x.array[:] = vm1restart.x.array[:]
        am1.x.array[:] = am1restart.x.array[:]
        if rank == 0:
            if verbose:
                print('Restarting  at  t  = {0:.4e}'.format(t))
                print('Restarting with dt = {0:.4e}'.format(dt))
                sys.stdout.flush()
        
    
# stopwatch stop
timer.stop()

# report runtime to screen
if rank == 0:
    print('') 
    print('-----------------------------')
    print('elapsed time:  ', timer.elapsed())
    print('-----------------------------')
    print('total # iters: ', iters_counter)
    print('-----------------------------')
    print('') 
    sys.stdout.flush()
