  
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc

import ufl 
import numpy as np
import os 
import glob
import sys

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


import alex.os

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# set FEniCSX log level
# dlfx.log.set_log_level(log.LogLevel.INFO)
# dlfx.log.set_output_file('xxx.log')

def mpi_print(output):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# chek for real or complex PETsc
#if rank == 0:
#    print('==================================================')
#    print('PETSc data type:', PETSc.ScalarType)    
#    print('==================================================\n\n')
#    sys.stdout.flush()

# mesh 
# N = 32 

# generate domain
#domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
# domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'foam_mesh.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh()
    

# get dimension and bounds for each mpi process
dim = domain.topology.dim
x_min = np.min(domain.geometry.x[:,0]) 
x_max = np.max(domain.geometry.x[:,0])   
y_min = np.min(domain.geometry.x[:,1]) 
y_max = np.max(domain.geometry.x[:,1])   
z_min = np.min(domain.geometry.x[:,2]) 
z_max = np.max(domain.geometry.x[:,2])

# find global min/max over all mpi processes
comm.Barrier()
x_min_all = comm.allreduce(x_min, op=MPI.MIN)
x_max_all = comm.allreduce(x_max, op=MPI.MAX)
y_min_all = comm.allreduce(y_min, op=MPI.MIN)
y_max_all = comm.allreduce(y_max, op=MPI.MAX)
z_min_all = comm.allreduce(z_min, op=MPI.MIN)
z_max_all = comm.allreduce(z_max, op=MPI.MAX)
comm.Barrier()

mpi_print('spatial dimensions: '+str(dim))
mpi_print('x_min, x_max: '+str(x_min_all)+', '+str(x_max_all))
mpi_print('y_min, y_max: '+str(y_min_all)+', '+str(y_max_all))
mpi_print('z_min, z_max: '+str(z_min_all)+', '+str(z_max_all))
    


# time stepping
dt = 0.05
Tend = 1000.0
max_iters = 8
min_iters = 4
dt_scale_down = 0.5
dt_scale_up = 2.0

# elastic constants
lam = dlfx.fem.Constant(domain, 10.0)
mu = dlfx.fem.Constant(domain, 10.0)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.001)

# phase field parameters
Gc = dlfx.fem.Constant(domain, 1.0)
epsilon = dlfx.fem.Constant(domain, 0.05*(x_max_all - x_min_all))
Mob = dlfx.fem.Constant(domain, 1.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

# define stiffness matrix 
cmat = ufl.as_matrix([[lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
                      [lam, lam+2*mu, lam, 0.0, 0.0, 0.0],
                      [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0,      mu,  0.0, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  mu, 0.0],
                      [0.0, 0.0, 0.0,      0.0,  0.0, mu],
                      ])

# time dependent displacement
umax = 0.25
utop = dlfx.fem.Constant(domain, umax)
ubottom = dlfx.fem.Constant(domain, -umax)

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # fracture fiels
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))


# define top boundary
def top(x):
    return np.isclose(x[1], y_max_all)


# define bottom boundary
def bottom(x):
    return np.isclose(x[1], y_min_all)


def left(x):
    return np.isclose(x[0], x_min_all)

def right(x):
    return np.isclose(x[0], x_max_all)


def front(x):
    return np.isclose(x[2], z_max_all)

def back(x):
    return np.isclose(x[2], z_min_all)


# define crack by boundary
def crack(x):
    x_log = x[0]< (0.25*(x_max_all-x_min_all) + x_min_all)
    y_log = np.isclose(x[1],(y_max_all / 2.0),atol=(0.05*((y_max_all-y_min_all))))
    return np.logical_and(y_log,x_log)
    #return np.logical_and(np.isclose(x[1], ((y_max_all-y_min_all)/2.0 + y_min_all), rtol=(0.05*((y_max_all-y_min_all)))), x[0]< (0.25*(x_max_all-x_min_all) + x_min_all)) 


# define boundary condition on top and bottom
fdim = domain.topology.dim -1
topfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, top)
bottomfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, bottom)

leftfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, left)
rightfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, right)

frontfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, front)
backfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, back)

crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)

topdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, topfacets)
topdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, topfacets)
topdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, topfacets)
bottomdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, bottomfacets)
bottomdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, bottomfacets)
bottomdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, bottomfacets)

rightdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, rightfacets)
rightdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, rightfacets)
rightdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, rightfacets)

leftdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, leftfacets)
leftdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, leftfacets)
leftdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, leftfacets)

frontdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, frontfacets)
frontdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, frontfacets)
frontdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, frontfacets)


backdofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0), fdim, backfacets)
backdofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, backfacets)
backdofs_z = dlfx.fem.locate_dofs_topological(W.sub(0).sub(2), fdim, backfacets)

crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)


bctop_x = dlfx.fem.dirichletbc(0.0, topdofs_x, W.sub(0).sub(0))
bctop_y = dlfx.fem.dirichletbc(0.0, topdofs_y, W.sub(0).sub(1))
bctop_z = dlfx.fem.dirichletbc(0.0, topdofs_z, W.sub(0).sub(2))

bcbottom_x = dlfx.fem.dirichletbc(0.0, bottomdofs_x, W.sub(0).sub(0))
bcbottom_y = dlfx.fem.dirichletbc(0.0, bottomdofs_y, W.sub(0).sub(1))
bcbottom_z = dlfx.fem.dirichletbc(0.0, bottomdofs_z, W.sub(0).sub(2))

bcright_x = dlfx.fem.dirichletbc(0.0, rightdofs_x, W.sub(0).sub(0))
bcright_y = dlfx.fem.dirichletbc(0.0, rightdofs_y, W.sub(0).sub(1))
bcright_z = dlfx.fem.dirichletbc(0.0, rightdofs_z, W.sub(0).sub(2))

bcleft_x = dlfx.fem.dirichletbc(0.0, leftdofs_x, W.sub(0).sub(0))
bcleft_y = dlfx.fem.dirichletbc(0.0, leftdofs_y, W.sub(0).sub(1))
bcleft_z = dlfx.fem.dirichletbc(0.0, leftdofs_z, W.sub(0).sub(2))

bcfront_x = dlfx.fem.dirichletbc(0.0, frontdofs_x, W.sub(0).sub(0))
bcfront_y = dlfx.fem.dirichletbc(0.0, frontdofs_y, W.sub(0).sub(1))
bcfront_z = dlfx.fem.dirichletbc(0.0, frontdofs_z, W.sub(0).sub(2))

bcback_x = dlfx.fem.dirichletbc(0.0, backdofs_x, W.sub(0).sub(0))
bcback_y = dlfx.fem.dirichletbc(0.0, backdofs_y, W.sub(0).sub(1))
bcback_z = dlfx.fem.dirichletbc(0.0, backdofs_z, W.sub(0).sub(2))

bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

# define degradation function
def degrad(s):
    degrad = s**2+eta
    return degrad


# compute strain in Voigt notation
def eps(u):
    eps = ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2), u[1].dx(2)+u[2].dx(1), u[2].dx(0)+u[0].dx(2), u[0].dx(1)+u[1].dx(0)])
    return eps


# compute strain energy in Voigt noation
def psiel(u):
    psiel = 0.5*ufl.dot(eps(u), cmat*eps(u))
    return psiel


# compute surface energy
def psisurf(s):
    psisurf = Gc*(((1-s)**2)/(4*epsilon)+epsilon*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psisurf


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)



# initialize s=1 
wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
wrestart.x.array[:] = wm1.x.array[:]

# prepare newton-log-file
if rank == 0:
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()

# prepare xdmf output 
xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w')
xdmfout.write_mesh(domain)
xdmfout.close()




t = 0
trestart = 0
delta_t = dlfx.fem.Constant(domain, dt)
while t < Tend:

    # update time constant 
    delta_t.value = dt

    # update displacement bc on top and bottom
    utop.value = umax*t
    ubottom.value = -umax*t

    # bctop_y = dlfx.fem.dirichletbc(utop, topdofs_y, W.sub(0).sub(1))
    # bcbottom_y = dlfx.fem.dirichletbc(ubottom, bottomdofs_y, W.sub(0).sub(1))

    # bcs = [bctop_x, bctop_y, bcbottom_x, bcbottom_y, bccrack]
    
    bcs = [bctop_x, bctop_y, bctop_z, bcbottom_x, bcbottom_y, bcbottom_z, bcleft_x, bcleft_y, bcleft_z,
           bcright_x, bcright_y, bcright_z, 
           bcfront_x, bcfront_y, bcfront_z,
           bcback_x, bcback_y, bcback_z, bccrack]

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

    # split to displacement and crack field
    u, s = ufl.split(w)
    um1, sm1 = ufl.split(wm1)
    du, ds = ufl.split(dw)

    # potential, equilibrium, rate and drivign term, residue, derivative
    pot = (degrad(s)*psiel(u)+psisurf(s))*ufl.dx
    equi = ufl.derivative(pot, u, du)
    sdrive = ufl.derivative(pot, s, ds)
    rate = (s-sm1)/delta_t*ds*ufl.dx
    Res = iMob*rate+sdrive+equi
    dResdw = ufl.derivative(Res, w, ddw)

    # define nonlinear problem and solver
    problem = NonlinearProblem(Res, w, bcs, dResdw)
    solver = NewtonSolver(comm, problem)
    solver.report = True
    solver.max_it = max_iters
    
    # set lienar solver
    # ksp = solver.krylov_solver
    # opts = PETSc.Options()
    # opts_prefix = ksp.getOptionsPrefix()
    # print(opts_prefix)
    # opts[f'{opts_prefix}ksp_type'] = 'preonly'
    # opts[f'{opts_prefix}pc_type'] = 'lu'
    # opts[f'{opts_prefix}pc_factor_mat_solver_type'] = 'mumps'
    # ksp.setFromOptions()

    # control adaptive time adjustment
    restart_solution = False
    converged = False
    try:
        (iters, converged) = solver.solve(w)
    except RuntimeError:
        dt = dt_scale_down*dt
        restart_solution = True
        if rank == 0:
            print('-----------------------------')
            print('!!! NO CONVERGENCE => dt: ', dt)
            print('-----------------------------')
            sys.stdout.flush()
    
    if converged and iters < min_iters and t > np.finfo(float).eps:
        dt = dt_scale_up*dt
        if rank == 0:
            print('-----------------------------')
            print('!!! Increasing dt to: ', dt)
            print('-----------------------------')
            sys.stdout.flush()
    if iters > max_iters:
        dt = dt_scale_down*dt
        restart_solution = True
        if rank == 0:
            print('-----------------------------')
            print('!!! Decreasing dt to: ', dt)
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
        u.name='u'
        s.name='s'

        # append xdmf-file
        xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
        xdmfout.write_function(u.collapse(), t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
        xdmfout.write_function(s, t)
        xdmfout.close()

        # write to newton-log-file
        if rank == 0:
            logfile = open(logfile_path, 'a')
            logfile.write(str(t)+'  '+str(dt)+'  '+str(iters)+'\n')
            logfile.close()

        # update
        wm1.x.array[:] = w.x.array[:]
        wrestart.x.array[:] = w.x.array[:]
        # dlf.assign(wm1, w)
        # dlf.assign(wrestart, w)
        trestart = t
        t = t+dt
    else:
        t = trestart+dt
        w.x.array[:] = wrestart.x.array[:]

# stopwatch stop
timer.stop()

# report runtime to screen
if rank == 0:
    print('') 
    print('-----------------------------')
    print('elapsed time:', timer.elapsed())
    print('-----------------------------')
    print('') 
    sys.stdout.flush()

# write runtime to newton-log-file
if rank == 0:
    logfile = open(logfile_path, 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  '+str(timer.elapsed())+'\n')
    logfile.write('# \n')
    logfile.close()
