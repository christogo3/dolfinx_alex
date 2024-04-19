  
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

# set FEniCSX log level
# dlfx.log.set_log_level(log.LogLevel.INFO)
# dlfx.log.set_output_file('xxx.log')

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
N = 16 

# generate domain
#domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

# time stepping
dt = 0.05
Tend = 5.0
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
epsilon = dlfx.fem.Constant(domain, 0.1)
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
# umax = 0.25
# utop = dlfx.fem.Constant(domain, umax)
# ubottom = dlfx.fem.Constant(domain, -umax)

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain,Ve)
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # fracture fiels
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))


# define top boundary
def top(x):
    return np.isclose(x[1], 1.0)


# define bottom boundary
def bottom(x):
    return np.isclose(x[1], 0.0)

def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], 1.0)


def front(x):
    return np.isclose(x[2], 1.0)

def back(x):
    return np.isclose(x[2], 0.0)


# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.5), x[0]<0.25) 




# eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
#                     [0.0, 0.6, 0.0],
#                     [0.0, 0.0, 0.0]]))

# def u_macro(x):
#     u_x = eps_mac[0, 0]*x[0] + eps_mac[0, 1]*x[1] + eps_mac[0, 2]*x[2] 
#     u_y = eps_mac[1, 0]*x[0] + eps_mac[1, 1]*x[1] + eps_mac[1, 2]*x[2]
#     u_z = eps_mac[2, 0]*x[0] + eps_mac[2, 1]*x[1] + eps_mac[2, 2]*x[2]
#     return ufl.as_vector((u_x, u_y, u_z))

# mesh cooridnates into x
# x = ufl.SpatialCoordinate(domain)

# u_mac = dlfx.fem.Function(V)
# u_mac_expr = dlfx.fem.Expression(u_macro(x), V.element.interpolation_points())
# u_mac.interpolate(u_mac_expr)

#fdim = domain.topology.dim -1
#bottomfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, bottom)


# # compute macroscopic displacements
#w_mac = dlfx.fem.Function(W)
# # u_mac, s_mac = ufl.split(w_mac)
# w_mac.sub(0).sub(0).interpolate(lambda x: eps_mac[0, 0]*x[0] + eps_mac[0, 1]*x[1] + eps_mac[0, 2]*x[2] )
# w_mac.sub(0).x.scatter_forward()
# w_mac.sub(0).sub(1).interpolate(lambda x: eps_mac[1, 0]*x[0] + eps_mac[1, 1]*x[1] + eps_mac[1, 2]*x[2] )
# w_mac.sub(0).x.scatter_forward()
# w_mac.sub(0).sub(2).interpolate(lambda x: eps_mac[2, 0]*x[0] + eps_mac[2, 1]*x[1] + eps_mac[2, 2]*x[2] )
# w_mac.sub(0).x.scatter_forward()

# u_mac = w_mac.sub(0)
# #u_mac_expr = dlfx.fem.Expression(u_macro(x), W.sub(0).element.interpolation_points())
# #u_mac.interpolate(u_mac_expr)


eps_mac = np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0],
                     [0.0, 0.0, 0.0]])

def linear_displacement_value(x):
    values = np.zeros((3, x.shape[1]))
    # res = 1.0*x[0] + 0.0*x[1] + 0.0*x[2]
    # values[0] = 0.001*x[0] 
    values[0] = eps_mac[0, 0]*x[0] + eps_mac[0, 1]*x[1] + eps_mac[0, 2]*x[2] 
    values[1] = eps_mac[1, 0]*x[0] + eps_mac[1, 1]*x[1] + eps_mac[1, 2]*x[2]
    values[2] = eps_mac[2, 0]*x[0] + eps_mac[2, 1]*x[1] + eps_mac[2, 2]*x[2] 
    return values

def define_bc(value,where_function,in_V, in_mesh):
    '''
    in_V can also be a subspace
    '''
    fdim = domain.topology.dim -1
    boundary_facets = dlfx.mesh.locate_entities_boundary(in_mesh,fdim,where_function)
    bc = dlfx.fem.dirichletbc(value, 
                     dlfx.fem.locate_dofs_topological(in_V, fdim, boundary_facets))
    # bc = fem.dirichletbc(value, 
    #                  fem.locate_dofs_topological(in_V, fdim, boundary_facets), in_V)
    return bc

value = dlfx.fem.Function(V)
value.interpolate(linear_displacement_value)

bc_bottom = define_bc(value,bottom,W.sub(0),domain)
bc_top = define_bc(value,top,W.sub(0),domain)
bc_left = define_bc(value,left,W.sub(0),domain)
bc_right = define_bc(value,right,W.sub(0),domain)
bc_front = define_bc(value,front,W.sub(0),domain)
bc_back = define_bc(value,back,W.sub(0),domain)

# define boundary condition on top and bottom
#fdim = domain.topology.dim -1

crackfacets = dlfx.mesh.locate_entities(domain, domain.topology.dim -1, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), domain.topology.dim -1, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))
#bccrack = define_bc(0.0,crack,W.sub(1),domain)

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
    for file in glob.glob('07_Phasefield/pfmfrac_log.txt'):
        os.remove('07_Phasefield/pfmfrac_log.txt')
    logfile = open('07_Phasefield/pfmfrac_log.txt', 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()

# prepare xdmf output 
xdmfout = dlfx.io.XDMFFile(comm, '07_Phasefield/pfmfrac.xdmf', 'w')
xdmfout.write_mesh(domain)
xdmfout.close()




t = 0
trestart = 0
delta_t = dlfx.fem.Constant(domain, dt)


# compute macroscopic displacements

# eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
#                     [0.0, 0.6*t, 0.0],
#                     [0.0, 0.0, 0.0]]))


while t < Tend:
    # macroscopic strain
    # eps_mac.value = np.array([[0.0, 0.0, 0.0],
    #                 [0.0, 0.6*t, 0.0],
    #                 [0.0, 0.0, 0.0]])
    
    # u_mac_expr = dlfx.fem.Expression(u_macro(x), V.element.interpolation_points())
    # u_mac.interpolate(u_mac_expr)

    # update time constant 
    delta_t.value = dt

    # update displacement bc on top and bottom
    # utop.value = umax*t
    # ubottom.value = -umax*t

    # bctop_y = dlfx.fem.dirichletbc(utop, topdofs_y, W.sub(0).sub(1))
    # bcbottom_y = dlfx.fem.dirichletbc(ubottom, bottomdofs_y, W.sub(0).sub(1))

    # bcs = [bctop_x, bctop_y, bcbottom_x, bcbottom_y, bccrack]
    
    bcs = [bc_top, bc_bottom, bc_front, bc_back, bc_left, bc_right, bccrack]
    bcs = []

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

    # potential, equilibrium, rate and driving term, residual, derivative
    pot = (degrad(s)*psiel(u)+psisurf(s))*ufl.dx # decomposition in fluctuating and macro doesnt work here because nonlinear also?
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
    iters = max_iters + 1 # iters aalways needs to be defined
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
        u_out = u.collapse()
        u_out.name = 'u'
        s.name='s'
        
        # u_mac_out = dlfx.fem.Function(V)
        # u_mac_out.name = 'u_mac'
        # u_mac_out.x.array[:] = u_mac.collapse().x.array

        # append xdmf-file
        xdmfout = dlfx.io.XDMFFile(comm, '07_Phasefield/pfmfrac.xdmf', 'a')
        xdmfout.write_function(u_out, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
        xdmfout.write_function(s, t)
        # xdmfout.write_function(u_mac_out,t)
        xdmfout.close()

        # write to newton-log-file
        if rank == 0:
            logfile = open('07_Phasefield/pfmfrac_log.txt', 'a')
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
    logfile = open('07_Phasefield/pfmfrac_log.txt', 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  '+str(timer.elapsed())+'\n')
    logfile.write('# \n')
    logfile.close()
