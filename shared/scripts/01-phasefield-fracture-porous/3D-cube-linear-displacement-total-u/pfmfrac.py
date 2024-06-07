import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc

import ufl 
import numpy as np
import os 
import sys

import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

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

# mesh 
N = 16 

# generate domain
#domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

Tend = 0.4
dt = 0.05

# elastic constants
lam = dlfx.fem.Constant(domain, 10.0)
mu = dlfx.fem.Constant(domain, 10.0)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.001)

# phase field parameters
Gc = dlfx.fem.Constant(domain, 1.0)
epsilon = dlfx.fem.Constant(domain, 0.05)
Mob = dlfx.fem.Constant(domain, 1.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)


# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain,Ve)
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 2) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.5), x[0]<0.25) 

eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.6, 0.0],
                    [0.0, 0.0, 0.0]]))

# # define boundary condition on top and bottom
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))
bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,eps_mac,0)
bcs.append(bccrack)


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

def before_first_time_step():
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
    # prepare xdmf output 
    xdmfout = pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)


def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
        

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    phaseFieldProblem = pf.StaticPhaseFieldProblem3D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf)
    
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mu,
        Gc=Gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]
    
def get_bcs(t):
    # can be updated here
    return bcs

def after_timestep_success(t,dt,iters):
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)

    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]
    
    
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
    

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    w,
    Tend,
    dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm
)

