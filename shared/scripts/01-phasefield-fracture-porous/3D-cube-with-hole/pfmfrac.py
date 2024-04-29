import alex.linearelastic
import dolfinx as dlfx
import dolfinx.plot as plot
import pyvista
from mpi4py import MPI
from petsc4py import PETSc

import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_J_path = alex.os.outputfile_J_full_path(script_path,script_name_without_extension)
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

 
# generate domain
with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'cube_with_hole.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh(name="Grid")
    
    



Tend = 0.4
dt = 0.001

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


# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 2) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

crack_tip_start_location_x = 0.1
crack_tip_start_location_y = 0.5
# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], crack_tip_start_location_y,atol=0.02), x[0]<crack_tip_start_location_x) 


# # define boundary condition on top and bottom
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))


E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
K1 = dlfx.fem.Constant(domain, 1.5 * math.sqrt(Gc.value*E_mod))
xtip = np.array([crack_tip_start_location_x, crack_tip_start_location_y])
xK1 = dlfx.fem.Constant(domain, xtip)

bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
# bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
bcs.append(bccrack)


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

def before_first_time_step():
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
    bc.print_dimensions(x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all, comm)
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare n
    # ewton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_J_output_file(outputfile_J_path)
    # prepare xdmf output 
    
    pp.write_mesh_and_get_outputfile_xdmf(domain, outputfile_xdmf_path, comm)
    # pp.write_phasefield_mixed_solution(domain,outputfile_J_path,w,0,comm)
    

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)

        
phaseFieldProblem = pf.StaticPhaseFieldProblem3D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf)

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mu,
        Gc=Gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]
    
def get_bcs(t):
    v_crack = (1.0+crack_tip_start_location_x-crack_tip_start_location_x)/Tend
    xtip = np.array([crack_tip_start_location_x + v_crack * t, crack_tip_start_location_y])
    xK1 = dlfx.fem.Constant(domain, xtip)

    bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
    # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
    bcs.append(bccrack)
    return bcs

n = ufl.FacetNormal(domain)
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm),5)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)
def after_timestep_success(t,dt,iters):
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    # compute J-Integral
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    J3D_loc_x, J3D_loc_y, J3D_loc_z = alex.linearelastic.get_J_3D(eshelby, outer_normal=n,ds=ds(5))
    
    comm.Barrier()
    J3D_glob_x = comm.allreduce(J3D_loc_x, op=MPI.SUM)
    J3D_glob_y = comm.allreduce(J3D_loc_y, op=MPI.SUM)
    J3D_glob_z = comm.allreduce(J3D_loc_z, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        pp.write_to_J_output_file(outputfile_J_path,t, J3D_glob_x, J3D_glob_y, J3D_glob_z)

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
        pp.print_J_plot(outputfile_J_path,script_path)
        
        # cleanup
        results_folder_path = alex.os.create_results_folder(script_path)
        alex.os.copy_contents_to_results_folder(script_path,results_folder_path)
    

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

