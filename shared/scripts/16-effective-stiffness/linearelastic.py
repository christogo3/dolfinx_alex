import alex.homogenization
import alex.linearelastic
import alex.phasefield
import alex.util
import dolfinx as dlfx
from mpi4py import MPI


import ufl 
import numpy as np
import os 
import sys

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'coarse_pores.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh()

dt = 0.05
Tend = 128.0 * dt

# elastic constants
lam = dlfx.fem.Constant(domain, 10.0)
mu = dlfx.fem.Constant(domain, 10.0)
E_mod = alex.linearelastic.get_emod(lam.value, mu.value)

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2) # displacements
V = dlfx.fem.FunctionSpace(domain, Ve)

# define boundary condition on top and bottom
fdim = domain.topology.dim -1

bcs = []
             
# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
urestart =  dlfx.fem.Function(V)
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

def before_first_time_step():
    urestart.x.array[:] = np.ones_like(urestart.x.array[:])
    
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
      
linearElasticProblem = alex.linearelastic.StaticLinearElasticProblem()

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = linearElasticProblem.prep_newton(u,du,ddu,lam,mu)
    return [Res, dResdw]

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)

atol=(x_max_all-x_min_all)*0.02 # for selection of boundary
def get_bcs(t):
    if column_of_cmat_computed[0] < 6:
        eps_mac = alex.homogenization.unit_macro_strain_tensor_for_voigt_eps(domain,column_of_cmat_computed[0])
    else: # to avoid out of bounds index
        eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]]))
    bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, V,eps_mac=eps_mac)
    return bcs

n = ufl.FacetNormal(domain)
simulation_result = np.array([0.0])
vol = (x_max_all-x_min_all) * (y_max_all - y_min_all) * (z_max_all - z_min_all)
Chom = np.zeros((6, 6))

column_of_cmat_computed=np.array([0])

def after_timestep_success(t,dt,iters):
    u.name = "u"
    pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    
    sigma_for_unit_strain = alex.homogenization.compute_averaged_sigma(u,lam,mu, vol)
    
    # write to newton-log-file
    if rank == 0:
        if column_of_cmat_computed[0] < 6:
            Chom[column_of_cmat_computed[0],:] = sigma_for_unit_strain
        else:
            t = 2.0*Tend # exit
            return
        print(column_of_cmat_computed[0])
        column_of_cmat_computed[0] = column_of_cmat_computed[0] + 1
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    urestart.x.array[:] = u.x.array[:] 


               
def after_timestep_restart(t,dt,iters):
    u.x.array[:] = urestart.x.array[:]
     
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        print(np.array_str(Chom, precision=2))
        
        print(alex.homogenization.print_results(Chom))
        
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    u,
    Tend,
    dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm,
    print_bool=True
)

