import argparse
import os
import shutil
from datetime import datetime
from mpi4py import MPI
import alex.util
import pfmfrac_function as sim

import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc
import basix


import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
import alex.phasefield as pf

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    alex.util.print_dolfinx_version()

# Define argument parser
parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
parser.add_argument("--mesh_file", type=str, required=True, help="Name of the mesh file")
parser.add_argument("--lam_param", type=float, required=True, help="Lambda parameter")
parser.add_argument("--mue_param", type=float, required=True, help="Mu parameter")
parser.add_argument("--Gc_param", type=float, required=True, help="Gc parameter")
parser.add_argument("--eps_factor_param", type=float, required=True, help="Epsilon factor parameter")
parser.add_argument("--element_order", type=int, required=True, help="Element order")

# Parse arguments
args = parser.parse_args()

# Extract script path and name
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Run the simulation
# sim.run_simulation(args.mesh_file,
#                    args.lam_param,
#                    args.mue_param,
#                    args.Gc_param,
#                    args.eps_factor_param,
#                    args.element_order,
#                    comm)

#################### START DOLFINX
# script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
working_folder = script_path #alex.os.scratch_directory # or script_path if local
logfile_path = alex.os.logfile_full_path(working_folder,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(working_folder,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(working_folder,script_name_without_extension)
    # set FEniCSX log level
    # dlfx.log.set_log_level(log.LogLevel.INFO)
    # dlfx.log.set_output_file('xxx.log')

    # set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
# comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory, args.mesh_file+".xdmf"), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh()

dt = dlfx.fem.Constant(domain,0.0001)
t_global = dlfx.fem.Constant(domain,0.0)
Tend = 10.0 * dt.value



# function space using mesh and degree
Ve = basix.ufl.element("P", domain.basix_cell(), args.element_order, shape=(domain.geometry.dim,)) #displacements
Se = basix.ufl.element("P", domain.basix_cell(), args.element_order, shape=())# fracture fields
W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Se]))
# Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), args.element_order) # displacements
# Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), args.element_order) # fracture fields
# W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if comm.Get_rank() == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

# elastic constants
lam = dlfx.fem.Constant(domain, args.lam_param)
mu = dlfx.fem.Constant(domain, args.mue_param)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.001)
Gc = dlfx.fem.Constant(domain, args.Gc_param)
epsilon = dlfx.fem.Constant(domain, (y_max_all-y_min_all)/args.eps_factor_param)

Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)


E_mod = le.get_emod(lam=lam,mu=mu)

# setting K1 so it always breaks
epsilon0 = dlfx.fem.Constant(domain, (y_max_all-y_min_all) / 50.0)
h_coarse_mean=0.024636717648428213 # TODO needs to be adapted to actual mesh
if args.mesh_file == "coarse_pores":
    hh = h_coarse_mean
elif args.mesh_file == "medium_pores":
    hh = h_coarse_mean/2.0
elif args.mesh_file == "fine_pores": 
    hh = h_coarse_mean/4.0
Gc_num = (1.0 + hh / epsilon.value ) * Gc.value
K1 = dlfx.fem.Constant(domain, 2.5 * math.sqrt(epsilon0) / math.sqrt(epsilon) * math.sqrt(Gc_num * E_mod))


# define crack by boundary
crack_tip_start_location_x = 0.1*(x_max_all-x_min_all) + x_min_all
crack_tip_start_location_y = (y_max_all + y_min_all) / 2.0
def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1],crack_tip_start_location_y,atol=(0.02*((y_max_all-y_min_all))))
    return np.logical_and(y_log,x_log)


# define boundary condition on top and bottom
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

            
# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
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
        pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    # pp.write_meshoutputfile(domain, outputfile_vtk_path, comm)

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


# setup tracking
S = dlfx.fem.functionspace(domain,Se)
s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
s_zero_for_tracking_at_nodes.interpolate(sub_expr)

atol=(x_max_all-x_min_all)*0.02 # for selection of boundary


xtip = np.array([0.0,0.0,0.0],dtype=dlfx.default_scalar_type)
xK1 = dlfx.fem.Constant(domain, xtip)
v_crack = 2.0*(x_max_all-crack_tip_start_location_x)/Tend
vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack,0.0,0.0],dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([crack_tip_start_location_x,crack_tip_start_location_y,0.0],dtype=dlfx.default_scalar_type))

[Res, dResdw] = get_residuum_and_gateaux(delta_t=dt)
w_D = dlfx.fem.Function(W) # for dirichlet BCs

front_back = bc.get_frontback_boundary_of_box_as_function(domain,comm,atol=0.1*atol)
bc_front_back = bc.define_dirichlet_bc_from_value(domain,0.0,2,front_back,W,0)

def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    xxK1 = crack_start + vcrack_const * t_global 
    dx = x[0] - xxK1[0]
    dy = x[1] - xxK1[1]
    
    nu = le.get_nu(lam=lam.value, mu=mu.value)
    r = ufl.sqrt(ufl.inner(dx,dx) + ufl.inner(dy,dy))
    theta = ufl.atan2(dy, dx)
    
    u_x = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    u_z = ufl.as_ufl(0.0)
    return ufl.as_vector([u_x, u_y, u_z])
  
bc_expression = dlfx.fem.Expression(compute_surf_displacement(),W.sub(0).element.interpolation_points())

top_bottom = bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, top_bottom)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary)

def get_bcs(t):
    bcs = []
    w_D.sub(0).interpolate(bc_expression)
    # w_D.x.scatter_forward()
    
    # if rank == 0:
    #     print(f"BC arrays are equal after: {np.array_equal(w_D_old,w_D.x.array[:])}")
        
    bc_surf : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D,dofs_at_boundary)
    
    xtip[0] = crack_tip_start_location_x + v_crack * t
    xtip[1] = crack_tip_start_location_y
    
    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    bcs.append(bccrack)
    bcs.append(bc_surf)
    bcs.append(bc_front_back)
    
    return bcs

n = ufl.FacetNormal(domain)
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol),5)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)

top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_top_boundary_of_box_as_function(domain, comm,atol=atol),1)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,100.0)

Work = dlfx.fem.Constant(domain,0.0)
def after_timestep_success(t,dt,iters):
    
    # pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path,w,t,comm)
    
    sigma = phaseFieldProblem.sigma_degraded(u,s,lam.value,mu.value,eta)
    Rx_top, Ry_top, Rz_top = pp.reaction_force_3D(sigma,n=n,ds=ds_top_tagged(1),comm=comm)
    
    um1, _ = ufl.split(wm1)

    dW = pp.work_increment_external_forces(sigma,u,um1,n,ds(5),comm=comm)
    Work.value = Work.value + dW
    
    A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
    # compute J-Integral
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    # divEshelby = ufl.div(eshelby)
    # pp.write_vector_fields(domain=domain,comm=comm,vector_fields_as_functions=[divEshelby],
    #                         vector_field_names=["Ge"], 
    #                         outputfile_xdmf_path=outputfile_xdmf_path,t=t)
    
    J3D_glob_x, J3D_glob_y, J3D_glob_z = alex.linearelastic.get_J_3D(eshelby, ds=ds(5), n=n,comm=comm)

    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        

    
    # s_aux = dlfx.fem.Function(S)
    # s_aux.interpolate(s)
    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking_at_nodes.interpolate(s)
    x_tip, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    if rank == 0:
        print("Crack tip position x: " + str(x_tip))
        pp.write_to_graphs_output_file(outputfile_graph_path,t, J3D_glob_x, J3D_glob_y, J3D_glob_z,x_tip, xtip[0], Rx_top, Ry_top, Rz_top, dW, Work.value, A)
    
    
    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    # wrestart.x.scatter_forward()
    # wm1.x.scatter_forward()
    
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    
    # pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    # pp.write_phasefield_mixed_solution(domain,outputfile_vtk_path, w, t, comm)
    # write to newton-log-file
    
    
def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]
    # w.x.scatter_forward()
    
    
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # only write final crack pattern
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, 0.0, comm)

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy", "Jz", "x_tip", "xtip", "Rx", "Ry", "Rz", "dW", "W", "A"])

        # cleanup only necessary on cluster
        # results_folder_path = alex.os.create_results_folder(script_path)
        # alex.os.copy_contents_to_results_folder(script_path,results_folder_path)

# sol.solve_with_newton_adaptive_time_stepping_old(
#     domain,
#     w,
#     Tend,
#     dt,
#     before_first_timestep_hook=before_first_time_step,
#     after_last_timestep_hook=after_last_timestep,
#     before_each_timestep_hook=before_each_time_step,
#     get_residuum_and_gateaux=get_residuum_and_gateaux,
#     get_bcs=get_bcs,
#     after_timestep_restart_hook=after_timestep_restart,
#     after_timestep_success_hook=after_timestep_success,
#     comm=comm,
#     print=True
# )

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
    comm=comm,
    print_bool=True,
    solver=None,
    t=t_global
)

#################### END DOLFINX


# Put files in folders
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a folder name based on the parameters, current time, and mesh file name
folder_name = (f"simulation_{current_time}_"
               f"{args.mesh_file}_"
               f"lam{args.lam_param}_mue{args.mue_param}_Gc{args.Gc_param}_eps{args.eps_factor_param}_order{args.element_order}")
comm.barrier()
if comm.Get_rank() == 0:
    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.join(script_path, folder_name)):
        os.makedirs(os.path.join(script_path, folder_name))

    files_to_move = ["script_combined.xdmf", "script_combined.h5", "script_combined_graphs.txt", "script_combined_log.txt"]  # Replace with actual files

    for file in files_to_move:
        file_path = os.path.join(script_path, file)
        if os.path.exists(file_path):
            shutil.move(file_path, os.path.join(script_path, folder_name, os.path.basename(file)))
        else:
            print(f"File {file_path} does not exist and cannot be moved.")