import argparse
import dolfinx as dlfx
import os
from mpi4py import MPI
import numpy as np
from array import array
import ufl

import alex.heterogeneous as het
import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic
import math

from petsc4py import PETSc as petsc
import sys
import basix

import shutil
from datetime import datetime
import alex.plasticity



script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)
parameter_path = os.path.join(script_path,"parameters.txt")

# set MPI environment
comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)

if rank == 0:
    alex.util.print_dolfinx_version()


    
N=50
domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
    
dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)
    
x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)


# Material definition ##################################################
micro_material_marker = 1
effective_material_marker = 0





# Simulation parameters ####
dt_start = 0.001
dt_max_in_critical_area = 2.0e-7
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
# Tend = 10.0 * dt_global.value
dt_global.value = dt_max_in_critical_area
dt_max = dlfx.fem.Constant(domain,10*dt_start)



la = dlfx.fem.Constant(domain, 1.0)
mu = dlfx.fem.Constant(domain, 1.0)
gc = dlfx.fem.Constant(domain, 1.0)
eta = dlfx.fem.Constant(domain, 0.00001)
epsilon = dlfx.fem.Constant(domain, 0.1)
Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

# Function space and FE functions ########################################################

Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,)) #displacements
V = dlfx.fem.functionspace(domain, Ve)

# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
du = ufl.TestFunction(V)





# setting K1 so it always breaks
K1 = dlfx.fem.Constant(domain, 10.0 * math.sqrt(1.0 * 2.5))

# define crack by boundary
crack_tip_start_location_x = 0.1
crack_tip_start_location_y = 0.5 #(y_max_all + y_min_all) / 2.0
def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1],crack_tip_start_location_y,atol=0.01)
    return np.logical_and(y_log,x_log)

v_crack = 1.0 # const for all simulations
Tend = (x_max_all-x_min_all) * 2.0 / v_crack

## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)



# TODO clean up
CC = dlfx.fem.Constant(domain,0.001)
nn = dlfx.fem.Constant(domain,1.0)
tol = dlfx.fem.Constant(domain,0.000001)
z = dlfx.fem.Constant(domain,0.0)
Id = ufl.as_matrix(((1,z),
                    (z,1)))
rambergOsgoodProblem = alex.plasticity.Ramberg_Osgood(dx=ufl.dx,Id=Id,tol=tol,C=CC,n=nn)



timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    
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



def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = rambergOsgoodProblem.prep_newton(u=u,lam=la,mu=mu,du=du)
    return [Res, dResdw]



atol=(x_max_all-x_min_all)*0.000 # for selection of boundary

# surfing BCs
xtip = np.array([0.0,0.0,0.0],dtype=dlfx.default_scalar_type)
xK1 = dlfx.fem.Constant(domain, xtip)
# v_crack = 1.2*(x_max_all-crack_tip_start_location_x)/Tend
vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack,0.0,0.0],dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([0.0,crack_tip_start_location_y,0.0],dtype=dlfx.default_scalar_type))

[Res, dResdw] = get_residuum_and_gateaux(delta_t=dt_global)
u_D = dlfx.fem.Function(V) # for dirichlet BCs

# front_back = bc.get_frontback_boundary_of_box_as_function(domain,comm,atol=0.1*atol)
# bc_front_back = bc.define_dirichlet_bc_from_value(domain,0.0,2,front_back,W,0)

def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    xxK1 = crack_start + vcrack_const * t_global 
    dx = x[0] - xxK1[0]
    dy = x[1] - xxK1[1]
    
    nu = alex.linearelastic.get_nu(lam=la, mu=mu) # should be effective values?
    r = ufl.sqrt(ufl.inner(dx,dx) + ufl.inner(dy,dy))
    theta = ufl.atan2(dy, dx)
    
    u_x = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    u_z = ufl.as_ufl(0.0)
    return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

bc_expression = dlfx.fem.Expression(compute_surf_displacement(),V.element.interpolation_points())
# boundary_surfing_bc = bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol*0.0) #bc.get_boundary_for_surfing_boundary_condition_2D(domain,comm,atol=atol,epsilon=epsilon.value) #bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol)
boundary_surfing_bc = bc.get_2D_boundary_of_box_as_function(domain,comm,atol=atol*0.0,epsilon=epsilon.value)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(V, fdim, facets_at_boundary) 



def get_bcs(t):
    bcs = []
    #t_global.value = t
    #xtip[0] = 0.0 + v_crack * t
    #xtip[1] = crack_tip_start_location_y
    
    
    u_D.interpolate(bc_expression)
    bc_surf : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(u_D,dofs_at_boundary)

    bcs.append(bc_surf)
    # bcs.append(bc_front_back)
    return bcs




# def in_steg_to_be_measured(x_ct):
#     #x_center = (w_cell) * 1.5 + dhole/2
#     first_low, first_high, second_low, second_high = steg_bounds_to_be_measured()
    
#     in_first_steg = first_low <= x_ct <= first_high
#     in_second_steg = second_low <= x_ct <= second_high
    
#     return in_first_steg or in_second_steg

# def steg_bounds_to_be_measured():
#     first_low = w_cell + wsteg/2.0 #+ dhole
#     first_high = first_low + wsteg #- (0.01*wsteg)
    
#     second_low = first_high #
#     second_high = second_low + wsteg 
#     return first_low,first_high,second_low,second_high


n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)
# s_zero_for_tracking = pp.get_s_zero_field_for_tracking(domain)

top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_top_boundary_of_box_as_function(domain, comm,atol=atol*0.0),top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags)

Work = dlfx.fem.Constant(domain,0.0)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,20.0)
def after_timestep_success(t,dt,iters):
    # update u from Δu
    
    
    
    
    sigma = rambergOsgoodProblem.sigma_undegraded_vol_deviatoric(u,la,mu)
    Rx_top, Ry_top = pp.reaction_force(sigma,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    
   

    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    
    
    
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    
    pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm,V)

    #pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

def after_timestep_restart(t,dt,iters):
    return
    #w.x.array[:] = wrestart.x.array[:]

def after_last_timestep():
    timer.stop()
    #pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t_global.value, comm)
    # stopwatch stop
    
    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        #pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy","x_pf_crack","x_macr","Rx", "Ry", "dW", "W", "A", "dt", "E_el"])
        

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    u,
    Tend,
    dt_global,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm,
    print_bool=True,
    t=t_global,
    dt_max=dt_max,
    trestart=trestart_global,
    #max_iters=20
)




# copy relevant files

# Step 1: Create a unique timestamped directory
def create_timestamped_directory(base_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = os.path.join(base_dir, f"simulation_{timestamp}")
    os.makedirs(directory_name, exist_ok=True)
    return directory_name

# Step 2: Copy files to the timestamped directory
def copy_files_to_directory(files, target_directory):
    for file in files:
        if os.path.exists(file):
            shutil.copy(file, target_directory)
        else:
            print(f"Warning: File '{file}' does not exist and will not be copied.")

# if rank == 0:
#     # pp.append_to_file(parameters=parameters_to_write,filename=parameter_path,comm=comm)
#     files_to_copy = [
#         parameter_path,
#         outputfile_graph_path,
#         os.path.join(script_path,script_name_without_extension+".py"),
#         #mesh_file,  # Add more files as needed
#         os.path.join(script_path,"graphs.png"),
#         os.path.join(script_path,script_name_without_extension+".xdmf"),
#         os.path.join(script_path,script_name_without_extension+".h5")
#     ]
        
#     # Create the directory
#     target_directory = create_timestamped_directory(base_dir=script_path)
#     print(f"Created directory: {target_directory}")

#     # Copy the files
#     copy_files_to_directory(files_to_copy, target_directory)
#     print("Files copied successfully.")
