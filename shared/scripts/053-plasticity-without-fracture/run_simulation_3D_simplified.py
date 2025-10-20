import argparse
import dolfinx as dlfx
import os
from mpi4py import MPI
import numpy as np
from array import array
import ufl
import dolfinx.fem as fem

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

N=10
domain = dlfx.mesh.create_unit_cube(comm, N, N, N, cell_type=dlfx.mesh.CellType.hexahedron)
    
dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)
    
x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)


# Material definition ##################################################
micro_material_marker = 1
effective_material_marker = 0


# Simulation parameters ####
dt_start = 0.01
dt_max_in_critical_area = dt_start
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
Tend = 3.0
dt_global.value = dt_max_in_critical_area
dt_max = dlfx.fem.Constant(domain,dt_max_in_critical_area)



la = dlfx.fem.Constant(domain, 1.0)
mu = dlfx.fem.Constant(domain, 1.0)

sig_y = dlfx.fem.Constant(domain, 1.0)
hard = dlfx.fem.Constant(domain, 0.6)

# Function space and FE functions ########################################################
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain, Ve)


# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
urestart =  dlfx.fem.Function(V)
um1 =  dlfx.fem.Function(V) # trial space
um1.x.array[:] = np.zeros_like(um1.x.array[:])
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

deg_quad = 1  # quadrature degree for internal state variable representation
gdim = 3

'''# function space for 3d fields
Ve_3d = ufl.TensorElement("Lagrange", domain.ufl_cell(), 1, shape=(3,3)) # displacements
V_3d = dlfx.fem.FunctionSpace(domain, Ve_3d)
# Set e_p and e_p_n up in a TensorFunctionSpace
e_p_n = fem.Function(V_3d, name='e_p')
e_p_n_tmp = fem.Function(V_3d, name='e_p_tmp')
e_p_n.x.array[:] = 0.0
e_p_n_tmp.x.array[:] = 0.0'''

'''# Get the sub-space for the (0,0) component and its dofmap
V_00, map_00 = V.sub([0, 0]).collapse()
# Create the numpy array you want to assign
num_dofs_component = V_00.dofmap.index_map.size_local
zero_array = np.zeros_like(num_dofs_component)'''

H,alpha_n,alpha_tmp = alex.plasticity.define_internal_state_variables_basix_b(gdim, domain, deg_quad,quad_scheme="default")
W0e = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad)
W0 = fem.functionspace(domain, W0e)

e_p_11_n_tmp = fem.Function(W0, name="e_p_11_tmp")
e_p_22_n_tmp = fem.Function(W0, name="e_p_22_tmp")
e_p_12_n_tmp = fem.Function(W0, name="e_p_12_tmp")
e_p_33_n_tmp = fem.Function(W0, name="e_p_33_tmp")
e_p_13_n_tmp = fem.Function(W0, name="e_p_13_tmp")
e_p_23_n_tmp = fem.Function(W0, name="e_p_23_tmp")
e_p_11_n = fem.Function(W0, name="e_p_11")
e_p_22_n = fem.Function(W0, name="e_p_22")
e_p_12_n = fem.Function(W0, name="e_p_12")
e_p_33_n = fem.Function(W0, name="e_p_33")
e_p_13_n = fem.Function(W0, name="e_p_13")
e_p_23_n = fem.Function(W0, name="e_p_23")

dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)
H.x.array[:] = np.zeros_like(H.x.array[:])
alpha_n.x.array[:] = np.zeros_like(alpha_n.x.array[:])
alpha_tmp.x.array[:] = np.zeros_like(alpha_tmp.x.array[:])
e_p_11_n.x.array[:] = np.zeros_like(e_p_11_n.x.array[:])
e_p_22_n.x.array[:] = np.zeros_like(e_p_22_n.x.array[:])
e_p_12_n.x.array[:] = np.zeros_like(e_p_12_n.x.array[:])
e_p_33_n.x.array[:] = np.zeros_like(e_p_33_n.x.array[:])
e_p_13_n.x.array[:] = np.zeros_like(e_p_13_n.x.array[:])
e_p_23_n.x.array[:] = np.zeros_like(e_p_23_n.x.array[:])
e_p_11_n_tmp.x.array[:] = np.zeros_like(e_p_11_n_tmp.x.array[:])
e_p_22_n_tmp.x.array[:] = np.zeros_like(e_p_22_n_tmp.x.array[:])
e_p_12_n_tmp.x.array[:] = np.zeros_like(e_p_12_n_tmp.x.array[:])
e_p_33_n_tmp.x.array[:] = np.zeros_like(e_p_33_n_tmp.x.array[:])
e_p_13_n_tmp.x.array[:] = np.zeros_like(e_p_13_n_tmp.x.array[:])
e_p_23_n_tmp.x.array[:] = np.zeros_like(e_p_23_n_tmp.x.array[:])

# setting K1 so it always breaks
#K1 = dlfx.fem.Constant(domain, 1.0 * math.sqrt(1.0 * 2.5))



## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


e_p_n_3D = ufl.as_tensor([[e_p_11_n, e_p_12_n, e_p_13_n], 
                          [e_p_12_n, e_p_22_n, e_p_23_n],
                          [e_p_13_n, e_p_23_n, e_p_33_n]])
plasticityProblem = alex.plasticity.Plasticity_incremental_2D(sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n_3D,H=H)

# pf.StaticPhaseFieldProblem2D_incremental_plasticity(degradationFunction=pf.degrad_cubic,
#                                                    psisurf=pf.psisurf_from_function,dx=dx, sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n_3D,H=H)
timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    urestart.x.array[:] = um1.x.array[:]
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
    [Res, dResdw] = plasticityProblem.prep_newton(u=u,um1=um1,du=du,ddu=ddu,lam=la, mu=mu) 
    return [Res, dResdw]



atol=(x_max_all-x_min_all)*0.000 # for selection of boundary


def all(x):
        return np.full_like(x[0],True)
    


u_D = dlfx.fem.Function(V) # for dirichlet BCs
def top_displacement():    
    u_y = ufl.conditional(ufl.le(t_global,ufl.as_ufl(1.0)),t_global,ufl.as_ufl(1.0-(t_global-1.0)))
    u_x = ufl.as_ufl(0.0)
    u_z = ufl.as_ufl(0.0)
    return ufl.as_vector([u_x, u_y, u_z]) # 3 components in 3D

bc_top_expression = dlfx.fem.Expression(top_displacement(),V.element.interpolation_points())

boundary_top_bc = bc.get_top_boundary_of_box_as_function(domain,comm,atol=atol*0.0)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_top_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(V, fdim, facets_at_boundary) 

def get_bcs(t):
    
    u_D.interpolate(bc_top_expression)
    bc_top : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(u_D,dofs_at_boundary)
    
    bc_bottom_z = bc.define_dirichlet_bc_from_value(domain,0.0,2,bc.get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)
    bc_bottom_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,bc.get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)
    bc_bottom_x = bc.define_dirichlet_bc_from_value(domain,0.0,0,bc.get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)

    bcs = [bc_top,bc_bottom_z,bc_bottom_y,bc_bottom_x]
    return bcs


n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_top_boundary_of_box_as_function(domain, comm,atol=atol*0.0),top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

Work = dlfx.fem.Constant(domain,0.0)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,20.0)
TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
def after_timestep_success(t,dt,iters):
    
    delta_u = u - um1  
    H_expr = plasticityProblem.update_H(u,delta_u=delta_u,lam=la,mu=mu)
    H.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,H_expr)
    
    
    alex.plasticity.update_e_p_n_and_alpha_arrays(u,e_p_11_n_tmp,e_p_22_n_tmp,e_p_12_n_tmp,e_p_33_n_tmp,e_p_13_n_tmp,e_p_23_n_tmp,
                           e_p_11_n,e_p_22_n,e_p_12_n,e_p_33_n,e_p_13_n,e_p_23_n,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu)
    
    
    # update u from Δu
    
    sigma = plasticityProblem.sigma(u,la,mu)
    tensor_field_expression = dlfx.fem.Expression(sigma, 
                                                         TEN.element.interpolation_points())
    tensor_field_name = "sigma"
    sigma_interpolated = dlfx.fem.Function(TEN) 
    sigma_interpolated.interpolate(tensor_field_expression)
    sigma_interpolated.name = tensor_field_name
    
    #pp.write_tensor_fields(domain,comm,[sigma],["sigma"],outputfile_xdmf_path,t)
    Rx_top, Ry_top, Rz_top = pp.reaction_force(sigma_interpolated,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    

    dW = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    
    E_el = plasticityProblem.get_E_el_global(u,la,mu,dx=ufl.dx,comm=comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    
    if rank == 0:
        if (t>1):
            u_y = 1.0-(t-1.0)
        else:
            u_y = t
        pp.write_to_graphs_output_file(outputfile_graph_path,t,  Ry_top,u_y)


    # update
    um1.x.array[:] = u.x.array[:]
    urestart.x.array[:] = u.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    
    pp.write_vector_fields(domain,comm,[u],["u"],outputfile_xdmf_path,t)
    pp.write_tensor_fields(domain,comm,[sigma_interpolated],["sigma"],outputfile_xdmf_path,t)

def after_timestep_restart(t,dt,iters):
    u.x.array[:] = urestart.x.array[:]

def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=[ "R_y", "u_y"])
        

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

