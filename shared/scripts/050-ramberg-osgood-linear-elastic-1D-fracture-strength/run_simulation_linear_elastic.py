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
    
N=200
element_size = 2.0 / 200.0
domain = dlfx.mesh.create_rectangle(comm,[np.array([-1.0, 0.0]), np.array([1.0, 1.0])], [N,1], cell_type=dlfx.mesh.CellType.triangle) # Important to use traingle elements, else display wont work
    
dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)
    
x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)




# Simulation parameters ####
dt_start = 0.0001
dt_max_in_critical_area = 2.0e-7
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
# Tend = 10.0 * dt_global.value
dt_global.value = dt_max_in_critical_area
dt_max = dlfx.fem.Constant(domain,50*dt_start)
Tend = 2.0


la = dlfx.fem.Constant(domain, 0.0)
mu = dlfx.fem.Constant(domain, 1.0)


#gc = dlfx.fem.Constant(domain, 1.0)



eta = dlfx.fem.Constant(domain, 0.00001)
epsilon = dlfx.fem.Constant(domain, 0.1)
Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

# Function space and FE functions ########################################################
Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,)) #displacements
Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())# fracture fields
W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Se]))


SS = dlfx.fem.functionspace(domain, Se)
x = ufl.SpatialCoordinate(domain)

gc_val = 0.5679
gc_expr = dlfx.fem.Expression(ufl.conditional(ufl.Or(ufl.le(x[0],-3.0*element_size),
                            ufl.ge(x[0],3.0*element_size)),gc_val,0.99*gc_val),SS.element.interpolation_points())

gc = dlfx.fem.Function(SS)
gc.interpolate(gc_expr)

#gc = dlfx.fem.Constant(domain, 1.0)

# gc_expr = ufl.conditional(ufl.ge(1.0,2.0),0.0,1.0)

# gc_f = dlfx.fem.Function(W)
# gc_f.interpolate(gc_expr,W.sub(1).element.interpolation_points())
# _, gc = gc_f.split()

# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
wm1.x.array[:] = np.zeros_like(wm1.x.array[:])
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

deg_quad = 1  # quadrature degree for internal state variable representation
gdim = 2
H = alex.plasticity.define_internal_state_variables_basix(gdim, domain, deg_quad,quad_scheme="default")
dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)
H.x.array[:] = np.zeros_like(H.x.array[:])


## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


phaseFieldProblem = pf.StaticPhaseFieldProblem2D(degradationFunction=pf.degrad_cubic,
                                                    psisurf=pf.psisurf_from_function)


# phaseFieldProblem = pf.StaticPhaseFieldProblem2D_incremental(degradationFunction=pf.degrad_cubic,
#                                                    psisurf=pf.psisurf_from_function,dx=dx, yield_stress_1d=1.0, b_hardening_parameter=0.1, r_transition_smoothness_parameter=10.0,H=H)

timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    
    w.sub(1).x.array[:] = np.ones_like(w.sub(1).x.array[:])
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
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
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=la, mu = mu,
        Gc=gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]

# setup tracking
S = dlfx.fem.functionspace(domain,Se)
s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
s_zero_for_tracking_at_nodes.interpolate(sub_expr)

atol=(x_max_all-x_min_all)*0.000 # for selection of boundary

# surfing BCs
# xtip = np.array([0.0,0.0,0.0],dtype=dlfx.default_scalar_type)
# xK1 = dlfx.fem.Constant(domain, xtip)
# v_crack = 1.2*(x_max_all-crack_tip_start_location_x)/Tend
# vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack,0.0,0.0],dtype=dlfx.default_scalar_type))
# crack_start = dlfx.fem.Constant(domain, np.array([0.0,crack_tip_start_location_y,0.0],dtype=dlfx.default_scalar_type))

[Res, dResdw] = get_residuum_and_gateaux(delta_t=dt_global)
w_D_left = dlfx.fem.Function(W) # for dirichlet BCs

# front_back = bc.get_frontback_boundary_of_box_as_function(domain,comm,atol=0.1*atol)
# bc_front_back = bc.define_dirichlet_bc_from_value(domain,0.0,2,front_back,W,0)

# def compute_surf_displacement():
#     x = ufl.SpatialCoordinate(domain)
#     xxK1 = crack_start + vcrack_const * t_global 
#     dx = x[0] - xxK1[0]
#     dy = x[1] - xxK1[1]
    
#     nu = alex.linearelastic.get_nu(lam=la, mu=mu) # should be effective values?
#     r = ufl.sqrt(ufl.inner(dx,dx) + ufl.inner(dy,dy))
#     theta = ufl.atan2(dy, dx)
    
#     u_x = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.cos(0.5 * theta)
#     u_y = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.sin(0.5 * theta)
#     u_z = ufl.as_ufl(0.0)
#     return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

# bc_expression = dlfx.fem.Expression(compute_surf_displacement(),W.sub(0).element.interpolation_points())
# # boundary_surfing_bc = bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol*0.0) #bc.get_boundary_for_surfing_boundary_condition_2D(domain,comm,atol=atol,epsilon=epsilon.value) #bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol)
# boundary_surfing_bc = bc.get_2D_boundary_of_box_as_function(domain,comm,atol=atol*0.0,epsilon=epsilon.value)
# facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
# dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary) 


def all(x):
        return np.full_like(x[0],True)
    
allfacets = dlfx.mesh.locate_entities(domain, fdim, all)
all_ydofs = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, allfacets)
bc_y_all = dlfx.fem.dirichletbc(0.0, all_ydofs, W.sub(0).sub(1))



w_D_left = dlfx.fem.Function(W) # for dirichlet BCs
def left_displacement():    
    u_y = ufl.as_ufl(0.0)
    u_x = ufl.as_ufl(-t_global)
    return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

bc_left_expression = dlfx.fem.Expression(left_displacement(),W.sub(0).element.interpolation_points())
boundary_left_bc = bc.get_left_boundary_of_box_as_function(domain,comm,atol=atol*0.0)
facets_at_boundary_left = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_left_bc)
dofs_at_boundary_left = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary_left) 


w_D_right = dlfx.fem.Function(W) # for dirichlet BCs
def right_displacement():    
    u_y = ufl.as_ufl(0.0)
    u_x = ufl.as_ufl(t_global)
    return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

bc_right_expression = dlfx.fem.Expression(right_displacement(),W.sub(0).element.interpolation_points())
boundary_right_bc = bc.get_right_boundary_of_box_as_function(domain,comm,atol=atol*0.0)
facets_at_boundary_right = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_right_bc)
dofs_at_boundary_right = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary_right) 


def get_bcs(t):
    
    w_D_left.sub(0).interpolate(bc_left_expression)
    bc_left : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D_left,dofs_at_boundary_left)
    
    w_D_right.sub(0).interpolate(bc_right_expression)
    bc_right : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D_right,dofs_at_boundary_right)
    
    bc_x_left = bc.define_dirichlet_bc_from_value(domain,-t,0,bc.get_left_boundary_of_box_as_function(domain,comm,atol=atol),W,0)
    bc_x_right = bc.define_dirichlet_bc_from_value(domain,t,0,bc.get_right_boundary_of_box_as_function(domain,comm,atol=atol),W,0)
    
        # irreversibility
    bcs = [bc_y_all,bc_x_right,bc_x_left]
    #bcs = [bc_y_all,bc_right,bc_left]
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
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

right_surface_tag = 9
right_surface_tags = pp.tag_part_of_boundary(domain,bc.get_right_boundary_of_box_as_function(domain, comm,atol=atol*0.0),right_surface_tag)
ds_right_tagged = ufl.Measure('ds', domain=domain, subdomain_data=right_surface_tags)

Work = dlfx.fem.Constant(domain,0.0)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,10.0)
TEN = dlfx.fem.functionspace(domain, ("DP", deg_quad-1, (dim, dim)))
def after_timestep_success(t,dt,iters):
    # update u from Δu
    
    
    
    
    sigma = phaseFieldProblem.sigma_degraded(u,s,la,mu,eta)
    Rx_right, Ry_top = pp.reaction_force(sigma,n=n,ds=ds_right_tagged(right_surface_tag),comm=comm)
    
    um1, _ = ufl.split(wm1)
    delta_u = u - um1  
    #H_expr = H + ufl.inner(phaseFieldProblem.sigma_undegraded(u=u,lam=la,mu=mu),0.5*(ufl.grad(delta_u) + ufl.grad(delta_u).T))
    # H_expr = phaseFieldProblem.update_H(u,delta_u=delta_u,lam=la,mu=mu)
    # H.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,H_expr)
    

    dW = pp.work_increment_external_forces(sigma,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
    E_el = phaseFieldProblem.get_E_el_global(s,eta,u,la,mu,dx=ufl.dx,comm=comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    eshelby = phaseFieldProblem.getEshelby(w,eta,la,mu)
    #eshelby = phaseFieldProblem.getEshelby(w,eta,la,mu)
    tensor_field_expression = dlfx.fem.Expression(eshelby, 
                                                         TEN.element.interpolation_points())
    tensor_field_name = "eshelby"
    eshelby_interpolated = dlfx.fem.Function(TEN) 
    eshelby_interpolated.interpolate(tensor_field_expression)
    eshelby_interpolated.name = tensor_field_name
    
    
    Jx, Jy = alex.linearelastic.get_J_2D(eshelby_interpolated,n,ds=ds(external_surface_tag),comm=comm)
    # Jx_vol, Jy_vol = alex.linearelastic.get_J_2D_volume_integral(eshelby,ufl.dx,comm)
    
    #alex.os.mpi_print(pp.getJString2D(Jx,Jy),rank)
    

    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking_at_nodes.interpolate(s)
    max_x, max_y, min_x, min_y  = pp.crack_bounding_box_2D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    x_ct = max_x
    
    # only output to graphs file if timestep is correct in measured area
    # if (rank == 0 and in_steg_to_be_measured(x_ct=x_ct) and dt <= dt_max_in_critical_area) or ( rank == 0 and not in_steg_to_be_measured(x_ct=x_ct)):
    if rank == 0:
        print("Crack tip position x: " + str(x_ct))
        pp.write_to_graphs_output_file(outputfile_graph_path,t,Rx_right)

        # pp.write_to_graphs_output_file(outputfile_graph_path,t,Jx, Jy, Jx_vol, Jy_vol, x_ct)

    # if in_steg_to_be_measured(x_ct=x_ct):
    #     if rank == 0:
    #         first_low, first_high, second_low, second_high = steg_bounds_to_be_measured()
    #         print(f"Crack currently progressing in measured area [{first_low},{first_high}] or [{second_low},{second_high}]. dt restricted to max {dt_max_in_critical_area}")
        
    #     # restricting time step    
    #     dt_max.value = dt_max_in_critical_area
    #     dt_global.value = dt_max_in_critical_area
        
    #     # restart if dt is to large
    #     # if (dt > dt_max_in_critical_area): # need to reset time and w in addition to time
    #     #     w.x.array[:] = wrestart.x.array[:]
    #     #     t_global.value = t_global.value - dt
    #     #     #t_global.value = trestart_global.value
             
    # else:
    #     dt_max.value = dt_start # reset to larger time step bound
    
    # update H 
    


    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    

    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    pp.write_scalar_fields(domain,comm,[gc],["gc"],outputfile_xdmf_path,t)

def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]

def after_last_timestep():
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t_global.value, comm)
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Rx"])
        

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    w,
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
