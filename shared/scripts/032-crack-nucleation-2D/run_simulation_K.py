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

# Define argument parser
parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--mesh_file", type=str, required=True, help="Name of the mesh file")
    parser.add_argument("--lam_param", type=float, required=True, help="Lambda parameter")
    parser.add_argument("--mue_param", type=float, required=True, help="Mu parameter")
    parser.add_argument("--gc_param", type=float, required=True, help="Gc parameter")
    parser.add_argument("--eps_param", type=float, required=True, help="Epsilon factor parameter")
    parser.add_argument("--element_order", type=int, required=True, help="Element order")
    args = parser.parse_args()
    mesh_file = args.mesh_file
    la_param = args.lam_param
    mu_param = args.mue_param
    gc_param = args.gc_param
    eps_param = args.eps_param
except (argparse.ArgumentError, SystemExit, Exception) as e:
    if rank == 0:
        print("Could not parse arguments")
        print(e)
    la_param = 1.0
    mu_param = 1.0
    gc_param = 1.0
    mesh_file = "mesh_adaptive.xdmf"
    eps_param = 0.08
    

with dlfx.io.XDMFFile(comm, os.path.join(script_path,mesh_file), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh(name="Grid")
    
dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)
    
x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)


# Material definition ##################################################



# Simulation parameters ####
dt_start = 0.001
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
# Tend = 10.0


la = dlfx.fem.Constant(domain,la_param) 
mu = dlfx.fem.Constant(domain, mu_param)
gc = dlfx.fem.Constant(domain,gc_param)

eta = dlfx.fem.Constant(domain, 0.00001)
epsilon = dlfx.fem.Constant(domain, eps_param)


# Function space and FE functions ########################################################
# Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1,dim=2) # displacements
# Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) # fracture fields
# W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))
Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,)) #displacements
Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())# fracture fields
W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Se]))

# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

# setting K1 so it always breaks
E_mod = alex.linearelastic.get_emod(lam=la.value,mu=mu.value) # TODO should be effective elastic parameters
# epsilon0 = dlfx.fem.Constant(domain, (y_max_all-y_min_all) / 50.0)
hh = 0.02 # TODO change
Gc_num = (1.0 + hh / epsilon.value ) * gc.value
K1 = dlfx.fem.Constant(domain, 1.5 * math.sqrt(Gc_num * E_mod))



# define crack by boundary
dhole = 1.0
crack_tip_start_location_x = 0.5*dhole # center of hole
crack_tip_start_location_y = 0.0 #(y_max_all + y_min_all) / 2.0
def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1],crack_tip_start_location_y,atol=0.5*epsilon.value)
    return np.logical_and(y_log,x_log)

v_crack = 1.0 # const for all simulations
Tend = (x_max_all-crack_tip_start_location_x) * 1.2 / v_crack

Mob = dlfx.fem.Constant(domain, 100.0 * v_crack / (gc.value)) # Mobility changed
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value) # CK Paper

## define boundary conditions crack



tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

phaseFieldProblem = pf.StaticPhaseFieldProblem2D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf_from_function)

timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    
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
xtip = np.array([0.0,0.0,0.0],dtype=dlfx.default_scalar_type)

vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack,0.0,0.0],dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([crack_tip_start_location_x,crack_tip_start_location_y,0.0],dtype=dlfx.default_scalar_type))

w_D = dlfx.fem.Function(W) # for dirichlet BCs

# front_back = bc.get_frontback_boundary_of_box_as_function(domain,comm,atol=0.1*atol)
# bc_front_back = bc.define_dirichlet_bc_from_value(domain,0.0,2,front_back,W,0)

def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    xxK1 = crack_start + vcrack_const * t_global 
    dx = x[0] - xxK1[0]
    dy = x[1] - xxK1[1]
    
    nu = alex.linearelastic.get_nu(lam=la.value, mu=mu.value) # should be effective values?
    r = ufl.sqrt(ufl.inner(dx,dx) + ufl.inner(dy,dy))
    theta = ufl.atan2(dy, dx)
    
    u_x = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu * math.sqrt(2.0 * math.pi))  * ufl.sqrt(r) * (3.0 - 4.0 * nu - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    u_z = ufl.as_ufl(0.0)
    return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

bc_expression = dlfx.fem.Expression(compute_surf_displacement(),W.sub(0).element.interpolation_points())
boundary_surfing_bc = bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol*0.0) #bc.get_boundary_for_surfing_boundary_condition_2D(domain,comm,atol=atol,epsilon=epsilon.value) #bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary) 



def get_bcs(t):
    
    bcs = []
    xtip[0] = crack_tip_start_location_x + v_crack * t
    xtip[1] = crack_tip_start_location_y
    w_D.sub(0).interpolate(bc_expression)
    bc_surf : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D,dofs_at_boundary)

    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    bcs.append(bc_surf)
    bcs.append(bccrack)
    

    return bcs


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
postprocessing_interval = dlfx.fem.Constant(domain,1.0)
def after_timestep_success(t,dt,iters):
    sigma = phaseFieldProblem.sigma_degraded(u,s,la,mu,eta)
    Rx_top, Ry_top = pp.reaction_force(sigma,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    
    um1, _ = ufl.split(wm1)

    dW = pp.work_increment_external_forces(sigma,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    eshelby = phaseFieldProblem.getEshelby(w,eta,la,mu)
    Jx, Jy = alex.linearelastic.get_J_2D(eshelby,n,ds=ds(external_surface_tag),comm=comm)
    # Jx_vol, Jy_vol = alex.linearelastic.get_J_2D_volume_integral(eshelby,ufl.dx,comm)
    
    alex.os.mpi_print(pp.getJString2D(Jx,Jy),rank)
    

    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking_at_nodes.interpolate(s)
    max_x, max_y, min_x, min_y  = pp.crack_bounding_box_2D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    if abs(max_x) <= x_max_all:
        x_ct = max_x
    else:
        dhole = 1.0
        x_ct = dhole/2
    
    if rank == 0:
        pp.write_to_graphs_output_file(outputfile_graph_path,t, Jx, Jy,x_ct,xtip[0], Rx_top, Ry_top, dW, Work.value, A, dt)

    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    

    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)

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
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy","x_pf_crack","x_macr","Rx", "Ry", "dW", "W", "A", "dt"])
        

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
    trestart=trestart_global,
)

parameters_to_write = {
        'mesh_file': mesh_file,
        'lam_simulation': la.value,
        'mue_simulation': mu.value,
        'Gc_simulation': gc.value,
        'eps_simulation': eps_param,
        'eps': epsilon.value,
        'eta': eta.value,
        'mob': Mob.value,
        'element_order': 1,
    }



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

if rank == 0:
    pp.append_to_file(parameters=parameters_to_write,filename=parameter_path,comm=comm)
    
    files_to_copy = [
        parameter_path,
        outputfile_graph_path,
        #mesh_file,  # Add more files as needed
        os.path.join(script_path,"graphs.png"),
        os.path.join(script_path,"run_simulation_K.py"),
        os.path.join(script_path,script_name_without_extension+".xdmf"),
        os.path.join(script_path,script_name_without_extension+".h5")
    ]
        
    # Create the directory
    target_directory = create_timestamped_directory(base_dir=script_path)
    print(f"Created directory: {target_directory}")

    # Copy the files
    copy_files_to_directory(files_to_copy, target_directory)
    print("Files copied successfully.")
