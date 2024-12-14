import alex.hyperelastic
import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
from mpi4py import MPI
import basix

import ufl 
import numpy as np
import os 
import sys

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import math
import argparse
from datetime import datetime
import shutil
import math

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)
parameter_path = os.path.join(script_path,"parameters.txt")

# Define argument parser
parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--lam", type=float, required=False, default=1.0, help="Lambda parameter (default: 1.0)")
    parser.add_argument("--mue", type=float, required=False, default=1.0, help="Mu parameter (default: 1.0)")
    parser.add_argument("--rh0", type=float, required=False, default=1.0, help="Density parameter rho0 (default: 1.0)")
    parser.add_argument("--psi", type=str, required=False, default="SV", help="Elastic potential")
    args = parser.parse_args()

    lam_value = args.lam
    mue_value = args.mue
    rh0_value = args.rh0
    psi_value = args.psi
except (argparse.ArgumentError, SystemExit, Exception) as e:
    # Default fallback values
    lam_value = 1.0
    mue_value = 1.0
    rh0_value = 1.0
    psi_value = "SV"
    print("Using default parameter values due to parsing failure.")
    print(e)

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
nx = 64
ny = 64 
L=1.0

# generate domain
# domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_rectangle(comm,[np.array([-L/2, -L/2]), np.array([L/2, L/2])], [nx,ny], cell_type=dlfx.mesh.CellType.quadrilateral)

dt = dlfx.fem.Constant(domain, 0.05)
Tend = 3.0
gamma = 0.5 #Newmark parameters
beta = 0.25

# elastic constants
lam = dlfx.fem.Constant(domain, lam_value)
mu = dlfx.fem.Constant(domain, mue_value)
rho0 = dlfx.fem.Constant(domain,rh0_value)
E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
nu=alex.linearelastic.get_nu(lam.value,mu.value)

# function space using mesh and degree
Ve = basix.ufl.element("P", domain.basix_cell(), 1 , shape=(domain.geometry.dim,))
V = dlfx.fem.functionspace(domain, Ve)

# define boundary condition on top and bottom
fdim = domain.topology.dim -1

bcs = []
             
# define solution, restart, trial and test space
u =  dlfx.fem.Function(V)
um1 =  dlfx.fem.Function(V)
vm1 =  dlfx.fem.Function(V)
am1 =  dlfx.fem.Function(V)
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


if psi_value == "SV":
    psi_elastic = alex.hyperelastic.psi_Saint_Venant
elif psi_value == "NH":
    psi_elastic = alex.hyperelastic.psi_Neo_Hooke

hyperElasticProblem = alex.hyperelastic.ElasticProblem(psi=psi_elastic)

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    accel, vel = sol.update_newmark(beta=beta,
                                    gamma=gamma,
                                    dt=dt,
                                    u=u,
                                    um1=um1,
                                    vm1=vm1,
                                    am1=am1,
                                    is_ufl=True)
    [Res, dResdw] = hyperElasticProblem.prep_newton(u,du,ddu,lam,mu,rho0=rho0,accel=accel)
    return [Res, dResdw]


# eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
#                     [0.0, 0.6, 0.0],
#                     [0.0, 0.0, 0.0]]))

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
def left(x):
    return np.isclose(x[0],x_min_all)
    
def right(x):
    return np.isclose(x[0],x_max_all)

def bottom(x):
    return np.isclose(x[1],y_min_all)
    
def top(x):
    return np.isclose(x[1],y_max_all)

def topAndBottom(x):
    return np.logical_or(np.isclose(x[1],y_min_all,rtol=0.01),
                         np.isclose(x[1],y_max_all,rtol=0.01))


def leftAndRight(x):
    return np.logical_or(np.isclose(x[0],x_min_all,rtol=0.01),np.isclose(x[0],x_max_all,rtol=0.01))

def right_bottom(x):
    return np.logical_and(np.isclose(x[1],y_min_all,rtol=0.01),np.isclose(x[0],x_max_all,rtol=0.01))

def top_left(x):
    return np.logical_and(np.isclose(x[1],y_max_all,rtol=0.01),np.isclose(x[0],x_min_all,rtol=0.01))


n = ufl.FacetNormal(domain)
surface_with_traction_tag = 1
tags = pp.tag_part_of_boundary(domain,
                               topAndBottom,surface_with_traction_tag)
ds_top_bottom_tagged = ufl.Measure('ds', domain=domain, subdomain_data=tags)


def get_bcs(t):
    
    # vertices_at_corner = dlfx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
    # dofs_at_corner_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner)
    # bc_corner_x = dlfx.fem.dirichletbc(0.0,dofs_at_corner_x,V.sub(0))
    # dofs_at_corner_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner)
    # bc_corner_y = dlfx.fem.dirichletbc(0.0,dofs_at_corner_y,V.sub(1))
    # bcs = [bc_corner_x,  bc_corner_y]
    # bcs = []
    
    bc_bottom_x = bc.define_dirichlet_bc_from_value(domain,0.0,0,bottom,V,-1)
    bc_bottom_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,bottom,V,-1)
    
    if psi_value == "SV":
        A = 0.03
    elif psi_value == "NH":
        A = 0.1
        
    if t <= 2.0:
        ux_t = 2.0 / math.pi * A * (1.0 - math.cos(math.pi/2.0 * t))
    else:
        ux_t = 2.0 / math.pi * A * 2.0
   
    bc_top_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,top,V,-1)
    bc_top_x = bc.define_dirichlet_bc_from_value(domain,ux_t,0,top,V,-1)
    
    bcs = [bc_bottom_x, bc_bottom_y, bc_top_x, bc_top_y]
    
    # characteristic_time = L*math.sqrt(rho0.value/mu.value)
    # characteristic_load = mu.value * L / L
    # rate = 1.0 * characteristic_load / characteristic_time
    # if t <= 2.0:
    #     P0 = dlfx.fem.Constant(domain, np.array([[0.0, 0.0],
    #                 [0.0, 0.175*math.sin(math.pi / 4.0 * t) * math.sin(math.pi / 4.0 * t)]]))
    # else:
    #     P0 = dlfx.fem.Constant(domain, np.array([[0.0, 0.0],
    #                 [0.0, 0.175]]))
    
    # hyperElasticProblem.set_traction_bc(P0=P0,u=u,N=n,ds=ds_top_bottom_tagged(surface_with_traction_tag))
    
    return bcs


def after_timestep_success(t,dt,iters):
    u.name = "u"
    pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
        # displacement at ends of bar
        vertices_at_corner = dlfx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
        dofs_at_bottom_left_corner_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner)
        
        vertices_at_corner_top = dlfx.mesh.locate_entities(domain,fdim-1,top_left)
        dofs_at_corner_y_top = dlfx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner_top)
        u.x.scatter_forward()
        u_bottom = u.x.array[dofs_at_bottom_left_corner_y][0]
        u_top = u.x.array[dofs_at_corner_y_top][0]
        print(f"u: bottom:{u_bottom} top:{u_top}")
        pp.write_to_graphs_output_file(outputfile_graph_path,t, u_bottom, u_top)
        
    urestart.x.array[:] = u.x.array[:] 
    accel, vel = sol.update_newmark(beta=beta,
                                    gamma=gamma,
                                    dt=dt,
                                    u=u,
                                    um1=um1,
                                    vm1=vm1,
                                    am1=am1,
                                    is_ufl=False)
    um1.x.array[:] = u.x.array[:]
    vm1.x.array[:] = vel[:]
    am1.x.array[:] = accel[:]
    
             
    
    
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
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["u_bottom_left", "u_bottom_right"])

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
    print_bool=True,
    dt_never_scale_up=True
)

parameters_to_write = {
        'lam': lam.value,
        'mue': mu.value,
        'rh0': rho0.value,
        'psi': psi_value,
        'nu': nu,
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
    pp.write_to_file(parameters=parameters_to_write,filename=parameter_path,comm=comm)
    
    files_to_copy = [
        parameter_path,
        outputfile_graph_path,
        #mesh_file,  # Add more files as needed
        os.path.join(script_path,"graphs.png"),
        os.path.join(script_path,script_name_without_extension+".py"),
        os.path.join(script_path,script_name_without_extension+".xdmf"),
        os.path.join(script_path,script_name_without_extension+".h5")
    ]
        
    # Create the directory
    target_directory = create_timestamped_directory(base_dir=script_path)
    print(f"Created directory: {target_directory}")

    # Copy the files
    copy_files_to_directory(files_to_copy, target_directory)
    print("Files copied successfully.")