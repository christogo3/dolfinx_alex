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

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
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
nx = 32
ny = 16 
L=1.0

# generate domain
# domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_rectangle(comm,[np.array([0.0, 0.0]), np.array([L, L])], [nx,ny], cell_type=dlfx.mesh.CellType.quadrilateral)

dt = dlfx.fem.Constant(domain, 0.05)
Tend = 3.0
gamma = 0.5 #Newmark parameters
beta = 0.25

# elastic constants
lam = dlfx.fem.Constant(domain, 1.0)
mu = dlfx.fem.Constant(domain, 1.0)
rho0 = dlfx.fem.Constant(domain,1.0)
E_mod = alex.linearelastic.get_emod(lam.value, mu.value)

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
      
hyperElasticProblem = alex.hyperelastic.ElasticProblem()

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

def leftAndRight(x):
    return np.logical_or(np.isclose(x[0],x_min_all,rtol=0.01),np.isclose(x[0],x_max_all,rtol=0.01))

def right_bottom(x):
    return np.logical_and(np.isclose(x[1],y_min_all,rtol=0.01),np.isclose(x[0],x_max_all,rtol=0.01))

n = ufl.FacetNormal(domain)
surface_with_traction_tag = 1
tags = pp.tag_part_of_boundary(domain,
                               leftAndRight,surface_with_traction_tag)
ds_left_right_tagged = ufl.Measure('ds', domain=domain, subdomain_data=tags)



def get_bcs(t):
    # facets_left = dlfx.mesh.locate_entities_boundary(domain, fdim, left)
    # dofs_left_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim,facets_left)
    # dofs_left_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim,facets_left)
    # dofs_left_z = dlfx.fem.locate_dofs_topological(V.sub(2),fdim,facets_left)
    # bc_left_x = dlfx.fem.dirichletbc(-0.1,dofs_left_x,V.sub(0))
    # bc_left_y = dlfx.fem.dirichletbc(0.0,dofs_left_y,V.sub(1))
    # bc_left_z = dlfx.fem.dirichletbc(0.0,dofs_left_z,V.sub(2))
    
    # facets_right = dlfx.mesh.locate_entities_boundary(domain, fdim, right)
    # dofs_right_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim,facets_right)
    # dofs_right_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim,facets_right)
    # dofs_right_z = dlfx.fem.locate_dofs_topological(V.sub(2),fdim,facets_right)
    # bc_right_x = dlfx.fem.dirichletbc(0.1,dofs_right_x,V.sub(0))
    # bc_right_y = dlfx.fem.dirichletbc(0.0,dofs_right_y,V.sub(1))
    # bc_right_z = dlfx.fem.dirichletbc(0.0,dofs_right_z,V.sub(2))
    
    
    # bcs = [bc_left_x, bc_left_y, bc_left_z,  bc_right_x, bc_right_y, bc_right_z,]
    
    vertices_at_corner = dlfx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
    dofs_at_corner_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner)
    bc_corner_x = dlfx.fem.dirichletbc(0.0,dofs_at_corner_x,V.sub(0))
    dofs_at_corner_y = dlfx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner)
    bc_corner_y = dlfx.fem.dirichletbc(0.0,dofs_at_corner_y,V.sub(1))
    # dofs_at_corner_z = dlfx.fem.locate_dofs_topological(V.sub(2),fdim-1,vertices_at_corner)
    # bc_corner_z = dlfx.fem.dirichletbc(0.0,dofs_at_corner_z,V.sub(2))
    bcs = [bc_corner_x,  bc_corner_y]
    bcs = []
    
    characteristic_time = L*math.sqrt(rho0.value/mu.value)
    characteristic_load = mu.value * L / L
    rate = 1.0 * characteristic_load / characteristic_time
    if t/characteristic_time <= 1.0:
        P0 = dlfx.fem.Constant(domain, np.array([[rate*t, 0.0],
                    [0.0, 0.0]]))
    else:
        P0 = dlfx.fem.Constant(domain, np.array([[1.0 * characteristic_load, 0.0],
                    [0.0, 0.0]]))
    
    hyperElasticProblem.set_traction_bc(P0=P0,u=u,N=n,ds=ds_left_right_tagged(surface_with_traction_tag))
    
    return bcs


def after_timestep_success(t,dt,iters):
    u.name = "u"
    pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
        # displacement at ends of bar
        vertices_at_corner = dlfx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
        dofs_at_corner_x = dlfx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner)
        
        vertices_at_corner_right = dlfx.mesh.locate_entities(domain,fdim-1,right_bottom)
        dofs_at_corner_x_right = dlfx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner_right)
        u.x.scatter_forward()
        u_left = u.x.array[dofs_at_corner_x][0]
        u_right = u.x.array[dofs_at_corner_x_right][0]
        print(f"u: left:{u_left} right:{u_right}")
        pp.write_to_graphs_output_file(outputfile_graph_path,t, u_left, u_right)
        
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