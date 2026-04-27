import dolfinx as dlfx
import os
import numpy as np
import ufl
import dolfinx.fem as fem
import basix

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol

import alex.plasticity

Ry_scaling = 1

data_path = '/home/resources/Nanomesh.xdmf'
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


N = 10
# import or create geometry
domain = dlfx.io.XDMFFile(comm, data_path, 'r').read_mesh()
#domain = dlfx.mesh.create_unit_cube(comm,N,N,N,dlfx.mesh.CellType.tetrahedron) #hexahedron or tetrahedron
deg_quad = 1  # quadrature degree for internal state variable representation


def mesh_box_select(x_range,y_range,z_range,domain,dim):
    """Select a smaller subset of a mesh to reduce simulation times
    WARNING: This will result in jagged boundaries, an appropriate tolerance should be implemented for boundary conditions

    Args:
        x_range, y_range, z_range (tuple of int): Tuple of upper and lower bound on respective axis
        domain: Full domain on which a subset is to be selected
        dim: Topological dimension of the mesh entities to consider.
        
    Returns: 
        Subdomain within selected box

    """

    # define a smaller bounding box to simulate
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    def bounding_box_marker(x):
        is_in_x = np.logical_and(x[0] >= x_min, x[0] <= x_max)
        is_in_y = np.logical_and(x[1] >= y_min, x[1] <= y_max)
        is_in_z = np.logical_and(x[2] >= z_min, x[2] <= z_max)
        return np.logical_and(is_in_x, np.logical_and(is_in_y, is_in_z))
    
    cells_in_subset = dlfx.mesh.locate_entities(domain, dim, bounding_box_marker)

    marker_value = 1
    values = np.full(len(cells_in_subset), marker_value, dtype=np.int32)

    # Create the MeshTags object
    cell_indices = np.arange(domain.topology.index_map(dim).size_local, dtype=np.int32)

    # Create a boolean array where True indicates cells in the bounding box
    marked_cells = np.zeros_like(cell_indices, dtype=bool)
    marked_cells[cells_in_subset] = True

    # Create MeshTags
    cell_tags = dlfx.mesh.meshtags(domain, dim, cells_in_subset, values)

    # Generate a submesh based on the bounding box, and select it as the domain
    sub_domain, entity_map, vertex_map, geom_map = dlfx.mesh.create_submesh(domain, dim, cell_tags.find(marker_value))

    return sub_domain

dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)


x_range = (x_min_all,x_max_all/2)
y_range = (y_min_all,y_max_all/2)
z_range = (z_min_all,z_max_all/2)
#mesh = mesh_box_select(x_range,y_range,z_range,domain,dim)

# Simulation parameters ####
dt_start = 0.01 
dt_max_in_critical_area = dt_start
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
Tend = 2.5 #3.0
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

gdim = 3

# function space for 3d fields
#Ve_3d = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=2, dim=9, quad_scheme="default")
Ve_3d = ufl.TensorElement("Quadrature", domain.ufl_cell(), degree=deg_quad, shape=(3,3), quad_scheme="default") # Falsche definition, sollte von Quadratur Elementen abhängen
V_3d = dlfx.fem.FunctionSpace(domain, Ve_3d)
# Set e_p and e_p_n up in a TensorFunctionSpace
e_p_n = fem.Function(V_3d, name='e_p')
e_p_n_tmp = fem.Function(V_3d, name='e_p_tmp')
e_p_n.x.array[:] = 0.0
e_p_n_tmp.x.array[:] = 0.0

H,alpha_n,alpha_tmp,_,_,_,_,_,_,_,_ = alex.plasticity.define_internal_state_variables_basix_b(gdim, domain, deg_quad,quad_scheme="default")

dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)
H.x.array[:] = np.zeros_like(H.x.array[:])
alpha_n.x.array[:] = np.zeros_like(alpha_n.x.array[:])
alpha_tmp.x.array[:] = np.zeros_like(alpha_tmp.x.array[:])


## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

plasticityProblem = alex.plasticity.Plasticity_incremental_3D(sig_y=sig_y.value, hard=hard.value,alpha_n=alpha_n,e_p_n=e_p_n,H=H)

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
    [Res, dResdw] = plasticityProblem.prep_newton(u=u,um1=um1,du=du,ddu=ddu,lam=la,mu=mu) 
    return [Res, dResdw]



atol=(x_max_all-x_min_all)*0.05 # for selection of boundary


def all(x):
        return np.full_like(x[0],True)
    

def get_bcs(t):
    
    if t<= 1: 
        u_y_val = t/2
    else: 
        u_y_val = (1 - (t-1))/2

    bc_bottom_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,bc.get_bottom_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)
    bc_top_y = bc.define_dirichlet_bc_from_value(domain,u_y_val,1,bc.get_top_boundary_of_box_as_function(domain,comm,atol=atol),V,-1)
    bc_bottom_corner_x = bc.define_dirichlet_bc_from_value(domain,0.0,0,bc.get_corner_of_box_as_function(domain,comm),V,-1)
    bc_bottom_corner_z = bc.define_dirichlet_bc_from_value(domain,0.0,2,bc.get_corner_of_box_as_function(domain,comm),V,-1)
    
    bcs = [bc_top_y,bc_bottom_y,bc_bottom_corner_x,bc_bottom_corner_z]
    return bcs


n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol),external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain,bc.get_top_boundary_of_box_as_function(domain, comm,atol=atol),top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags,metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

Work = dlfx.fem.Constant(domain,0.0)

success_timestep_counter = dlfx.fem.Constant(domain,0.0)
postprocessing_interval = dlfx.fem.Constant(domain,20.0)

TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
S0e = basix.ufl.element("DP", domain.basix_cell(), 0, shape=())
S0 = dlfx.fem.functionspace(domain, S0e)

def after_timestep_success(t,dt,iters):
    
    delta_u = u - um1  
    H_expr = plasticityProblem.update_H(u,delta_u=delta_u,lam=la,mu=mu)
    H.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,H_expr)
    
    
    alex.plasticity.update_e_p_n_and_alpha_arrays_tensorial(u,e_p_n,e_p_n_tmp,
                           alpha_tmp,alpha_n,domain,cells,quadrature_points,sig_y,hard,mu)

    sigma = plasticityProblem.sigma(u,la,mu,mode='3d')

    problem = alex.plasticity.linear_problem(TEN,dx,sigma,deg_quad)

    sigma_interpolated = problem.solve()
    sigma_interpolated.name = "sigma"
    
    #pp.write_tensor_fields(domain,comm,[sigma],["sigma"],outputfile_xdmf_path,t)
    _, Ry_top, _ = pp.reaction_force(sigma_interpolated,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    

    dW = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds,comm=comm)
    Work.value = Work.value + dW
    
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    
    if rank == 0:
        if t<= 1: 
            u_y = t/2
        else: 
            u_y = (1 - (t-1))/2
        # ----------------------------------------------------------------------------------------------------------------------------------------
        pp.write_to_graphs_output_file(outputfile_graph_path,t, Ry_top*Ry_scaling,u_y)     # Arbitrary scaling factor introduced!!!!
        # ----------------------------------------------------------------------------------------------------------------------------------------


    # update
    um1.x.array[:] = u.x.array[:]
    urestart.x.array[:] = u.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    
    pp.write_vector_fields(domain,comm,[u],["u"],outputfile_xdmf_path,t)
    #pp.write_field(domain,outputfile_xdmf_path,alpha_n,t,comm,S=S0)
    #pp.write_tensor_fields(domain,comm,[e_p_n_interpolated],["e_p_n"],outputfile_xdmf_path,t)
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