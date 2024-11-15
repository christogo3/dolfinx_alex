import basix.ufl
import alex.homogenization
import alex.linearelastic
import alex.phasefield
import alex.postprocessing
import alex.util
import dolfinx as dlfx
from mpi4py import MPI
import basix
import matplotlib.pyplot as plt


import ufl 
import numpy as np
import os 
import sys
from petsc4py import PETSc as petsc

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
import alex.phasefield as pf
import pandas as pd
from scipy.interpolate import griddata

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)


# Load node data
def load_data(file_path):
    nodes_df = pd.read_csv(file_path)
    return nodes_df

def infer_mesh_dimensions_from_nodes(nodes_df):
    unique_y_coords = nodes_df['Points_1'].unique()
    unique_x_coords = nodes_df['Points_0'].unique()

    # Sort unique coordinates in ascending order
    unique_y_coords.sort()
    unique_x_coords.sort()

    # Calculate mesh dimensions (cells = nodes - 1 along each dimension)
    num_rows = len(unique_y_coords) - 1  # Number of cells along the y dimension
    num_cols = len(unique_x_coords) - 1  # Number of cells along the x dimension
    return num_rows, num_cols


# Create 2D array of cell IDs based on inferred mesh dimensions
def arrange_cells_2D(connectivity_df, mesh_dims):
    cell_grid = np.zeros(mesh_dims, dtype=int)  # 2D array to store cell IDs

    # Fill the cell_grid row by row based on Cell ID order in the file
    for index, row in connectivity_df.iterrows():
        cell_id = row['Cell ID']
        row_idx = index // mesh_dims[1]
        col_idx = index % mesh_dims[1]
        cell_grid[row_idx, col_idx] = cell_id

    return cell_grid


# Create a 2D array for density values based on cell_id_grid
def map_E_to_grid(cell_id_grid, cell_data_df):
    # Create an empty array for densities with the same shape as cell_id_grid
    E_Grid = np.full(cell_id_grid.shape, np.nan)  # Using NaN for any missing values

    # Access density column from cell_data_df
    E_from_from_data_frame = cell_data_df['E-Modul'].values

    # Populate the E-Modul grid based on cell_id_grid
    for row in range(cell_id_grid.shape[0]):
        for col in range(cell_id_grid.shape[1]):
            cell_id = cell_id_grid[row, col]
            if cell_id < len(E_from_from_data_frame):
                E_Grid[row, col] = E_from_from_data_frame[cell_id]
            else:
                E_Grid[row, col] = np.nan  # Handle cases where cell_id exceeds density data
    return E_Grid


# Calculate the size of each square element (distance between adjacent nodes)
def calculate_element_size(nodes_df):
    # Get the x and y coordinates of the first two nodes (assuming they are adjacent)
    x1, y1 = nodes_df.iloc[0]['Points_0'], nodes_df.iloc[0]['Points_1']
    x2, y2 = nodes_df.iloc[1]['Points_0'], nodes_df.iloc[1]['Points_1']
    
    # Compute the Euclidean distance between the two nodes
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Paths to the text files
folder_path = os.path.join(script_path,"mbb")
node_file = os.path.join(folder_path,'node_coord.csv')
point_data_file = os.path.join(folder_path,'point_data.csv')
connectivity_file = os.path.join(folder_path,'connectivity.csv')
cell_data_file = os.path.join(folder_path,'cell_data.csv') 

# Load data
nodes_df = load_data(node_file)
point_data_df = load_data(point_data_file)
cell_data_df = load_data(cell_data_file)
connectivity_df = load_data(connectivity_file)

# Infer mesh dimensions
mesh_dims = infer_mesh_dimensions_from_nodes(nodes_df)

# Arrange cell IDs in a 2D array
cell_id_grid = arrange_cells_2D(connectivity_df, mesh_dims)

# Map density values to a second grid
E_grid = map_E_to_grid(cell_id_grid, cell_data_df)
E_max = np.max(E_grid)
E_min = np.min(E_grid)

plt.figure(figsize=(10, 8))
plt.imshow(E_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='E')
plt.title('E Distribution')

# Save the plot to a PNG file
output_image_path = os.path.join(script_path,'E_distribution.png') 
plt.savefig(output_image_path, dpi=300)



# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

with dlfx.io.XDMFFile(comm, os.path.join(script_path,'dlfx_mesh.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh()
    
#domain = dlfx.mesh.create_rectangle(comm, [[0,0],[2,1]], [100, 50], cell_type=dlfx.mesh.CellType.quadrilateral)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)
    
points_x = nodes_df[['Points_0']].values
points_y = nodes_df[['Points_1']].values

# function space using mesh and degree
Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,)) #displacements
Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())# fracture fields
W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Se]))
# Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
# V = dlfx.fem.FunctionSpace(domain, Ve)
# Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
# Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
S = dlfx.fem.FunctionSpace(domain, Se)

E = dlfx.fem.Function(S)
nu = dlfx.fem.Constant(domain=domain,c=0.3)


xx  = np.array(domain.geometry.x).T
xx = xx[0:2]

def interpolate_pixel_data(data, element_size, x_coords, y_coords, method='linear'):
    """
    Interpolate pixel data for given coordinates using scipy's griddata with extrapolation.
    
    Parameters:
    - data (2D array): The pixel data on a grid (numpy array).
    - element_size (float): The size of each element in the grid.
    - x_coords (list of float): The x-coordinates where interpolation is needed.
    - y_coords (list of float): The y-coordinates where interpolation is needed.
    - method (str): Interpolation method ('linear', 'nearest', 'cubic'). Defaults to 'linear'.
    
    Returns:
    - interpolated_values (array): The interpolated values at the specified coordinates,
      with extrapolated values where necessary.
    """
    # Generate the grid of known points based on the element size and shift to pixel centers
    grid_x, grid_y = np.meshgrid(
        (np.arange(data.shape[1]) + 0.5) * element_size, 
        (np.arange(data.shape[0]) + 0.5) * element_size
    )
    
    # Flatten the grid and data to use with griddata
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    values = data.ravel()
    
    # Stack x and y coordinates for interpolation
    query_points = np.column_stack((x_coords, y_coords))
    
    # Interpolate data at the requested coordinates
    interpolated_values = griddata(points, values, query_points, method=method)
    
    # Use nearest-neighbor interpolation for any NaNs (extrapolation for out-of-bounds)
    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        interpolated_values[nan_mask] = griddata(points, values, query_points[nan_mask], method='nearest')
    
    return interpolated_values


def create_emodulus_interpolator(nodes_df):
    # Define lambda function that performs interpolation
    emodulus_interpolator = lambda x: interpolate_pixel_data(E_grid,calculate_element_size(nodes_df=nodes_df),x[0], x[1])

    return emodulus_interpolator



E.interpolate(create_emodulus_interpolator(nodes_df=nodes_df))

# TODO remove for varying E
# E_min = 100000.0
E.x.array[:] = np.full_like(E.x.array[:],E_max)

lam = le.get_lambda(E,nu)
mue = le.get_mu(E,nu)

dim = domain.topology.dim
alex.os.mpi_print('spatial dimensions: '+str(dim), rank)
x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)



# Simulation parameters ####
dt_start = 0.001
dt_global = dlfx.fem.Constant(domain, dt_start)
t_global = dlfx.fem.Constant(domain,0.0)
trestart_global = dlfx.fem.Constant(domain,0.0)
Tend = 30.0 * dt_global.value


gc = dlfx.fem.Constant(domain, 1.0)
eta = dlfx.fem.Constant(domain, 0.00001)
epsilon = dlfx.fem.Constant(domain, 0.05)
Mob = dlfx.fem.Constant(domain, 1000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)




## define boundary conditions crack
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)


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
        # pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
        
def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mue,
        Gc=gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]

# setup tracking
s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
s_zero_for_tracking_at_nodes.interpolate(sub_expr)

atol=(x_max_all-x_min_all)*0.000 # for selection of boundary



[Res, dResdw] = get_residuum_and_gateaux(delta_t=dt_global)
# w_D = dlfx.fem.Function(W) # for dirichlet BCs

# front_back = bc.get_frontback_boundary_of_box_as_function(domain,comm,atol=0.1*atol)
# bc_front_back = bc.define_dirichlet_bc_from_value(domain,0.0,2,front_back,W,0)

# def compute_displacement():
#     x = ufl.SpatialCoordinate(domain)
#     u_x = 0.0 * x[0]
#     u_y = - t_global * 1.0 * x[1] 
#     return ufl.as_vector([u_x, u_y]) # only 2 components in 2D

# bc_expression = dlfx.fem.Expression(compute_displacement(),W.sub(0).element.interpolation_points())
# top_bottom_bc = bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol*0.0) #bc.get_boundary_for_surfing_boundary_condition_2D(domain,comm,atol=atol,epsilon=epsilon.value) #bc.get_topbottom_boundary_of_box_as_function(domain,comm,atol=atol)
# top_bottom_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, top_bottom_bc)
# dofs_at_top_bottom = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, top_bottom_facets) 



def get_bcs(t):
    bcs = []
    # w_D.sub(0).interpolate(bc_expression)
    # bc_top_bottom : dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(w_D,dofs_at_top_bottom)
    uy = -t_global.value * 1.0
    bc_top = bc.define_dirichlet_bc_from_value(domain,uy,1,bc.get_top_boundary_of_box_as_function(domain,comm,atol),W,0)
    bc_bottom = bc.define_dirichlet_bc_from_value(domain,0.0,1,bc.get_bottom_boundary_of_box_as_function(domain,comm,atol),W,0)
    
    bc_left_x = bc.define_dirichlet_bc_from_value(domain,0.0,0,bc.get_left_boundary_of_box_as_function(domain,comm,atol),W,0)
    bc_right_x = bc.define_dirichlet_bc_from_value(domain,0.0,0,bc.get_right_boundary_of_box_as_function(domain,comm,atol),W,0)
    # bc_left_right_y = bc.define_dirichlet_bc_from_value(domain,0.0,1,bc.get_leftright_boundary_of_box_as_function(domain,comm,atol),W,0)
        # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    bcs.append(bc_top)
    bcs.append(bc_bottom)
    bcs.append(bc_left_x)
    bcs.append(bc_right_x)
    # bcs.append(bc_left_right_y)
    # bcs.append(bc_front_back)
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
    sigma = phaseFieldProblem.sigma_degraded(u,s,lam,mue,eta)
    # Rx_top, Ry_top = pp.reaction_force(sigma,n=n,ds=ds_top_tagged(top_surface_tag),comm=comm)
    
    # um1, _ = ufl.split(wm1)

    # dW = pp.work_increment_external_forces(sigma,u,um1,n,ds,comm=comm)
    # Work.value = Work.value + dW
    
    # A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    # eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mue)
    # Jx, Jy = alex.linearelastic.get_J_2D(eshelby,n,ds=ds(external_surface_tag),comm=comm)
    # Jx_vol, Jy_vol = alex.linearelastic.get_J_2D_volume_integral(eshelby,ufl.dx,comm)
    
    # alex.os.mpi_print(pp.getJString2D(Jx,Jy),rank)
    

    
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    # s_zero_for_tracking_at_nodes.interpolate(s)
    # max_x, max_y, min_x, min_y  = pp.crack_bounding_box_2D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking_at_nodes),comm)
    # x_ct = max_x
    

        

            # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    # break out of loop if no postprocessing required
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    # break out of loop if no postprocessing required
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0: 
        return 
    

    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    E.name = "E"
    pp.write_field(domain,outputfile_xdmf_path,E,t,comm,S)
    pp.write_tensor_fields(domain,comm,[sigma],["sig"],outputfile_xdmf_path,t)

def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]

def after_last_timestep():
    # pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t_global.value, comm)
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
        # pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy","x_pf_crack","x_macr","Rx", "Ry", "dW", "W", "A", "dt"])
        

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

# pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
# pp.write_field(domain,outputfile_xdmf_path,E,0.0,comm,S)


