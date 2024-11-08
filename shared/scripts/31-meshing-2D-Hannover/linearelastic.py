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

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
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


# def arrange_cells_2D(connectivity_df, mesh_dims):
#     cell_grid = np.zeros(mesh_dims, dtype=int)  # 2D array to store cell IDs

#     # Fill the cell_grid row by row based on Cell ID order in the file
#     for index, row in connectivity_df.iterrows():
#         cell_id = row['Cell ID']
        
#         # Adjust row index to reverse the y-axis
#         row_idx = (mesh_dims[0] - 1) - (index // mesh_dims[1])
#         col_idx = index % mesh_dims[1]
        
#         cell_grid[row_idx, col_idx] = cell_id

#     return cell_grid

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
    E = cell_data_df['E-Modul'].values

    # Populate the density grid based on cell_id_grid
    for row in range(cell_id_grid.shape[0]):
        for col in range(cell_id_grid.shape[1]):
            cell_id = cell_id_grid[row, col]
            if cell_id < len(E):
                E_Grid[row, col] = E[cell_id]
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
folder_path = os.path.join(script_path,"dcb")
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
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
V = dlfx.fem.FunctionSpace(domain, Ve)
Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
# Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
S = dlfx.fem.FunctionSpace(domain, Se)

E = dlfx.fem.Function(S)
nu = dlfx.fem.Constant(domain=domain,c=0.3)


xx  = np.array(domain.geometry.x).T
xx = xx[0:2]


# points_to_interpolate_at = np.array((xx[0],xx[1])).T
# interp_data = griddata(points,
#                         e_modulus_values,
#                         points_to_interpolate_at,
#                         method="linear")

# from scipy.interpolate import LinearNDInterpolator

# interpolator = LinearNDInterpolator(points, e_modulus_values)
# interp_data = interpolator(xx[0], xx[1])

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


# E_interpolated = interpolate_pixel_data(data=E_grid,element_size=calculate_element_size(nodes_df=nodes_df),
#                                         x_coords=xx[0],
#                                         y_coords=xx[1])

def create_emodulus_interpolator(nodes_df, point_data_df):
    # Define lambda function that performs interpolation
    emodulus_interpolator = lambda x: interpolate_pixel_data(E_grid,calculate_element_size(nodes_df=nodes_df),x[0], x[1])

    return emodulus_interpolator


#Plot the interpolated data
# plt.figure(figsize=(8, 6))
# # plt.imshow(interp_data, extent=(0, 10, 0, 10), origin="lower", cmap="viridis")
# # plt.colorbar(label="E-modulus values")
# plt.scatter(xx[0], xx[1], c=E_interpolated, edgecolor="k", s=40, cmap="viridis")  # original points
# plt.title("Interpolated E-Modulus Values")
# plt.xlabel("X coordinate")
# plt.ylabel("Y coordinate")

# # Save the plot as a file
# plt.savefig("interpolated_e_modulus.png", dpi=300, bbox_inches="tight")
# plt.close()  # Close the pl

# E.x.array[all_dofs_E_local] = griddata(points,
#                         e_modulus_values,
#                         (xx[0],xx[1]),
#                         method="linear")
# # E.x.array[:] = np.ones_like(E.x.array[:])
# E.x.scatter_forward()

E.interpolate(create_emodulus_interpolator(nodes_df=nodes_df, point_data_df=point_data_df))

lam = le.get_lambda(E,nu)
mue = le.get_mu(E,nu)

pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
pp.write_field(domain,outputfile_xdmf_path,E,0.0,comm,S)

# def linear_displacements(V: dlfx.fem.FunctionSpace, 
#                                eps_mac: dlfx.fem.Constant):
#     E_D = dlfx.fem.Function(V)
    
#     E_D.interpolate(lambda x: )
#     dim = ut.get_dimension_of_function(E_D)
#     if dim == 3:
#         for k in range(0, dim):
#             u_D.sub(k).interpolate(lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] + eps_mac.value[k, 2]*x[2] )
#             u_D.x.scatter_forward()
#     elif dim == 2:
#          for k in range(0, dim):
#             u_D.sub(k).interpolate(lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] )
#             u_D.x.scatter_forward()
#     return E_D


# def interpolation_function(x):
#     return True
    
# expr = dlfx.fem.Expression(interpolation_function,S.element.interpolation_points())
# E.interpolate(expr)



a = 1
# # define boundary condition on top and bottom
# fdim = domain.topology.dim -1

# bcs = []
             
# # define solution, restart, trial and test space
# u =  dlfx.fem.Function(V)
# urestart =  dlfx.fem.Function(V)
# du = ufl.TestFunction(V)
# ddu = ufl.TrialFunction(V)

# def before_first_time_step():
#     urestart.x.array[:] = np.ones_like(urestart.x.array[:])
    
#     # prepare newton-log-file
#     if rank == 0:
#         sol.prepare_newton_logfile(logfile_path)
#         pp.prepare_graphs_output_file(outputfile_graph_path)
#     # prepare xdmf output 
#     pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

# def before_each_time_step(t,dt):
#     # report solution status
#     if rank == 0:
#         sol.print_time_and_dt(t,dt)
      
# linearElasticProblem = alex.linearelastic.StaticLinearElasticProblem()

# def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
#     [Res, dResdw] = linearElasticProblem.prep_newton(u,du,ddu,lam,mu)
#     return [Res, dResdw]

# x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)

# atol=(x_max_all-x_min_all)*0.02 # for selection of boundary
# def get_bcs(t):
#     if column_of_cmat_computed[0] < 6:
#         eps_mac = alex.homogenization.unit_macro_strain_tensor_for_voigt_eps(domain,column_of_cmat_computed[0])
#     else: # to avoid out of bounds index
#         eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0]]))
#     bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, V,eps_mac=eps_mac, atol=atol)
#     return bcs

# n = ufl.FacetNormal(domain)
# simulation_result = np.array([0.0])
# vol = (x_max_all-x_min_all) * (y_max_all - y_min_all) * (z_max_all - z_min_all)
# Chom = np.zeros((6, 6))

# column_of_cmat_computed=np.array([0])

# def after_timestep_success(t,dt,iters):
#     u.name = "u"
#     pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    
#     sigma_for_unit_strain = alex.homogenization.compute_averaged_sigma(u,lam,mu, vol)
    
#     # write to newton-log-file
#     if rank == 0:
#         if column_of_cmat_computed[0] < 6:
#             Chom[column_of_cmat_computed[0],:] = sigma_for_unit_strain
#         else:
#             t = 2.0*Tend # exit
#             return
#         print(column_of_cmat_computed[0])
#         column_of_cmat_computed[0] = column_of_cmat_computed[0] + 1
#         sol.write_to_newton_logfile(logfile_path,t,dt,iters)
        
#     urestart.x.array[:] = u.x.array[:] 


               
# def after_timestep_restart(t,dt,iters):
#     u.x.array[:] = urestart.x.array[:]
     
# def after_last_timestep():
#     # stopwatch stop
#     timer.stop()

#     # report runtime to screen
#     if rank == 0:
#         print(np.array_str(Chom, precision=2))
        
#         print(alex.homogenization.print_results(Chom))
        
#         runtime = timer.elapsed()
#         sol.print_runtime(runtime)
#         sol.write_runtime_to_newton_logfile(logfile_path,runtime)

# sol.solve_with_newton_adaptive_time_stepping(
#     domain,
#     u,
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
#     print_bool=True
# )

