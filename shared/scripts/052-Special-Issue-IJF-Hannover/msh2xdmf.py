import meshio
print(meshio.__version__)
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
import numpy as np
import os 
import glob
import sys
import copy

def scale_domain_to_bounds(domain, x_min_target, x_max_target, y_min_target, y_max_target, original_domain):
    """
    Scales the points in the domain so that the coordinates fit within the specified bounds.

    Parameters:
        domain: The domain object containing geometry to be scaled.
        x_min_target (float): Target minimum x value.
        x_max_target (float): Target maximum x value.
        y_min_target (float): Target minimum y value.
        y_max_target (float): Target maximum y value.

    Modifies:
        The domain's geometry is scaled in-place to match the target bounds.

    Returns:
        domain: The domain object with scaled geometry.
    """
    # Extract current min and max values for x and y
    x_min_current = np.min(domain.geometry.x[:, 0])
    x_max_current = np.max(domain.geometry.x[:, 0])
    y_min_current = np.min(domain.geometry.x[:, 1])
    y_max_current = np.max(domain.geometry.x[:, 1])

    # Calculate scaling factors
    scal_x = (x_max_target - x_min_target) / (x_max_current - x_min_current)
    scal_y = (y_max_target - y_min_target) / (y_max_current - y_min_current)

    # Apply scaling and shifting
    domain.geometry.x[:, 0] = x_min_target + (domain.geometry.x[:, 0] - x_min_current) * scal_x
    domain.geometry.x[:, 1] = y_min_target + (domain.geometry.x[:, 1] - y_min_current) * scal_y

    # Return the updated domain
    return domain


def scale_points_to_target_domain(points, 
                                  x_min_target, x_max_target, 
                                  y_min_target, y_max_target):
    # Find current bounds of points
    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
    
    # Compute scale factors for each axis
    scale_x = (x_max_target - x_min_target) / (x_max - x_min)
    scale_y = (y_max_target - y_min_target) / (y_max - y_min)
    
    # Scale points
    points_scaled = np.empty_like(points)
    points_scaled[:,0] = x_min_target + (points[:,0] - x_min) * scale_x
    points_scaled[:,1] = y_min_target + (points[:,1] - y_min) * scale_y
    
    return points_scaled

script_path = os.path.dirname(__file__)

# control parameter
# filename = 'foam_4096'
# a_edge = 400
# f_vol = 0.6

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1:
    quit()

# read msh    
print('Reading '+'out.msh ...')
data = meshio.read(os.path.join(script_path,"out_4.msh"))
points_tmp = data.points
points_tmp = points_tmp[:,0:2]
points = copy.deepcopy(points_tmp)
points[:,0] = points_tmp[:,1]
points[:,1] = points_tmp[:,0]

# Define your target bounds
x_min_target, x_max_target = 0, 10.0
y_min_target, y_max_target = 0, 1.0

points = scale_points_to_target_domain(points,x_min_target,x_max_target,y_min_target,y_max_target)

# y_min, y_max = points[:, 1].min(), points[:, 1].max()
# # Reflect y-values across the midpoint
# points[:, 1] = y_max - (points[:, 1] - y_min)

cells = data.cells_dict['triangle']
cells_id = data.cell_data_dict['physical']['triangle']

cells_id_max = np.max(cells_id)
# active_cells = cells #  look into gmesh geometrical for polys! #[cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if polyhedras_id[index] not in [5]]
active_cells = [cell for index, cell in enumerate(cells) if (cells_id[index]) == 1]


# setup dolfinx mesh
cell = ufl.Cell('triangle', geometric_dimension=2) # 3D
element = ufl.VectorElement('Lagrange', cell, 1,dim=2)
mesh = ufl.Mesh(element)
domain = dlfx.mesh.create_mesh(comm, active_cells, points, mesh)
    
# max_x = np.max(domain.geometry.x[:,0])
# max_y = np.max(domain.geometry.x[:,1])
# min_x = np.min(domain.geometry.x[:,0])
# min_y = np.min(domain.geometry.x[:,1])
    
# # print('Before scaling ...')
# # print(min_x, max_x)
# # print(min_y, max_y)
# # print(min_z, max_z)
# # sys.stdout.flush()
    
# scal_x = 0.0111
# scal_y = scal_x
# # scal_z = a_edge/(max_z-min_z)
   
# domain.geometry.x[:,0] = (domain.geometry.x[:,0]-min_x)*scal_x
# domain.geometry.x[:,1] = (domain.geometry.x[:,1]-min_y)*scal_y
# domain.geometry.x[:,2] = (domain.geometry.x[:,2]-min_z)*scal_z
    
# max_x = np.max(domain.geometry.x[:,0])
# max_y = np.max(domain.geometry.x[:,1])
# max_z = np.max(domain.geometry.x[:,2])
# min_x = np.min(domain.geometry.x[:,0])
# min_y = np.min(domain.geometry.x[:,1])
# min_z = np.min(domain.geometry.x[:,2])
    
# print('After scaling ...')
# print(min_x, max_x)
# print(min_y, max_y)
# print(min_z, max_z)



# Scale the domain geometry
#scaled_domain = scale_domain_to_bounds(domain, x_min_target, x_max_target, y_min_target, y_max_target)


# write xdmf-file with mesh
print('Writing ''dlfx_mesh.xdmf ...')
    
with dlfx.io.XDMFFile(comm, os.path.join(script_path,"dlfx_mesh.xdmf"), 'w') as xdmf:
    xdmf.write_mesh(domain)
    