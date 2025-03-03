import meshio

import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
import numpy as np
import os 
import glob
import sys
import copy

def scale_domain_to_bounds_3d(domain, x_min_target, x_max_target, y_min_target, y_max_target, z_min_target, z_max_target):
    """
    Scales the points in the domain so that the coordinates fit within the specified 3D bounds.

    Parameters:
        domain: The domain object containing geometry to be scaled.
        x_min_target (float): Target minimum x value.
        x_max_target (float): Target maximum x value.
        y_min_target (float): Target minimum y value.
        y_max_target (float): Target maximum y value.
        z_min_target (float): Target minimum z value.
        z_max_target (float): Target maximum z value.

    Modifies:
        The domain's geometry is scaled in-place to match the target bounds.

    Returns:
        domain: The domain object with scaled geometry.
    """
    # Extract current min and max values for x, y, and z
    x_min_current = np.min(domain.geometry.x[:, 0])
    x_max_current = np.max(domain.geometry.x[:, 0])
    y_min_current = np.min(domain.geometry.x[:, 1])
    y_max_current = np.max(domain.geometry.x[:, 1])
    z_min_current = np.min(domain.geometry.x[:, 2])
    z_max_current = np.max(domain.geometry.x[:, 2])

    # Calculate scaling factors
    scal_x = (x_max_target - x_min_target) / (x_max_current - x_min_current)
    scal_y = (y_max_target - y_min_target) / (y_max_current - y_min_current)
    scal_z = (z_max_target - z_min_target) / (z_max_current - z_min_current)

    # Apply scaling and shifting
    domain.geometry.x[:, 0] = x_min_target + (domain.geometry.x[:, 0] - x_min_current) * scal_x
    domain.geometry.x[:, 1] = y_min_target + (domain.geometry.x[:, 1] - y_min_current) * scal_y
    domain.geometry.x[:, 2] = z_min_target + (domain.geometry.x[:, 2] - z_min_current) * scal_z

    # Return the updated domain
    return domain

script_path = os.path.dirname(__file__)

dx = 0.0151284074783325 # edge length of voxel
Nx = 150 # number of voxels in x - direction
Ny = Nx
Nz = Nx
# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1:
    quit()

# read msh    
print('Reading '+"out"+'.msh ...')
data = meshio.read(os.path.join(script_path,"mesh.xdmf"))
# match orientation of picture
points_tmp = data.points
points_tmp = points_tmp[:,0:3]
points = copy.deepcopy(points_tmp)
points[:,0] = points_tmp[:,1]
points[:,1] = points_tmp[:,0]
# points = data.points

# y_min, y_max = points[:, 1].min(), points[:, 1].max()
# # Reflect y-values across the midpoint
# points[:, 1] = y_max - (points[:, 1] - y_min)

cells = data.cells_dict['tetra']
cells_id = data.cell_data_dict['tetgen:ref']['tetra']

cells_id_max = np.max(cells_id)
# active_cells = cells #  look into gmesh geometrical for polys! #[cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if polyhedras_id[index] not in [5]]
active_cells = [cell for index, cell in enumerate(cells) if (cells_id[index]) == 1]


# setup dolfinx mesh
cell = ufl.Cell('tetrahedron', geometric_dimension=3) # 3D
element = ufl.VectorElement('Lagrange', cell, 1,dim=3)
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

# Define your target bounds
x_min_target, x_max_target = 0, Nx*dx
y_min_target, y_max_target = 0, Ny*dx
z_min_target, z_max_target = 0, Nz*dx

# Scale the domain geometry
scaled_domain = scale_domain_to_bounds_3d(domain, x_min_target, x_max_target, y_min_target, y_max_target,z_min_target, z_max_target)


# write xdmf-file with mesh
print('Writing '+"dlfx_mesh"+'.xdmf ...')
    
with dlfx.io.XDMFFile(comm, os.path.join(script_path,"dlfx_mesh.xdmf"), 'w') as xdmf:
    xdmf.write_mesh(scaled_domain)
    