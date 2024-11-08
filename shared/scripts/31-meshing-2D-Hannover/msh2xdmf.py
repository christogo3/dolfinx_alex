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

script_path = os.path.dirname(__file__)

# control parameter
filename = 'foam_4096'
a_edge = 400
f_vol = 0.6

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1:
    quit()

# read msh    
print('Reading '+filename+'.msh ...')
data = meshio.read(os.path.join(script_path,"out.msh"))
points_tmp = data.points
points_tmp = points_tmp[:,0:2]
points = copy.deepcopy(points_tmp)
points[:,0] = points_tmp[:,1]
points[:,1] = points_tmp[:,0]

# y_min, y_max = points[:, 1].min(), points[:, 1].max()
# # Reflect y-values across the midpoint
# points[:, 1] = y_max - (points[:, 1] - y_min)

cells = data.cells_dict['triangle']
cells_id = data.cell_data_dict['gmsh:physical']['triangle']

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
    
max_x = np.max(domain.geometry.x[:,0])
max_y = np.max(domain.geometry.x[:,1])
min_x = np.min(domain.geometry.x[:,0])
min_y = np.min(domain.geometry.x[:,1])
    
# print('Before scaling ...')
# print(min_x, max_x)
# print(min_y, max_y)
# print(min_z, max_z)
# sys.stdout.flush()
    
scal_x = 0.0111
scal_y = scal_x
# scal_z = a_edge/(max_z-min_z)
   
domain.geometry.x[:,0] = (domain.geometry.x[:,0]-min_x)*scal_x
domain.geometry.x[:,1] = (domain.geometry.x[:,1]-min_y)*scal_y
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

# write xdmf-file with mesh
print('Writing '+filename+'.xdmf ...')
    
with dlfx.io.XDMFFile(comm, os.path.join(script_path,"dlfx_mesh.xdmf"), 'w') as xdmf:
    xdmf.write_mesh(domain)
    