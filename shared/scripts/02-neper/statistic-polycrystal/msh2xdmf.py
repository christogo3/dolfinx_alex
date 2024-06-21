import meshio

import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
import numpy as np
import os 
import glob
import sys

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]


# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1:
    quit()




# read msh    
data = meshio.read(os.path.join(script_path,"gene_grou_3.msh"))

def read_cell_group_mappings(file_path):
    cell_group_mappings = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Skip the header lines until we find "$Groups"
        reading_groups = False
        for line in lines:
            line = line.strip()
            
            if line == "elset":
                reading_groups = True
                continue
            if line == "$EndGroups":
                reading_groups = False
                continue
            
            if reading_groups:
                parts = line.split()
                if len(parts) == 2:
                    cell_id = int(parts[0])
                    group_id = int(parts[1])
                    cell_group_mappings.append((cell_id, group_id))

    return cell_group_mappings

# Path to your text file
file_path = os.path.join(script_path,"gene_grou_3.msh")

# Reading the file and getting the mappings
cell_group_mappings = read_cell_group_mappings(file_path)

points = data.points
cells = data.cells_dict['tetra']
cells_id = data.cell_data_dict['gmsh:physical']['tetra']
polyhedras_id = data.cell_data_dict['gmsh:geometrical']['tetra']

# activate cells according to fvol (approximately)
cells_id_max = np.max(cells_id)
# active_cells = cells #  look into gmesh geometrical for polys! #[cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if cells_id[index] <= f_vol*cells_id_max]
# active_cells = [cell for index, cell in enumerate(cells) if polyhedras_id[index] not in [5]]
# active_cells = [cell for index, cell in enumerate(cells) if (polyhedras_id[index]) % 6 != 0]
active_cells = [cell for index, cell in enumerate(cells) if (cell_group_mappings[polyhedras_id[index]-1][1]) != 1]


# setup dolfinx mesh
cell = ufl.Cell('tetrahedron', 3) # 3D
element = ufl.VectorElement('Lagrange', cell, 1)
mesh = ufl.Mesh(element)
domain = dlfx.mesh.create_mesh(comm, active_cells, points, mesh)
    
max_x = np.max(domain.geometry.x[:,0])
max_y = np.max(domain.geometry.x[:,1])
max_z = np.max(domain.geometry.x[:,2])
min_x = np.min(domain.geometry.x[:,0])
min_y = np.min(domain.geometry.x[:,1])
min_z = np.min(domain.geometry.x[:,2])
    
# print('Before scaling ...')
# print(min_x, max_x)
# print(min_y, max_y)
# print(min_z, max_z)
# sys.stdout.flush()
    
# scal_x = a_edge/(max_x-min_x)
# scal_y = a_edge/(max_y-min_y)
# scal_z = a_edge/(max_z-min_z)
   
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

# write xdmf-file with mesh
    
with dlfx.io.XDMFFile(comm, os.path.join(script_path,"polycrystal.xdmf"), 'w') as xdmf:
    xdmf.write_mesh(domain)
    