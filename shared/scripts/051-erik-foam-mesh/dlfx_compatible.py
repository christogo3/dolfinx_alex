import meshio

import alex.util
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
import numpy as np
import os 
import glob
import sys

import alex.os

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

alex.util.print_dolfinx_version()
sys.stdout.flush()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1: # cannot run in parallel
     quit()

# data = meshio.read(os.path.join(alex.os.resources_directory,"mirrored_hypo_test_128.xdmf"))
# data = meshio.read(os.path.join(script_path,"foam128_corrected_to_box.xdmf"))
# input_file_path = os.path.join(script_path,"mesh"+".xdmf")
# outputfile_xdmf_path = os.path.join(script_path,"foam"+str(60)+".xdmf")

input_file_path = os.path.join(script_path,"medium_pores"+".xdmf")
outputfile_xdmf_path = os.path.join(script_path,"medium_pores.xdmf")
data = meshio.read(input_file_path)
points = data.points
cells = data.cells_dict['tetra']
# cells_id = data.cell_data_dict['medit:ref']['tetra']
# cells = [cell for idx, cell in enumerate(cells) if cells_id[idx] == 1]



max = np.max(cells)
min = np.min(cells)
max_points = len(points)


# point_numbers = np.arange(len(points))

# contained = np.isin(point_numbers,cells)
# false_at = np.where(contained==False)

# points_f = []
# offset = np.full(len(points),0,dtype=np.int)
# for i in range(0,len(points)):
#     if np.isin(i,false_at).any():
#         offset[i+1:] -= 1
#     else:
#         points_f.append(points[i])
        

# for i in range(0,len(cells)):
#     for j in range(0,len(cells[i])):
#         cells[i][j] = cells[i][j] + offset[cells[i][j]]
#         a = 1
        

# point_numbers = np.arange(len(points_f))
# contained = np.isin(point_numbers,cells)
# false_at = np.where(contained==False)
    

# setup dolfinx mesh
cell = ufl.Cell('tetrahedron', 3) # 3D
element = ufl.VectorElement('Lagrange', cell, 1)
mesh = ufl.Mesh(element)
domain = dlfx.mesh.create_mesh(comm, cells, points, mesh)

    

with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w') as xdmf:
    xdmf.write_mesh(domain)