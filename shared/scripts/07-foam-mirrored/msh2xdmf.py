import meshio

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


# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 1:
    quit()

# data = meshio.read(os.path.join(alex.os.resources_directory,"mirrored_hypo_test_128.xdmf"))
data = meshio.read(os.path.join(script_path,"mirrored_hypo_test_128.xdmf"))
points = data.points
cells = data.cells_dict['tetra']

max = np.max(cells)
min = np.min(cells)
max_points = len(points)

# setup dolfinx mesh
cell = ufl.Cell('tetrahedron', 3) # 3D
element = ufl.VectorElement('Lagrange', cell, 1)
mesh = ufl.Mesh(element)
domain = dlfx.mesh.create_mesh(comm, cells, points, mesh)
    

with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w') as xdmf:
    xdmf.write_mesh(domain)
    