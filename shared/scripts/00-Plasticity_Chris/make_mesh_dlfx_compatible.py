import dolfinx as dlfx
from mpi4py import MPI
import meshio
import numpy as np
import os
import ufl 
import copy
import argparse

# ========== USER CONFIGURATION ==========
output_subfolder_name = "meshes"

# USAGE:  python3 make_mesh_dlfx_compatible.py "$MESH_INPUT_DIR" -f mesh.xdmf
# python3 ./scripts/053-plasticity-without-fracture/make_mesh_dlfx_compatible.py "$/home/resources" -f AluSchaum_4x.msh
# ========================================

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mesh_files = []
input_file = '/home/resources/nanomesh.vtk' #/home/resources/AluSchaum_4x_dolfinx.xdmf
output_file = '/home/resources/Nanomesh.xdmf'


# --- Process mesh files ---
if rank == 0:
    print(f"Processing mesh: {input_file}")
    meshio_data = meshio.read(input_file)

    # Adjust point orientation
    points_tmp = meshio_data.points[:, :3]
    points = copy.deepcopy(points_tmp)
    points[:, 0] = points_tmp[:, 0]
    points[:, 1] = points_tmp[:, 1]
    points = points.astype(np.float64)

    # Filter active tetrahedral cells
    tetra_cells = meshio_data.cells_dict.get("tetra").astype(np.int32)
else:
    points = None
    active_cells = None


# Create mesh
cell = ufl.Cell('tetrahedron', geometric_dimension=3)
element = ufl.VectorElement('Lagrange', cell, 1, dim=3)
mesh = ufl.Mesh(element)
domain = dlfx.mesh.create_mesh(comm, tetra_cells, points, mesh)


# Write mesh
if rank == 0:
    print(f"Writing converted mesh to: {output_file}")
with dlfx.io.XDMFFile(comm, output_file, "w") as xdmf:
    xdmf.write_mesh(domain)


