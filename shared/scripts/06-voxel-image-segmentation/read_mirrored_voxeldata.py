import os
import alex.os
import numpy as np
import alex.postprocessing
import dolfinx as dlfx
import ufl
from mpi4py import MPI

import alex.imageprocessing as img
import alex.solution as sol

# set up MPI parallel processing
comm = MPI.COMM_WORLD

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# This specifies the voxel file we want to read and use as I for chan veese
voxel_number = 128
voxel_file_in_path  = os.path.join(script_path,"output_mirrored128.dat")


# The voxel data as a 3D array
voxel_data_3d_array = img.read_voxel_data_leS(voxel_file_in_path, voxel_number*2,voxel_number, voxel_number)

# creating fem mesh
domain: dlfx.mesh.Mesh = dlfx.mesh.create_box(comm=comm, points=((0,0,0),(2,1,1)), 
                                              n=(voxel_number*2,voxel_number, voxel_number),
                                              cell_type=dlfx.mesh.CellType.hexahedron)

I : dlfx.fem.Function = img.set_cell_field_from_voxel_data(voxel_data_3d_array, domain) # cell wise constant field
alex.postprocessing.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        xdmf_out.write_function(I,0.0)
