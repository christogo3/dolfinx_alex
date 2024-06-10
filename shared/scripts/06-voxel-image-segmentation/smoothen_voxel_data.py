import os
import alex.os
import numpy as np
import alex.postprocessing
from mpi4py import MPI


import alex.imageprocessing as img




# set up MPI parallel processing
comm = MPI.COMM_WORLD

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# This specifies the voxel file we want to read and use as I for chan veese
voxel_number_x_reduced = 128
voxel_file_in_path  = os.path.join(script_path,"output_sub"+ str(voxel_number_x_reduced)+ ".dat")


# The voxel data as a 3D array
voxel_data_3d_array = img.read_voxel_data_leS(voxel_file_in_path, voxel_number_x_reduced, voxel_number_x_reduced, voxel_number_x_reduced)

# write back for test
# img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array.flatten(order="F"), os.path.join(script_path,"output_back" + str(voxel_number_x_reduced) + ".dat"))

# smooth data
voxel_data_3d_array_smooth = img.smooth_3d_array(voxel_data_3d_array,iterations=4,
                                             mode="nearest")

img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array_smooth.flatten(order="F"), os.path.join(script_path,"output_smooth_sub" + str(voxel_number_x_reduced) + ".dat"))



voxel_data_3d_array_smoothed_threshold = img.apply_threshold(voxel_data_3d_array_smooth,0.5,1,0)
img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array_smoothed_threshold.flatten(order="F"), os.path.join(script_path,"output_smoothed_theshhold_sub" + str(voxel_number_x_reduced) + ".dat"))
