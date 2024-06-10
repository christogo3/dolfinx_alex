import os
import alex.os
import numpy as np
import alex.postprocessing
from mpi4py import MPI


import alex.imageprocessing as img


def mirror_and_combine(data, axis):
    """
    Enlarge a 3D array of voxel data by adding its mirrored counterpart along the specified axis.
    
    Parameters:
    data (numpy.ndarray): The input 3D array.
    axis (int): The axis along which to mirror and concatenate the data (0, 1, or 2).
    
    Returns:
    numpy.ndarray: The enlarged 3D array.
    """
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2")
    
    # Mirror the data along the specified axis
    mirrored_data = np.flip(data, axis=axis)
    
    # Concatenate the original and mirrored data along the specified axis
    enlarged_data = np.concatenate((data, mirrored_data), axis=axis)
    
    return enlarged_data


# set up MPI parallel processing
comm = MPI.COMM_WORLD

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

voxel_number_x_reduced = 16
voxel_file_in_path  = os.path.join(script_path,"output_sub"+ str(voxel_number_x_reduced)+ ".dat")


# The voxel data as a 3D array
voxel_data_3d_array = img.read_voxel_data_leS(voxel_file_in_path, voxel_number_x_reduced, voxel_number_x_reduced, voxel_number_x_reduced)

voxel_data_mirrored = mirror_and_combine(voxel_data_3d_array,axis=0)


img.write_voxel_file_leS(voxel_number_x_reduced*2, np.array(voxel_data_mirrored,dtype=np.uint8).flatten(order="F"), os.path.join(script_path,"output_mirrored" + str(voxel_number_x_reduced) + ".dat"))
# write back for test
# img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array.flatten(order="F"), os.path.join(script_path,"output_back" + str(voxel_number_x_reduced) + ".dat"))





# img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array_smooth.flatten(order="F"), os.path.join(script_path,"output_smooth_sub" + str(voxel_number_x_reduced) + ".dat"))



# voxel_data_3d_array_smoothed_threshold = img.apply_threshold(voxel_data_3d_array_smooth,0.5,1,0)
# img.write_voxel_file_leS(voxel_number_x_reduced, voxel_data_3d_array_smoothed_threshold.flatten(order="F"), os.path.join(script_path,"output_smoothed_theshhold_sub" + str(voxel_number_x_reduced) + ".dat"))
