import os
import alex.os
import alex.postprocessing

import alex.imageprocessing as img

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

voxel_file_in_path  = os.path.join(alex.os.resources_directory,'scans-small','hypo_test_128.dat')
voxel_number_x = 128

voxel_number_x_reduced = 16
img.output_reduced_voxel_file_leS(script_path, "output", voxel_file_in_path, voxel_number_x, voxel_number_x, voxel_number_x,
                              subarray_size=voxel_number_x_reduced,
                              start_x=0,start_y=0,start_z=0)

