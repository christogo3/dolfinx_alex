import os
import alex.os
import numpy as np
import alex.postprocessing
import dolfinx as dlfx
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt

import alex.imageprocessing as img
import alex.solution as sol

import pydicom
import pydicom.dataset
from pydicom import dcmread

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# read dicom
dcm_file_path = os.path.join(script_path,"AluSchaum_AlSi10_P1_1024_1024_512_gray.dcm")
ds : pydicom.dataset.FileDataset = dcmread(dcm_file_path)
scan_intensity_as_array = np.array(ds.pixel_array) # this is 512 x 1024 x 1024
# only select a subarray TODO currently only works for cubes, TODO does not work in parallel
dimension = 256
scan_intensity_as_array = scan_intensity_as_array[:dimension,:dimension,:dimension]

min_intensity = scan_intensity_as_array.min()
max_intensity = scan_intensity_as_array.max()

np.save(os.path.join(script_path, "voxel_data.npy"), scan_intensity_as_array)

# Create 100 evenly spaced bins between the minimum and maximum intensities
def print_intensity_as_histogram(script_path, scan_intensity_as_array, min_intensity, max_intensity, output_path):
    bins = np.linspace(min_intensity, max_intensity, 101)

# Calculate the histogram
    hist, bin_edges = np.histogram(scan_intensity_as_array, bins=bins)

# Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scan_intensity_as_array.flatten(), bins=bins, edgecolor='black')
    plt.title('Histogram of Voxel Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Number of Voxels')
    plt.grid(True)

# Save the plot to a file
    plt.savefig(output_path)
    plt.close()

print_intensity_as_histogram(script_path, scan_intensity_as_array, 
                             min_intensity, max_intensity,
                             os.path.join(script_path,"histogram.png"))