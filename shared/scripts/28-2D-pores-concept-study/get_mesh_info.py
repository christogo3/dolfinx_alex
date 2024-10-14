import meshio
import numpy as np
import matplotlib.pyplot as plt
import os
import alex.util
import alex.os
import argparse


parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--mesh_file", type=str, required=True, help="Name of the mesh file")
    args = parser.parse_args()
    mesh_file = args.mesh_file
except: 
    mesh_file = "mesh_holes.xdmf"


# Set up paths
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path, script_name_without_extension)
input_file_path = os.path.join(script_path,mesh_file)
output_histogram_path = os.path.join(script_path, "edge_length_distribution.png")

# Load the mesh data
data = meshio.read(input_file_path)
points = data.points

# Initialize a list to store edge lengths from all types of cells
all_edge_lengths = []

# Define a function to compute edge lengths from cells
def compute_edge_lengths(points, cells, vertices_per_cell):
    edges = set()
    for cell in cells:
        for i in range(vertices_per_cell):
            for j in range(i + 1, vertices_per_cell):
                edge = tuple(sorted((cell[i], cell[j])))
                edges.add(edge)
    edges = np.array(list(edges))
    edge_vectors = points[edges[:, 0]] - points[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    return edge_lengths

# Loop through each cell type and compute edge lengths
for cell_type, cell_data in data.cells_dict.items():
    if cell_type == 'triangle':
        all_edge_lengths.extend(compute_edge_lengths(points, cell_data, 3))
    elif cell_type == 'tetra':
        all_edge_lengths.extend(compute_edge_lengths(points, cell_data, 4))

# Convert the list of all edge lengths to a NumPy array for easy processing
all_edge_lengths = np.array(all_edge_lengths)

# Compute distribution details
min_edge_length = np.min(all_edge_lengths)
max_edge_length = np.max(all_edge_lengths)
mean_edge_length = np.mean(all_edge_lengths)
std_edge_length = np.std(all_edge_lengths)
num_elements = len(all_edge_lengths)

# Print computed values
print(f"Minimum Edge Length: {min_edge_length}")
print(f"Maximum Edge Length: {max_edge_length}")
print(f"Mean Edge Length: {mean_edge_length}")
print(f"Standard Deviation of Edge Length: {std_edge_length}")
print(f"Number of Edge Elements: {num_elements}")

parameters_to_write = {
        'min_edge_length': min_edge_length,
        'max_edge_length': max_edge_length,
        'mean_edge_length': mean_edge_length,
        'std_edge_length': std_edge_length,
    }
# store parameters
def append_to_file(filename, parameters):
    with open(filename, 'a') as file:
        for key, value in parameters.items():
            file.write(f"{key}={value}\n")
parameter_path = os.path.join(script_path,"parameters.txt")
append_to_file(parameters=parameters_to_write,filename=parameter_path)

# Plot histogram and save
plt.hist(all_edge_lengths, bins=50, edgecolor='black')
plt.title("Edge Length Distribution")
plt.xlabel("Edge Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(output_histogram_path)
plt.close()

print(f"Histogram saved to {output_histogram_path}")
