import meshio
import numpy as np
import matplotlib.pyplot as plt
import os
import alex.util
import alex.os

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path, script_name_without_extension)

# Step 1: Read the Mesh File
# input_file_path = os.path.join(alex.os.resources_directory, "finest", "coarse_pores_y_rotated.xdmf")
input_file_path = os.path.join(script_path, "dlfx_mesh.xdmf")
# Save the histogram as a PNG file
output_histogram_path = os.path.join(script_path, "edge_length_distribution.png")
data = meshio.read(input_file_path)
points = data.points
cells = data.cells_dict['triangle']

# Step 2: Compute Edge Lengths
def compute_edge_lengths(points, cells):
    # Get all unique edges in the mesh
    edges = set()
    for cell in cells:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted((cell[i], cell[j])))
                edges.add(edge)
    edges = np.array(list(edges))
    
    # Compute lengths of these edges
    edge_vectors = points[edges[:, 0]] - points[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    return edge_lengths

edge_lengths = compute_edge_lengths(points, cells)

# Step 3: Calculate Minimum, Maximum, Mean, and Standard Deviation of Edge Lengths
min_edge_length = np.min(edge_lengths)
max_edge_length = np.max(edge_lengths)
mean_edge_length = np.mean(edge_lengths)
std_edge_length = np.std(edge_lengths)
num_elements = len(cells)

# Print computed values
print(f"Minimum Edge Length: {min_edge_length}")
print(f"Maximum Edge Length: {max_edge_length}")
print(f"Mean Edge Length: {mean_edge_length}")
print(f"Standard Deviation of Edge Length: {std_edge_length}")
print(f"Number of Elements: {num_elements}")

# Step 4: Generate and Save Histogram
plt.hist(edge_lengths, bins=50, edgecolor='black')
plt.title("Edge Length Distribution")
plt.xlabel("Edge Length")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig(output_histogram_path)
plt.close()

print(f"Histogram saved to {output_histogram_path}")







    