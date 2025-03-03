# run this first on terminal:
# Xvfb :99 -screen 0 1024x768x24 &
# export DISPLAY=:99


import numpy as np
import pyvista as pv

from scipy.spatial import cKDTree
import open3d as o3d
import os

script_path = os.path.dirname(__file__)

# Step 1: Read the Tecplot ASCII file
filename = "emodul"
input_filename = filename + ".plt"  # Replace with your Tecplot file name
points = []

with open(os.path.join(script_path, input_filename), "r") as file:
    lines = file.readlines()

# Step 2: Extract the data (skip headers)
start_data = False
for line in lines:
    if "ZONE" in line:  # Data starts after the ZONE line
        start_data = True
        continue
    if start_data:
        values = line.strip().split()
        if len(values) == 3:  # Ensure correct number of columns
            points.append([float(values[0]), float(values[1]), float(values[2])])

# Convert to NumPy array
points = np.array(points)

# Step 3: Remove duplicate points
tree = cKDTree(points)
unique_indices = np.unique(tree.query(points, k=1)[1])  # Get unique point indices
unique_points = points[unique_indices]

print(f"Original points: {len(points)}, Unique points: {len(unique_points)}")

# Step 4: Compute the normals by normalizing the location vector of each point
# Normals are just the unit vector of each point
normals = unique_points / np.linalg.norm(unique_points, axis=1)[:, np.newaxis]

# Open3D point cloud creation
# Explained in https://www.youtube.com/watch?v=Ydo7RXDl7MM
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(unique_points[:,:3])
pcd.normals = o3d.utility.Vector3dVector(normals[:,:3])

# Create a PyVista point cloud
point_cloud = pv.PolyData(unique_points)

# Visualize the point cloud using PyVista (off-screen)
plotter = pv.Plotter(off_screen=True)  
plotter.add_points(point_cloud, render_points_as_spheres=True)

# Save a screenshot of the point cloud
output_image_filename = filename + "_point_cloud.png"
plotter.screenshot(os.path.join(script_path, output_image_filename))
print(f"Point cloud visualization saved as {output_image_filename}")

# Strategy: BPA
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

# Create the mesh using Ball Pivoting
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

# Clean the mesh (optional)
dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

# Save the meshes as .obj files first
bpa_mesh_obj_filename = filename + "_bpa_mesh.obj"
dec_mesh_obj_filename = filename + "_dec_mesh.obj"

o3d.io.write_triangle_mesh(os.path.join(script_path, bpa_mesh_obj_filename), bpa_mesh)
o3d.io.write_triangle_mesh(os.path.join(script_path, dec_mesh_obj_filename), dec_mesh)

print(f"Meshes saved as {bpa_mesh_obj_filename} and {dec_mesh_obj_filename}")

# Convert .obj files to .vtk using PyVista
bpa_mesh_vtk_filename = filename + "_bpa_mesh.vtk"
dec_mesh_vtk_filename = filename + "_dec_mesh.vtk"

# Load the .obj files using PyVista
bpa_mesh_pv = pv.read(os.path.join(script_path, bpa_mesh_obj_filename))
dec_mesh_pv = pv.read(os.path.join(script_path, dec_mesh_obj_filename))

# Save the meshes as .vtk files
bpa_mesh_pv.save(os.path.join(script_path, bpa_mesh_vtk_filename))
dec_mesh_pv.save(os.path.join(script_path, dec_mesh_vtk_filename))
print(f"Meshes converted and saved as {bpa_mesh_vtk_filename} and {dec_mesh_vtk_filename}")

