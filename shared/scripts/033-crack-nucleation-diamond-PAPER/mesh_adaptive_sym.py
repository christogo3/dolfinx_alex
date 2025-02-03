import pygmsh
import meshio
import numpy as np
import gmsh
import os
import argparse
import alex.postprocessing

# Script setup
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
parameter_path = os.path.join(script_path, "parameters.txt")

# Argument parser
parser = argparse.ArgumentParser(description="Generate a rectangular domain with a diamond-shaped notch (top half only).")
try:
    parser.add_argument("--dhole", type=float, required=True, help="Diagonal length of the diamond-shaped notch")
    parser.add_argument("--width", type=float, required=True, help="Width of the rectangular domain")
    parser.add_argument("--height", type=float, required=True, help="Height of the rectangular domain")
    parser.add_argument("--e0", type=float, required=True, help="Mesh element size for fine regions")
    parser.add_argument("--e1", type=float, required=True, help="Mesh element size for coarser regions")
    args = parser.parse_args()
    dhole = args.dhole
    width = args.width
    height = args.height
    e0 = args.e0
    e1 = args.e1
except:
    print("Could not parse arguments. Using default values.")
    dhole = 1.0
    width = 10.0
    height = 5.0
    e0 = 0.02  # Fine mesh size
    e1 = 0.6   # Coarse mesh size

# Initialize geometry
geom = pygmsh.occ.Geometry()
model = geom.__enter__()

# Define top half rectangular domain
top_rect_bottom_left = [0.0, 0.0, 0.0]  # Bottom-left corner of the top half rectangle
top_rectangle = model.add_rectangle(top_rect_bottom_left, width, height / 2, 0)

# Define the diamond-shaped notch
notch_center = [width / 2.0, 0.0, 0.0]  # Center of the diamond
half_diagonal = dhole / 2

# Define the four vertices of the diamond
vertex1 = model.add_point([notch_center[0], notch_center[1] + half_diagonal, 0.0], e0)
vertex2 = model.add_point([notch_center[0] + half_diagonal, notch_center[1], 0.0], e0)
vertex3 = model.add_point([notch_center[0], notch_center[1] - half_diagonal, 0.0], e0)
vertex4 = model.add_point([notch_center[0] - half_diagonal, notch_center[1], 0.0], e0)

# Create the diamond by connecting the vertices
line1 = model.add_line(vertex1, vertex2)
line2 = model.add_line(vertex2, vertex3)
line3 = model.add_line(vertex3, vertex4)
line4 = model.add_line(vertex4, vertex1)
diamond_loop = model.add_curve_loop([line1, line2, line3, line4])
diamond_notch = model.add_plane_surface(diamond_loop)

# Subtract the diamond from the top rectangular domain
top_half_domain = model.boolean_difference([top_rectangle], [diamond_notch])

# Add physical groups
top_half_group = model.add_physical(top_half_domain, "Top Half Domain")

# Synchronize the model
model.synchronize()

# Set mesh fields for varying element sizes
# Field 1: Fine mesh around the diamond (notch)
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "NodesList", [vertex1._id, vertex2._id, vertex3._id, vertex4._id])

# Field 2: Fine mesh along the lower boundary (y=0)
line_start = model.add_point([0.0, 0.0, 0.0], e0)
line_end = model.add_point([width, 0.0, 0.0], e0)
lower_boundary = model.add_line(line_start, line_end)
gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "EdgesList", [lower_boundary._id])

# Field 3: Combine fine regions (notch + lower boundary)
gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

# Field 4: Transition to coarser mesh at the top boundary
gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 3)
gmsh.model.mesh.field.setNumber(4, "SizeMin", e0)  # Fine mesh size
gmsh.model.mesh.field.setNumber(4, "SizeMax", e1)  # Coarse mesh size
gmsh.model.mesh.field.setNumber(4, "DistMin", dhole / 2)
gmsh.model.mesh.field.setNumber(4, "DistMax", height / 2)

gmsh.model.mesh.field.setAsBackgroundMesh(4)

# Generate mesh
model.generate_mesh(dim=2, verbose=True)
gmsh.write(f"{script_path}/{script_name_without_extension}.msh")
gmsh.clear()
model.__exit__()

# Read and process the mesh
mesh = meshio.read(f"{script_path}/{script_name_without_extension}.msh")
nodes = mesh.points[:, :2]
elems = mesh.get_cells_type("triangle")
elem_data = mesh.cell_data_dict["gmsh:physical"]["triangle"] - 1  # Physical group indices

# Optional mesh information output
print("NODES:")
print(nodes)
print("ELEMENTS")
print(elems)
print("ELEMENT DATA")
print(elem_data)

# Write output mesh in XDMF format
cell_mesh = meshio.Mesh(
    points=nodes,
    cells={"triangle": elems},
    cell_data={"name_to_read": [elem_data]},
)
meshio.write(os.path.join(script_path, script_name_without_extension + ".xdmf"), cell_mesh)

parameters_to_write = {
    'dhole': dhole,
    'e0': e0,
    'e1': e1,
    'width': width,
    'height': height,
}

alex.postprocessing.write_to_file(parameter_path, parameters_to_write)