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
parser = argparse.ArgumentParser(description="Generate a rectangular domain with a full-circle notch.")
try:
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of the notch circle")
    parser.add_argument("--width", type=float, required=True, help="Width of the rectangular domain")
    parser.add_argument("--height", type=float, required=True, help="Height of the rectangular domain")
    parser.add_argument("--e0", type=float, required=True, help="Mesh element size for fine regions")
    parser.add_argument("--e1", type=float, required=True, help="Mesh element size for coarser regions")
    args = parser.parse_args()
    dhole = args.dhole
    width = args.width
    height = args.height / 2.0
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

# Define main rectangular domain
rect_bottom_left = [0.0, -height / 2, 0.0]  # Bottom-left corner of the main rectangle
main_rectangle = model.add_rectangle(rect_bottom_left, width, height, 0)

# Define top and bottom rectangles
top_rectangle = model.add_rectangle([0.0, height / 2, 0.0], width, height / 2, 0)
bottom_rectangle = model.add_rectangle([0.0, -height, 0.0], width, height / 2, 0)

# Define the full-circle notch
notch_center = [dhole, 0.0, 0.0]  # Center of the circle
circle_notch = model.add_disk(notch_center, dhole / 2)

# Subtract the full circle from the main rectangular domain
main_domain = model.boolean_difference([main_rectangle], [circle_notch])

# Combine all regions into one domain
final_domain = model.boolean_union([main_domain, top_rectangle, bottom_rectangle])

# Add physical groups
main_domain_group = model.add_physical(main_domain, "Main Domain")
top_rectangle_group = model.add_physical(top_rectangle, "Top Rectangle")
bottom_rectangle_group = model.add_physical(bottom_rectangle, "Bottom Rectangle")

# Synchronize the model
model.synchronize()

# Define points for the horizontal line extending from the circle center
line_start = model.add_point([0.0, 0.0, 0.0], e0)  # Circle center
line_end = model.add_point([width, 0.0, 0.0], e1)  # Horizontal line ends at the right boundary
line = model.add_line(line_start, line_end)

# Set mesh fields for varying element sizes
# Field 1: Distance from the circle's center
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "NodesList", [circle_notch.x0[0]])

# Field 2: Distance from the horizontal line
gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "EdgesList", [line._id])

# Combine the two distance fields
gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

# Field 4: Threshold for mesh size transition
gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 3)
gmsh.model.mesh.field.setNumber(4, "SizeMin", e0)  # Fine mesh size
gmsh.model.mesh.field.setNumber(4, "SizeMax", e1)  # Coarse mesh size
gmsh.model.mesh.field.setNumber(4, "DistMin", dhole / 2)
gmsh.model.mesh.field.setNumber(4, "DistMax", width)

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


