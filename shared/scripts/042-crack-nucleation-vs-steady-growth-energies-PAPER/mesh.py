import pygmsh
import meshio
import numpy as np
import gmsh
import os
import argparse

# Script setup
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Argument parser
parser = argparse.ArgumentParser(description="Generate a rectangular domain with a full-circle notch.")
try:
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of the notch circle")
    parser.add_argument("--width", type=float, required=True, help="Width of the rectangular domain")
    parser.add_argument("--height", type=float, required=True, help="Height of the rectangular domain")
    parser.add_argument("--e0", type=float, required=True, help="Mesh element size")
    args = parser.parse_args()
    dhole = args.dhole
    width = args.width
    height = args.height
    e0 = args.e0
except:
    print("Could not parse arguments. Using default values.")
    dhole = 1.0
    width = 5.0
    height = 2.0
    e0 = 0.02  # Default mesh size

# Initialize geometry
geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

# Define rectangular domain
rect_bottom_left = [0.0, -height / 2, 0.0]  # Bottom-left corner of the rectangle
rectangle = model.add_rectangle(rect_bottom_left, width, height, 0)

# Define the full-circle notch
# notch_center = [0.5*width, 0.0 , 0.0]  # Center of the circle on the left edge
notch_center = [0.0, 0.0 , 0.0]  
circle_notch = model.add_disk(notch_center, dhole / 2)

# Subtract the full circle from the rectangular domain
domain = model.boolean_difference([rectangle], [circle_notch])

# Add physical groups
model.synchronize()
model.add_physical(domain, "Domain")

# Generate mesh
model.generate_mesh(dim=2, verbose=True)
gmsh.write(f"{script_path}/{script_name_without_extension}.msh")
gmsh.clear()
model.__exit__()

# Read and process the mesh
mesh = meshio.read(f"{script_path}/{script_name_without_extension}.msh")
nodes = mesh.points[:, :2]
elems = mesh.get_cells_type("triangle")
elem_data = mesh.cell_data_dict["gmsh:physical"]["triangle"] - 1  # Domain: 0

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


