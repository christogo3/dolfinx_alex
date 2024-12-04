import pygmsh
import meshio
import numpy as np
import alex.postprocessing
import gmsh
import os
import argparse

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
parameter_path = os.path.join(script_path, "parameters.txt")

parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--nholes", type=int, required=True, help="Number of holes")
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of hole")
    parser.add_argument("--wsteg", type=float, required=True, help="Width of steg")
    parser.add_argument("--e0", type=float, required=True, help="Size of elements")
    parser.add_argument("--e1", type=float, required=True, help="Coarse element size")
    args = parser.parse_args()
    Nholes = args.nholes
    dhole = args.dhole
    wsteg = args.wsteg
    e0 = args.e0  # Fine mesh size
    e1 = args.e1  # Coarse mesh size
except:
    print("Could not parse arguments")
    Nholes = 4
    dhole = 1.0
    wsteg = 4.0
    e0 = 0.02  # Fine mesh size
    e1 = 0.8  # Coarse mesh size

w_cell = dhole + wsteg
h_cell = w_cell
l0 = (Nholes + 2) * w_cell
h0 = 2 * w_cell

filename = os.path.join(script_path, script_name_without_extension)
mesh_info = True

geom = pygmsh.occ.Geometry()
model = geom.__enter__()

# Generate surrounding matrix
p0 = model.add_point([0, -h0 / 2])
p1 = model.add_point([l0, -h0 / 2])
p2 = model.add_point([l0, +h0 / 2])
p3 = model.add_point([0, h0 / 2])
line0 = model.add_line(p0, p1)  # Bottom
line1 = model.add_line(p1, p2)  # Right
line2 = model.add_line(p2, p3)  # Top
line3 = model.add_line(p3, p0)  # Left
outer = model.add_curve_loop([line0, line1, line2, line3])
eff_matr = model.add_plane_surface(outer)

model.synchronize()

# Generate quadratic unit cells with circular pores
cells = []
for n in range(0, Nholes):
    x_center = w_cell + w_cell / 2.0 + n * w_cell
    y_center = 0.0
    left_bottom_rectangle = [x_center - w_cell / 2.0, -h_cell / 2, 0]
    cell_matrix_tmp = model.add_rectangle(left_bottom_rectangle, w_cell, h_cell, 0)
    eff_matr = model.boolean_fragments(eff_matr, cell_matrix_tmp, delete_first=True, delete_other=True)
    
    hole = model.add_disk([x_center, y_center], dhole / 2)
    eff_matr = model.boolean_difference(eff_matr, hole)

# Add crack as a line
p8 = model.add_point([0.0, 0.0])         
p9 = model.add_point([l0, 0.0])
crack = model.add_line(p8, p9)  # Horizontal crack at y=0

# Create mesh fields for variable element sizes
# Field 1: Distance to holes
gmsh.model.mesh.field.add("Distance", 1)
hole_centers = [[w_cell + w_cell / 2.0 + n * w_cell, 0.0, 0.0] for n in range(Nholes)]
gmsh.model.mesh.field.setNumbers(1, "NodesList", [model.add_point(center, e0)._id for center in hole_centers])

# Field 2: Distance to the crack
gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "EdgesList", [crack._id])

# Combine the two distance fields
gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

# Field 4: Threshold for element size transition
gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 3)
gmsh.model.mesh.field.setNumber(4, "SizeMin", e0)  # Fine mesh size
gmsh.model.mesh.field.setNumber(4, "SizeMax", e1)  # Coarse mesh size
gmsh.model.mesh.field.setNumber(4, "DistMin", dhole / 2)  # Distance for fine mesh
gmsh.model.mesh.field.setNumber(4, "DistMax", w_cell)  # Transition distance

gmsh.model.mesh.field.setAsBackgroundMesh(4)

# Synchronize the model and generate the mesh
model.synchronize()
model.add_physical(eff_matr, 'eff_matrix')
model.add_physical(crack, 'crack')

model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + '.msh')
gmsh.clear()
model.__exit__()

# Post-process and save the mesh
mesh = meshio.read(filename + '.msh')
nodes = mesh.points[:, 0:2]
elems = mesh.get_cells_type('triangle')
elem_data = mesh.cell_data_dict['gmsh:physical']['triangle'] - 1  # Adjusted for physical groups

if mesh_info:
    print('NODES:')
    print(nodes)
    print('ELEMENTS')
    print(elems)
    print('ELEMENT DATA')
    print(elem_data)

cell_mesh = meshio.Mesh(points=nodes, cells={'triangle': elems}, cell_data={'name_to_read': [elem_data]})
meshio.write(os.path.join(script_path, script_name_without_extension + '.xdmf'), cell_mesh)

parameters_to_write = {
    'nholes': Nholes,
    'dhole': dhole,
    'wsteg': wsteg,
    'e0': e0,
    'e1': e1
}

alex.postprocessing.append_to_file(parameter_path, parameters_to_write)




