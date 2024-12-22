import pygmsh
import meshio
import numpy as np
import alex.postprocessing
import gmsh
import os
import argparse
import math  # Add this import for angle calculations

# Path setup
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
parameter_path = os.path.join(script_path, "parameters.txt")

# Argument parsing
parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--nholes", type=int, required=True, help="Number of holes")
    parser.add_argument("--dhole", type=float, required=True, help="Diagonal length of diamond holes")
    parser.add_argument("--wsteg", type=float, required=True, help="Width of steg")
    parser.add_argument("--e0", type=float, required=True, help="Size of elements")
    parser.add_argument("--e1", type=float, required=True, help="Coarse element size")
    parser.add_argument("--hole_angle", type=float, required=True, help="Opening angle of the holes in degrees")
    args = parser.parse_args()
    Nholes = args.nholes
    dhole = args.dhole
    wsteg = args.wsteg
    e0 = args.e0  # Fine mesh size
    e1 = args.e1  # Coarse mesh size
    hole_angle_deg = args.hole_angle  # Opening angle of the holes in degrees
except:
    print("Could not parse arguments")
    Nholes = 4
    dhole = 1.0
    wsteg = 0.25
    e0 = 0.02  # Fine mesh size
    e1 = 0.8  # Coarse mesh size
    hole_angle_deg = 45  # Default opening angle of the holes in degrees

# Convert angle to radians
hole_angle_rad = math.radians(hole_angle_deg)

# Derived geometry parameters
w_cell = dhole + wsteg
h_cell = w_cell
l0 = (Nholes + 2) * w_cell
h0 = 7.0  # Fixed height for crack jump

filename = os.path.join(script_path, script_name_without_extension)
mesh_info = True

# Create geometry with pygmsh
geom = pygmsh.occ.Geometry()
model = geom.__enter__()

# Outer matrix definition
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

# Generate unit cells with holes
for n in range(Nholes):
    x_center = w_cell + w_cell / 2.0 + n * w_cell
    y_center = 0.0
    left_bottom_rectangle = [x_center - w_cell / 2.0, -h_cell / 2, 0]
    cell_matrix_tmp = model.add_rectangle(left_bottom_rectangle, w_cell, h_cell, 0)
    eff_matr = model.boolean_fragments(eff_matr, cell_matrix_tmp, delete_first=True, delete_other=True)

    # Define diamond hole based on opening angle
    half_length = dhole / 2
    dx = half_length * math.cos(hole_angle_rad / 2)
    dy = half_length * math.sin(hole_angle_rad / 2)

    diamond_vertices = [
        [x_center, y_center - dy, 0],  # Bottom vertex
        [x_center + dx, y_center, 0],  # Right vertex
        [x_center, y_center + dy, 0],  # Top vertex
        [x_center - dx, y_center, 0],  # Left vertex
    ]
    diamond_points = [model.add_point(vertex, e0) for vertex in diamond_vertices]
    diamond_lines = [
        model.add_line(diamond_points[i], diamond_points[(i + 1) % 4])
        for i in range(4)
    ]
    diamond_loop = model.add_curve_loop(diamond_lines)
    diamond_surface = model.add_plane_surface(diamond_loop)

    # Subtract diamond hole
    eff_matr = model.boolean_difference(eff_matr, diamond_surface)

# Crack definition
p8 = model.add_point([0.0, 0.0])
p9 = model.add_point([l0, 0.0])
crack = model.add_line(p8, p9)

# Mesh refinement fields
gmsh.model.mesh.field.add("Distance", 1)
hole_centers = [[w_cell + w_cell / 2.0 + n * w_cell, 0.0, 0.0] for n in range(Nholes)]
hole_points = [model.add_point(center, e0)._id for center in hole_centers]

crack_points = [
    model.add_point([i * l0 / 50, 0.0, 0.0], e0)._id
    for i in range(51)
]
refinement_points = hole_points + crack_points
gmsh.model.mesh.field.setNumbers(1, "NodesList", refinement_points)

gmsh.model.mesh.field.add("Distance", 2)
gmsh.model.mesh.field.setNumbers(2, "EdgesList", [crack._id])

gmsh.model.mesh.field.add("Min", 3)
gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

gmsh.model.mesh.field.add("Threshold", 4)
gmsh.model.mesh.field.setNumber(4, "InField", 3)
gmsh.model.mesh.field.setNumber(4, "SizeMin", e0)
gmsh.model.mesh.field.setNumber(4, "SizeMax", e1)
gmsh.model.mesh.field.setNumber(4, "DistMin", dhole / 2)
gmsh.model.mesh.field.setNumber(4, "DistMax", w_cell)

gmsh.model.mesh.field.setAsBackgroundMesh(4)

# Assign physical groups
model.synchronize()
model.add_physical(eff_matr[Nholes], 'eff_matrix')
cells = eff_matr[0:(Nholes)]
model.add_physical(cells, "cell_matrix")

model.add_physical(crack, 'crack')

# Generate and save mesh
model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + ".msh")
gmsh.clear()
model.__exit__()

# Post-process mesh
mesh = meshio.read(filename + ".msh")
nodes = mesh.points[:, :2]
elems = mesh.get_cells_type("triangle")
elem_data = mesh.cell_data_dict["gmsh:physical"]["triangle"]-1

def filter_points_and_update_cells(cell_array, point_array):
    referenced_ids = {point_id for triangle in cell_array for point_id in triangle}
    filtered_points = [point_array[i] for i in referenced_ids]
    mapping = {old: new for new, old in enumerate(referenced_ids)}
    updated_cells = [[mapping[pt] for pt in triangle] for triangle in cell_array]
    return np.array(filtered_points), np.array(updated_cells)

nodes, elems = filter_points_and_update_cells(elems, nodes)

# Save final mesh
cell_mesh = meshio.Mesh(points=nodes, cells={"triangle": elems}, cell_data={"name_to_read": [elem_data]})
meshio.write(filename + ".xdmf", cell_mesh)

# Save parameters
params = {"nholes": Nholes, "dhole": dhole, "wsteg": wsteg, "e0": e0, "e1": e1, "hole_angle": hole_angle_deg}
alex.postprocessing.append_to_file(parameter_path, params)







