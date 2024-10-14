import pygmsh
import meshio
import numpy as np
import gmsh
import os
import argparse
import math 


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--l0", type=float, required=True, help="l0")
    parser.add_argument("--wsteg", type=float, required=True, help="width steg")
    parser.add_argument("--hsteg", type=float, required=True, help="height steg")
    parser.add_argument("--whole", type=float, required=True, help="width of  holes")
    args = parser.parse_args()
    l0 = args.l0
    wsteg = args.wsteg
    hsteg = args.hsteg
    dhole = args.whole
except:
    print("Could not parse arguments")
    l0 = 4.0
    wsteg = 0.4
    hsteg = 0.5
    dhole = 0.6
    
parameters_to_write = {
            'l0': l0,
            'wsteg': wsteg,
            'hsteg': hsteg,
            'whole': dhole,
        }
# store parameters
def append_to_file(filename, parameters):
    with open(filename, 'w') as file:
        for key, value in parameters.items():
            file.write(f"{key}={value}\n")



# l0 = 4.0                 # length rve
# w_steg = 0.4             # width steg 
# h_steg = 0.25
percentage_hetero = 0.5
N_holes = math.floor(percentage_hetero*(l0) / (dhole+wsteg) ) # number of holes   
h0 = 2.0                 # height (edges at +/- h0/2)
a0 = l0 / 32             # initial crack length
# whole = 1/4*l0-wsteg   # width of rectangular hole
rect_h = hsteg          # height of rectangular hole
e0 = 0.02                # mesh size
offset = 0               # offset 0 / 0.15*h0
mesh_inclusions = True   # mesh inclusions


filename = script_path + "/" + script_name_without_extension
mesh_info = True

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

# Create rectangular inclusions (holes)
holes = []
# Store x-coordinates for each hole
hole_coordinates = []
for n in range(0,N_holes):
    c = [(1.0-percentage_hetero)/2.0 * l0 + n* (dhole+wsteg),0.0]
    r = dhole    
    hole =model.add_disk(c,r ) # model.add_circle_arc(c,sp,ep)
    # hole = model.add_rectangle([(1.0-percentage_hetero)/2.0 * l0 + n* (whole+wsteg), offset - rect_h / 2, 0], whole, rect_h, 0)
    holes.append(hole)
    start_x = (1.0 - percentage_hetero) / 2.0 * l0 + n * (dhole + wsteg) - dhole / 2
    end_x = start_x + dhole
    hole_coordinates.append((start_x, end_x))
    

# Add hole coordinates to parameters
for idx, (start_x, end_x) in enumerate(hole_coordinates):
    parameters_to_write[f'hole_{idx+1}_start_x'] = start_x
    parameters_to_write[f'hole_{idx+1}_end_x'] = end_x
# hole1 = model.add_rectangle([0.25 * l0 - whole / 2, offset - rect_h / 2, 0], whole, rect_h, 0)  # 1
# hole2 = model.add_rectangle([0.5 * l0 - whole / 2, -offset - rect_h / 2, 0], whole, rect_h, 0)  # 2
# hole3 = model.add_rectangle([0.75 * l0 - whole / 2, offset - rect_h / 2, 0], whole, rect_h, 0)  # 3

# Define outer rectangular boundary
p0 = model.add_point([0, -h0 / 2])
p1 = model.add_point([l0, -h0 / 2])
p2 = model.add_point([l0, h0 / 2])
p3 = model.add_point([0, h0 / 2])
p4 = model.add_point([0.0, 0.0])
p5 = model.add_point([a0, 0.0])

line0 = model.add_line(p0, p1)            # 1 (bottom)
line1 = model.add_line(p1, p2)            # 2 (right)
line2 = model.add_line(p2, p3)            # 3 (top)
line3 = model.add_line(p3, p0)            # 4 (left)

outer = model.add_curve_loop([line0, line1, line2, line3])
matr = model.add_plane_surface(outer)

# Subtract the rectangular holes from the outer surface
matr = model.boolean_difference(matr, holes, delete_first=True, delete_other=True)

# Add crack as a line
crack = model.add_line(p4, p5)  # 4 (crack)
matr = model.boolean_fragments(matr, crack, delete_first=True, delete_other=True)

model.synchronize()

# Ensure we are adding the correct surfaces (2D entities) to the physical group
surfaces = [m for m in matr if m.dim == 2]  # Extract only 2D surfaces
if mesh_inclusions:
    model.add_physical(surfaces, 'matrix')  # Add the outer surface as 'matrix'

# Add the crack (a 1D line) as a physical group
model.add_physical(crack, 'crack')

# Generate mesh
model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + '.msh')
gmsh.clear()
model.__exit__()

# Read and process the mesh
mesh = meshio.read(filename + '.msh')

nodes = mesh.points[:, 0:2]
elems = mesh.get_cells_type('triangle')
elem_data = mesh.cell_data_dict['gmsh:physical']['triangle'] - 1  # matr.: 0 /incl.: 1

if mesh_info:
    print('NODES:')
    print(nodes)
    print('ELEMENTS')
    print(elems)
    print('ELEMENT DATA')
    print(elem_data)

# Create cell mesh
cell_mesh = meshio.Mesh(points=nodes, cells={'triangle': elems}, cell_data={'name_to_read': [elem_data]})
meshio.write(os.path.join(script_path, script_name_without_extension + '.xdmf'), cell_mesh)

parameter_path = os.path.join(script_path,"parameters.txt")
append_to_file(parameters=parameters_to_write,filename=parameter_path)

