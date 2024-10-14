import pygmsh
import meshio
import numpy as np
import gmsh
import os

import alex.os

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

l0 = 4.0                 # length rve    
h0 = 1.0                 # height (edges at +/- h0/2)
a0 = l0/32               # initial crack length
rect_w = 0.5             # width of rectangular inclusion
rect_h = 0.25            # height of rectangular inclusion
e0 = 0.02                # mesh size
offset = 0               # offset 0 / 0.15*h0
mesh_inclusions = True   # mesh inclusions

filename = script_path + "/" + script_name_without_extension
mesh_info = True

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

# Create rectangular inclusions
incl1 = model.add_rectangle([0.25 * l0 - rect_w / 2, offset - rect_h / 2, 0], rect_w, rect_h, 0)  # 1
incl2 = model.add_rectangle([0.5 * l0 - rect_w / 2, -offset - rect_h / 2, 0], rect_w, rect_h, 0)  # 2
incl3 = model.add_rectangle([0.75 * l0 - rect_w / 2, offset - rect_h / 2, 0], rect_w, rect_h, 0)  # 3

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

# Subtract the rectangular inclusions from the outer surface
if mesh_inclusions:
    matr = model.boolean_fragments(matr, incl1, delete_first=True, delete_other=True)
    matr = model.boolean_fragments(matr, incl2, delete_first=True, delete_other=True)
    matr = model.boolean_fragments(matr, incl3, delete_first=True, delete_other=True)
else:
    matr = model.boolean_difference(matr, incl1)
    matr = model.boolean_difference(matr, incl2)
    matr = model.boolean_difference(matr, incl3)

# Add crack
crack = model.add_line(p4, p5)  # 4 (crack)
matr = model.boolean_fragments(matr, crack, delete_first=True, delete_other=True)

model.synchronize()

if mesh_inclusions:
    model.add_physical(matr[3], 'matrix')  # name_to_read 1
    model.add_physical([matr[0], matr[1], matr[2]], 'rectangles')  # name to read 2
else:
    model.add_physical(matr[0], 'matrix')  # name_to_read 1

model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename + '.msh')
gmsh.clear()
model.__exit__()

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

cell_mesh = meshio.Mesh(points=nodes, cells={'triangle': elems}, cell_data={'name_to_read': [elem_data]})
meshio.write(os.path.join(script_path, script_name_without_extension + '.xdmf'), cell_mesh)
