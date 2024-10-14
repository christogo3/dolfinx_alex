import pygmsh
import meshio
import numpy as np
import gmsh
import os

import alex.os


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]


# l0 = 4.0                 # length rve    
# h0 = 1.0                 # height (edges at +/- h0/2)
# a0 = l0/32               # inital crack length
# r0 = 0.25*h0             # radius of inclusion

# offset = 0               # offset 0 / 0.15*h0


Nholes=3
dhole=0.25
wsteg=0.1
w_cell=dhole+wsteg
h_cell=w_cell
e0 = 0.02                # mesh size

l0 = (Nholes +2) * w_cell
h0 = 2*w_cell

mesh_inclusions = True  # mesh inclusions 

filename = script_path + "/" + script_name_without_extension
mesh_info = True

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0

# generate surrounding matrix 
p0 = model.add_point([0, -h0/2])          
p1 = model.add_point([l0, -h0/2])         
p2 = model.add_point([l0, +h0/2])         
p3 = model.add_point([0, h0/2])   
line0 = model.add_line(p0, p1)            # 1 (bottom)
line1 = model.add_line(p1, p2)            # 2 (right)
line2 = model.add_line(p2, p3)            # 3 (top)
line3 = model.add_line(p3, p0)            # 3 (left)
outer = model.add_curve_loop([line0, line1, line2, line3])
eff_matr = model.add_plane_surface(outer)

p4 = model.add_point([0+w_cell, -h_cell/2])          
p5 = model.add_point([l0-w_cell,-h_cell/2])         
p6 = model.add_point([l0-w_cell,+h_cell/2])         
p7 = model.add_point([0+w_cell, h_cell/2])
line4 = model.add_line(p4, p5)            # 1 (bottom)
line5 = model.add_line(p5, p6)            # 2 (right)
line6 = model.add_line(p6, p7)            # 3 (top)
line7 = model.add_line(p7, p4)            # 3 (left)
tmp_inner = model.add_curve_loop([line4, line5, line6, line7])
rect_hole = model.add_plane_surface(tmp_inner)

eff_matr = model.boolean_difference(eff_matr,rect_hole)
model.synchronize()


# generate quadratic unit cells with circular pores
cells = []
for n in range(0,Nholes):
    x_center = w_cell + w_cell / 2.0 + n*w_cell
    y_center = 0.0
    left_bottom_rectangle = [x_center - w_cell/2.0,-h_cell/2,0]
    cell_matrix_tmp = model.add_rectangle(left_bottom_rectangle,w_cell,h_cell,0)
    hole = model.add_disk([x_center, y_center],dhole/2)
    cell_matrix = model.boolean_difference(cell_matrix_tmp,hole)
    eff_matr = model.boolean_union([eff_matr,cell_matrix],delete_first=True, delete_other=True)
    # cell_matrix = model.boolean_fragments(cell_matrix_tmp,hole)
    cells.append(cell_matrix)

# Add crack as a line
p8 = model.add_point([0.0,0.0])         
p9 = model.add_point([w_cell, 0.0])
crack = model.add_line(p8, p9)  # 4 (crack)
eff_matr = model.boolean_fragments(eff_matr, crack, delete_first=True, delete_other=True)


model.add_physical(eff_matr[0], 'eff_matrix')
# model.add_physical(cells,'cell_matrix')
# Ensure we are adding the correct surfaces (2D entities) to the physical group
surfaces = [c[0] for c in cells if c[0].dim == 2]  # Extract only 2D surfaces
if mesh_inclusions:
    model.add_physical(surfaces, 'cell_matrix')  # Add the outer surface as 'matrix'



# Add the crack (a 1D line) as a physical group
model.add_physical(crack, 'crack')

model.generate_mesh(dim=2, verbose=True)
gmsh.write(filename+'.msh')
gmsh.clear()
model.__exit__()

mesh = meshio.read(filename+'.msh')

nodes = mesh.points[:, 0:2] 
elems = mesh.get_cells_type('triangle')
elem_data = mesh.cell_data_dict['gmsh:physical']['triangle']-1   # matr.: 0 /incl.: 1

if mesh_info:
    print('NODES:')
    print(nodes)
    print('ELEMENTS')
    print(elems)
    print('ELEMENT DATA')
    print(elem_data)

cell_mesh = meshio.Mesh(points=nodes, cells={'triangle': elems}, cell_data={'name_to_read': [elem_data]})
# print(os.path.join(alex.os.resources_directory,script_name_without_extension+'.xdmf'))
meshio.write(os.path.join(script_path,script_name_without_extension+'.xdmf'), cell_mesh)



