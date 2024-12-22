import pygmsh
import meshio
import numpy as np
import gmsh
import os
import argparse
import alex.os


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser(description="Run a simulation with specified parameters and organize output files.")
try:
    parser.add_argument("--dhole", type=float, required=True, help="Diameter of hole")
    parser.add_argument("--wsteg", type=float, required=True, help="width of steg")
    parser.add_argument("--e0", type=float, required=True, help="size of elements")
    args = parser.parse_args()
    dhole=args.dhole
    wsteg=args.wsteg
    e0 = args.e0                # mesh size
    # }
except:
    print("Could not parse arguments")
    dhole=0.25
    wsteg=0.1
    e0 = 0.02                # mesh size

w_cell=dhole+wsteg
h_cell=w_cell
h0 = 2*w_cell

mesh_inclusions = True  # mesh inclusions 

filename = script_path + "/" + script_name_without_extension
mesh_info = True

geom = pygmsh.occ.Geometry()
model = geom.__enter__()
model.characteristic_length_min = e0
model.characteristic_length_max = e0


n=0
x_center = w_cell + w_cell / 2.0 + n*w_cell
y_center = 0.0
left_bottom_rectangle = [x_center - w_cell/2.0,-h_cell/2,0]
matrix = model.add_rectangle(left_bottom_rectangle,w_cell,h_cell,0)

hole = model.add_disk([x_center, y_center],dhole/2)
matrix = model.boolean_difference(matrix,hole)
    
model.synchronize()
model.add_physical(matrix[0], 'matrix')

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
meshio.write(os.path.join(script_path,script_name_without_extension+'.xdmf'), cell_mesh)



