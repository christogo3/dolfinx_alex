from typing import Callable, Union
import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import pyvista #pip install "pyvista<0.42.0"
import dolfinx.plot as plot
import matplotlib.pyplot as plt
import glob 
import os
import sys
import basix
from petsc4py.PETSc import ScalarType

from petsc4py import PETSc as petsc

import alex.os
import alex.util as ut
import alex.imageprocessing as img

import numpy as np
from mpi4py import MPI



def write_meshoutputfile(domain: dlfx.mesh.Mesh,
                                       outputfile_path: str,
                                       comm: MPI.Intercomm,
                                       meshtags: any = None):
    
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'w') as xdmfout:
            xdmfout.write_mesh(domain)
            if( not meshtags is None):
                xdmfout.write_meshtags(meshtags, domain.geometry)
            xdmfout.close()
        # xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    # xdmfout.write_function()
        # xdmfout.write_function(field, t)
            # xdmfout.write_function(field_interp, t)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'w') as vtkout:
            vtkout.write_mesh(domain)
            # vtkout.write_function(field_interp,t)
    # xdmfout = dlfx.io.XDMFFile(comm, outputfile_path, 'w')
    else:
        return False
    
    return True

# def write_meshoutputfile_vtk(domain: dlfx.mesh.Mesh,
#                                        outputfile_vtk_path: str,
#                                        comm: MPI.Intercomm,):
#     with dlfx.io.VTKFile(comm, outputfile_vtk_path, 'w') as vtk_out:
#         vtk_out.write_mesh(domain,0.0)

def write_phasefield_mixed_solution(domain: dlfx.mesh.Mesh,
                                    outputfile_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    u, s = w.split() 
    
    u.name = "u"
    write_vector_field(domain,outputfile_path,u,t,comm)
    
    s.name = "s"
    write_field(domain,outputfile_path,s,t,comm)
    
def write_phasefield_mixed_solution_lagrange(domain: dlfx.mesh.Mesh,
                                    outputfile_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    u, s, _, _ = w.split() 
    
    u.name = "u"
    write_vector_field(domain,outputfile_path,u,t,comm)
    
    s.name = "s"
    write_field(domain,outputfile_path,s,t,comm)   
    


def write_field(domain: dlfx.mesh.Mesh,
                                    outputfile_path: str,
                                    field: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm,
                                    S: dlfx.fem.FunctionSpace = None,
                                    field_interp: dlfx.fem.Function = None) :
    
    # Se = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=2, quad_scheme="default")
    if S is None and field_interp is None:
        Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
        S = dlfx.fem.functionspace(domain, Se)
        
    if field_interp is None:
        field_interp = dlfx.fem.Function(S)
    
    interpolate_to_vertices_for_output(field, S, field_interp)
    
    write_to_output_file(outputfile_path, t, comm, field_interp)

def write_to_output_file(outputfile_path, t, comm, field_interp):
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'a') as xdmfout:
        # xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    # xdmfout.write_function()
        # xdmfout.write_function(field, t)
            xdmfout.write_function(field_interp, t)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'a') as vtkout:
            vtkout.write_function(field_interp,t)
    

def interpolate_to_vertices_for_output(field, S, field_interp):
    # def is_quadrature_element(element):
    # # https://github.com/michalhabera/dolfiny/blob/master/dolfiny/interpolation.py
    #     return element._family == "Quadrature"

    # if is_quadrature_element(field.function_space.ufl_element()): # quadrature elements need to be interpolated via expression
    expr = dlfx.fem.Expression(field,S.element.interpolation_points())
    field_interp.interpolate(expr)
    # else:
    #     field_interp.interpolate(field) 
    
    # expr = dlfx.fem.Expression(field,S.element.interpolation_points())
    # field_interp.interpolate(expr)
    field_interp.name = field.name
    
    
def compute_and_write_tensor_eigenvalue(domain: dlfx.mesh.Mesh,
                                        tensor: ufl.tensors.ListTensor,
                                        tensor_name: str,
                                        time: dlfx.fem.Constant,
                                        outputfile_path: str,
                                        comm: MPI.Intercomm):
    """
    Compute and write the three real eigenvalues of a symmetric 3x3 tensor field.
    Based on real-only formulation of Cardano's method.
    """

    a = ScalarType(-1)
    b = ufl.tr(tensor)
    c = -0.5 * (ufl.tr(tensor)**2 - ufl.tr(tensor * tensor))
    d = ufl.det(tensor)

    # Cardano's formula parameters
    q = (3 * a * c - b**2) / (9 * a**2)
    r = (9 * a * b * c - 27 * a**2 * d - 2 * b**3) / (54 * a**3)

    # Clamp acos argument to [-1, 1] to avoid NaNs
    cos_arg = r / ufl.sqrt(-q**3)
    cos_arg = ufl.max_value(ufl.min_value(cos_arg, 1.0), -1.0)
    theta = ufl.acos(cos_arg)
    sqrt_q = ufl.sqrt(-q)

    eig1 = 2 * sqrt_q * ufl.cos(theta / 3) - b / (3 * a)
    eig2 = 2 * sqrt_q * ufl.cos((theta + 2 * np.pi) / 3) - b / (3 * a)
    eig3 = 2 * sqrt_q * ufl.cos((theta + 4 * np.pi) / 3) - b / (3 * a)

    # Scalar function space
    element = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    function_space = dlfx.fem.functionspace(domain, element)

    for i, eig in enumerate([eig1, eig2, eig3]):
        expr = dlfx.fem.Expression(eig, function_space.element.interpolation_points())
        fn = dlfx.fem.Function(function_space)
        fn.interpolate(expr)
        fn.name = f"{tensor_name}_{i}"
        write_field(domain, outputfile_path, fn, time, comm, function_space)


def write_vector_field(domain: dlfx.mesh.Mesh,
                                    outputfile_path: str,
                                    field: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm,
                                    V: dlfx.fem.FunctionSpace = None,
                                    field_interp: dlfx.fem.Function = None) :
    
    if V is None and field_interp is None:
        Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        V = dlfx.fem.functionspace(domain, Ve)
        # Ve = ufl.VectorElement('CG', domain.ufl_cell(), 1)  
        # V = dlfx.fem.FunctionSpace(domain, Ve)
        
    if field_interp is None:
        field_interp = dlfx.fem.Function(V)
        
    # Ve = ufl.VectorElement('CG', domain.ufl_cell(), 1)
    # V = dlfx.fem.FunctionSpace(domain, Ve)
    
    # field_interp = dlfx.fem.Function(V)
    
    interpolate_to_vertices_for_output(field, V, field_interp)

    # field_interp.name = field.name
    write_to_output_file(outputfile_path, t, comm, field_interp)


    
def write_tensor_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, tensor_fields_as_functions, tensor_field_names, outputfile_xdmf_path: str, t: float):
    dim = domain.topology.dim
    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        for n  in range(0,len(tensor_fields_as_functions)):
            tensor_field_function = tensor_fields_as_functions[n]
            #tensor_field_function.function_space
            tensor_field_name = tensor_field_names[n]
            tensor_field_expression = dlfx.fem.Expression(tensor_field_function, 
                                                         TEN.element.interpolation_points())
            out_tensor_field = dlfx.fem.Function(TEN) 
            out_tensor_field.interpolate(tensor_field_expression)
            out_tensor_field.name = tensor_field_name
            
            xdmf_out.write_function(out_tensor_field,t)

def write_vector_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, vector_fields_as_functions, vector_field_names, outputfile_xdmf_path: str, t: float):
    Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = dlfx.fem.functionspace(domain, Ve)
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(vector_fields_as_functions)):
            vector_field_function = vector_fields_as_functions[n]
            vector_field_name = vector_field_names[n]
            vector_field_expression = dlfx.fem.Expression(vector_field_function, 
                                                        V.element.interpolation_points())
            out_vector_field = dlfx.fem.Function(V)
            out_vector_field.interpolate(vector_field_expression)
            out_vector_field.name = vector_field_name
            
            xdmf_out.write_function(out_vector_field,t)
    xdmf_out.close()
            
def write_scalar_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, scalar_fields_as_functions, scalar_field_names, outputfile_xdmf_path: str, t: float):
    Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    S = dlfx.fem.functionspace(domain, Se)
    # S= dlfx.fem.functionspace(domain, ("DP", 0, ()))
    xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    for n  in range(0,len(scalar_fields_as_functions)):
            scalar_field_function = scalar_fields_as_functions[n]
            scalar_field_name = scalar_field_names[n]
            scalar_field_expression = dlfx.fem.Expression(scalar_field_function, 
                                                        S.element.interpolation_points())
            out_scalar_field = dlfx.fem.Function(S)
            out_scalar_field.interpolate(scalar_field_expression)
            out_scalar_field.name = scalar_field_name
            
            xdmf_out.write_function(out_scalar_field,t)
    xdmf_out.close()
    
def get_extreme_values_of_scalar_field(domain: dlfx.mesh.Mesh, scalar_field_function: any, comm: MPI.Intercomm):
    S= dlfx.fem.functionspace(domain, ("DP", 0, ()))
    scalar_field_expression = dlfx.fem.Expression(scalar_field_function, 
                                                        S.element.interpolation_points())
    scalar_field = dlfx.fem.Function(S) 
    scalar_field.interpolate(scalar_field_expression)
    return comm.reduce(np.max(scalar_field.x.array), MPI.MAX), comm.reduce(np.min(scalar_field.x.array), MPI.MIN)
    # scalar_field.x.array


def tag_part_of_boundary(domain: dlfx.mesh.Mesh, where: Callable, tag: int) -> any:
    '''
        https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html
        assigns tags to part of the boundary, which can then be used for surface integral
        
        returns the facet_tags
    '''
    fdim = domain.topology.dim - 1
    facets = dlfx.mesh.locate_entities_boundary(domain, fdim, where)
    marked_facets = np.hstack([facets])
    marked_values = np.hstack([np.full_like(facets,tag)])
    sorted_facets = np.argsort(marked_facets)
    facet_tags = dlfx.mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
    
    
    return facet_tags
    
        
     
# configurational forces
   
def ufl_integration_subdomain(domain: dlfx.mesh.Mesh , where: Callable) -> Union[ufl.Measure, any]:
    """
        tags all cells ( 0 not in subdomain, 1 in subdomain)
        returns an integration measure for subdomain
    """
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    midpoints = dlfx.mesh.compute_midpoints(domain, domain.topology.dim, np.arange(num_cells, dtype=np.int32))
    cell_tags = dlfx.mesh.meshtags(domain, domain.topology.dim, np.arange(num_cells), where(midpoints))
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    return dx, cell_tags

def screenshot_of_subdomain(path: str, domain: dlfx.mesh.Mesh, cell_tags: any, t: float):
    """
        plots the ufl integration subdomain (marked with cell_tags = 1)
    """
    
    pyvista.start_xvfb()
    
    # Create VTK mesh
    cells, types, x = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvista grid
    grid.cell_data["Marker"] = cell_tags.values
    grid.set_active_scalars("Marker")

    # Create a plotter with two subplots, and add mesh tag plot to the
    # first sub-window
    subplotter = pyvista.Plotter(off_screen=True, shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Mesh with markers", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.show_axes()
    
    # We can visualize subsets of data, by creating a smaller topology
    # (set of cells). Here we create VTK mesh data for only cells with
    # that tag '1'.
    cells, types, x = plot.vtk_mesh(domain, entities=cell_tags.find(1))

    # Add this grid to the second plotter window
    sub_grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 1)
    subplotter.add_text("Subset of mesh", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")
    subplotter.show_axes()
    subplotter.screenshot(
            path + "/2D_markers"+str(t)+".png", transparent_background=False, window_size=[2 * 800, 800]
    )

def getJString(Jx, Jy, Jz):
    out_string = 'Jx: {0:.4e} Jy: {1:.4e} Jz: {2:.4e}'.format(Jx, Jy, Jz)
    return out_string

def getJString2D(Jx, Jy):
    out_string = 'Jx: {0:.4e} Jy: {1:.4e}'.format(Jx, Jy)
    return out_string


def prepare_graphs_output_file(output_file_path: str):
    for file in glob.glob(output_file_path):
        os.remove(output_file_path)
    logfile = open(output_file_path, 'w')  
    logfile.write('# This is a general outputfile for displaying scalar quantities vs time, first column is time, further columns are data \n')
    logfile.close()
    return True

# def write_to_J_output_file(output_file_path: str, t:float, Jx:float, Jy:float, Jz:float):
#     logfile = open(output_file_path, 'a')
#     logfile.write('{0:.4e} {1:.4e} {2:.4e} {3:.4e}\n'.format(t, Jx, Jy, Jz))
#     logfile.close()
#     return True

# def write_to_J_output_file_extended(output_file_path: str, t:float, Jx:float, Jy:float, Jz:float, Jx_alt:float, Jy_alt:float, Jz_alt:float):
#     logfile = open(output_file_path, 'a')
#     logfile.write('{0:.4e} {1:.4e} {2:.4e} {3:.4e} {4:.4e} {5:.4e} {6:.4e}\n'.format(t, Jx, Jy, Jz, Jx_alt, Jy_alt, Jz_alt))
#     logfile.close()
#     return True

def write_to_graphs_output_file(output_file_path: str, *args):
    logfile = open(output_file_path, 'a')
    formatted_data = ' '.join(['{:.4e}' for _ in range(len(args))])
    logfile.write(formatted_data.format(*args) + '\n')
    logfile.close()
    return True


import matplotlib.pyplot as plt



# def print_J_plot(output_file_path, print_path):
#     def read_from_J_output_file(output_file_path):
#         with open(output_file_path, 'r') as file:
#             data = [line for line in file.readlines() if not line.startswith('#')]
#         return data
    
#     data = read_from_J_output_file(output_file_path)
    
#     t_values = []
#     Jx_values = []
#     Jy_values = []
#     Jz_values = []

#     for line in data:
#         t, Jx, Jy, Jz = map(float, line.strip().split())
#         t_values.append(t)
#         Jx_values.append(Jx)
#         Jy_values.append(Jy)
#         Jz_values.append(Jz)

#     plt.plot(t_values, Jx_values, label='Jx')
#     plt.plot(t_values, Jy_values, label='Jy')
#     plt.plot(t_values, Jz_values, label='Jz')
#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.title('Jx, Jy, Jz vs Time')
#     plt.legend()
#     plt.savefig(print_path + '/J.png') 
#     plt.close()
    
import matplotlib.pyplot as plt

def print_graphs_plot(output_file_path, print_path, legend_labels=None, default_label="Column", filename=None):
    def read_from_graphs_output_file(output_file_path):
        with open(output_file_path, 'r') as file:
            data = [line.strip().split() for line in file.readlines() if not line.startswith('#')]
        return data
    
    data = read_from_graphs_output_file(output_file_path)
    
    t_values = []
    column_values = [[] for _ in range(len(data[0]) - 1)]  # Initialize lists for column values
    
    for line in data:
        t_values.append(float(line[0]))
        for i in range(1, len(line)):  # Start from index 1 to skip time column
            column_values[i - 1].append(float(line[i]))

    if legend_labels is None:
        legend_labels = [default_label + str(i + 1) for i in range(len(column_values))]
    elif len(legend_labels) < len(column_values):
        legend_labels += [default_label + str(i + 1) for i in range(len(legend_labels), len(column_values))]

    for i, values in enumerate(column_values):
        plt.plot(t_values, values, label=legend_labels[i])

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Columns vs Time')
    plt.legend()

    # Use provided filename or default to 'graphs.png'
    save_filename = filename if filename else 'graphs.png'
    plt.savefig(f"{print_path}/{save_filename}")
    plt.close()

    


def number_of_nodes(domain: dlfx.mesh.Mesh):
    return domain.topology.index_map(0).size_global

            
# Tracking ###############################    
        
def crack_bounding_box_3D(domain: dlfx.mesh.Mesh, crack_locator_function: Callable, comm: MPI.Intracomm):
    '''
    operates on nodes not on DOF locations
    
    crack locator function returns a boolean array that determines whether a crack is present or not
    
    returns the bounding box in which all cracks are contained
    
    '''
    
    try:
        xx  = np.array(domain.geometry.x).T
        crack_indices = crack_locator_function(xx)
        crack_x = xx.T[crack_indices]
       
        # if rank == 8:
        #     print("RANK:" + str(rank))
        #     print("SHAPE OF XX" + str(xx.shape))
        #     print("SHAPE OF CRACK INDICES" + str(crack_indices.shape))
        #     print("SHAPE OF CRACK X" + str(crack_x.shape))
        
        if(len(crack_x) != 0):
            max_x = np.max(crack_x.T[0])
            max_y = np.max(crack_x.T[1])
            max_z = np.max(crack_x.T[2])

            min_x = np.min(crack_x.T[0])
            min_y = np.min(crack_x.T[1])
            min_z = np.min(crack_x.T[2])
        else:
            val_fail = -sys.float_info.max
            max_x = val_fail
            max_y = val_fail
            max_z = val_fail

            min_x = -val_fail
            min_y = -val_fail
            min_z = -val_fail 
    except Exception as e:
        raise Exception(e) 
    return comm.allreduce(max_x, op=MPI.MAX), comm.allreduce(max_y, op=MPI.MAX), comm.allreduce(max_z, op=MPI.MAX), comm.allreduce(min_x, op=MPI.MIN), comm.allreduce(min_y, op=MPI.MIN), comm.allreduce(min_z, op=MPI.MIN)

def crack_bounding_box_2D(domain: dlfx.mesh.Mesh, crack_locator_function: Callable, comm: MPI.Intracomm):
    '''
    operates on nodes not on DOF locations
    
    crack locator function returns a boolean array that determines whether a crack is present or not
    
    returns the bounding box in which all cracks are contained
    
    '''
    
    try:
        xx  = np.array(domain.geometry.x).T
        crack_indices = crack_locator_function(xx)
        crack_x = xx.T[crack_indices]
       
        # if rank == 8:
        #     print("RANK:" + str(rank))
        #     print("SHAPE OF XX" + str(xx.shape))
        #     print("SHAPE OF CRACK INDICES" + str(crack_indices.shape))
        #     print("SHAPE OF CRACK X" + str(crack_x.shape))
        
        if(len(crack_x) != 0):
            max_x = np.max(crack_x.T[0])
            max_y = np.max(crack_x.T[1])

            min_x = np.min(crack_x.T[0])
            min_y = np.min(crack_x.T[1])
        else:
            val_fail = -sys.float_info.max
            max_x = val_fail
            max_y = val_fail

            min_x = -val_fail
            min_y = -val_fail
    except Exception as e:
        raise Exception(e) 
    return comm.allreduce(max_x, op=MPI.MAX), comm.allreduce(max_y, op=MPI.MAX), comm.allreduce(min_x, op=MPI.MIN), comm.allreduce(min_y, op=MPI.MIN)



# def crack_bounding_box(domain, crack_locator_function, comm):
#     try:
#         xx = np.array(domain.geometry.x).T
#         crack_indices = crack_locator_function(xx)
#         crack_x = xx.T[crack_indices]

#         if len(crack_x) != 0:
#             max_coords = np.max(crack_x, axis=0)
#             min_coords = np.min(crack_x, axis=0)
#         else:
#             val_fail = -np.finfo(float).max
#             max_coords = np.array([val_fail, val_fail, val_fail])
#             min_coords = -max_coords
#     except Exception as e:
#         raise Exception(e)

#     # Find global max/min over all MPI processes
#     max_global_coords = comm.allreduce(max_coords, op=MPI.MAX)
#     min_global_coords = comm.allreduce(min_coords, op=MPI.MIN)

#     dimensions = len(max_global_coords)

#     if dimensions == 2:
#         return max_global_coords[0], max_global_coords[1], \
#                min_global_coords[0], min_global_coords[1]
#     elif dimensions == 3:
#         return max_global_coords[0], max_global_coords[1], max_global_coords[2], \
#                min_global_coords[0], min_global_coords[1], min_global_coords[2]
#     else:
        # raise ValueError("Unsupported number of dimensions: {}".format(dimensions))




def get_s_zero_field_for_tracking(domain):
    Se = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) 
    S = dlfx.fem.FunctionSpace(domain,Se)
    s_zero_for_tracking = dlfx.fem.Function(S)
    c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
    sub_expr = dlfx.fem.Expression(c,S.element.interpolation_points())
    s_zero_for_tracking.interpolate(sub_expr)
    return s_zero_for_tracking

# ##################################


def compute_bounding_box(comm, domain):
    # get dimension and bounds for each mpi process
    dimensions = domain.geometry.x.shape[1]
    
    if dimensions == 2:
        x_min = np.min(domain.geometry.x[:, 0]) 
        x_max = np.max(domain.geometry.x[:, 0])   
        y_min = np.min(domain.geometry.x[:, 1]) 
        y_max = np.max(domain.geometry.x[:, 1])
        z_min_all = z_max_all = None
    elif dimensions == 3:
        x_min = np.min(domain.geometry.x[:, 0]) 
        x_max = np.max(domain.geometry.x[:, 0])   
        y_min = np.min(domain.geometry.x[:, 1]) 
        y_max = np.max(domain.geometry.x[:, 1])
        z_min = np.min(domain.geometry.x[:, 2]) 
        z_max = np.max(domain.geometry.x[:, 2])
        z_min_all = comm.allreduce(z_min, op=MPI.MIN)
        z_max_all = comm.allreduce(z_max, op=MPI.MAX)
    else:
        raise ValueError("Unsupported number of dimensions: {}".format(dimensions))

    # find global min/max over all mpi processes
    comm.Barrier()
    x_min_all = comm.allreduce(x_min, op=MPI.MIN)
    x_max_all = comm.allreduce(x_max, op=MPI.MAX)
    y_min_all = comm.allreduce(y_min, op=MPI.MIN)
    y_max_all = comm.allreduce(y_max, op=MPI.MAX)
    comm.Barrier()
    return x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all

def print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all=None, z_max_all=None):
    print('x_min, x_max: {}, {}'.format(x_min_all, x_max_all))
    print('y_min, y_max: {}, {}'.format(y_min_all, y_max_all))
    if z_min_all is not None and z_max_all is not None:
        print('z_min, z_max: {}, {}'.format(z_min_all, z_max_all))


# def compute_bounding_box_3D(comm, domain):
#     # get dimension and bounds for each mpi process
#     x_min = np.min(domain.geometry.x[:,0]) 
#     x_max = np.max(domain.geometry.x[:,0])   
#     y_min = np.min(domain.geometry.x[:,1]) 
#     y_max = np.max(domain.geometry.x[:,1])   
#     z_min = np.min(domain.geometry.x[:,2]) 
#     z_max = np.max(domain.geometry.x[:,2])

# # find global min/max over all mpi processes
#     comm.Barrier()
#     x_min_all = comm.allreduce(x_min, op=MPI.MIN)
#     x_max_all = comm.allreduce(x_max, op=MPI.MAX)
#     y_min_all = comm.allreduce(y_min, op=MPI.MIN)
#     y_max_all = comm.allreduce(y_max, op=MPI.MAX)
#     z_min_all = comm.allreduce(z_min, op=MPI.MIN)
#     z_max_all = comm.allreduce(z_max, op=MPI.MAX)
#     comm.Barrier()
#     return x_min_all,x_max_all,y_min_all,y_max_all,z_min_all,z_max_all

# def print_bounding_box_3D(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all):
#     alex.os.mpi_print('x_min, x_max: '+str(x_min_all)+', '+str(x_max_all), rank)
#     alex.os.mpi_print('y_min, y_max: '+str(y_min_all)+', '+str(y_max_all), rank)
#     alex.os.mpi_print('z_min, z_max: '+str(z_min_all)+', '+str(z_max_all), rank)


def volume_of_mesh(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, dx : ufl.Measure):
    vol = dlfx.fem.assemble_scalar(dlfx.fem.form( ( dlfx.fem.Constant(domain,1.0) ) * ufl.dx ))
    return comm.allreduce(vol,MPI.SUM)

def volume_of_mesh_above_threshhold(domain: dlfx.mesh.Mesh, function: dlfx.fem.Function, threshhold: float, comm: MPI.Intercomm, dx : ufl.Measure):
    #S = ut.get_CG_functionspace(domain)
    #switch_function = dlfx.fem.Function(S)
    #img.clip(function,switch_function,threshhold,1.0,0.0)
    
    #vol_above = dlfx.fem.assemble_scalar(dlfx.fem.form( ( switch_function  ) * dx ))
    
    vol_above_threshhold = comm.allreduce(dlfx.fem.assemble_scalar(dlfx.fem.form( ( dlfx.fem.Constant(domain,1.0) * ufl.conditional(ufl.ge(function,threshhold),1.0,0.0) ) * dx )),MPI.SUM)
    return comm.allreduce(vol_above_threshhold,MPI.SUM)

def percentage_of_volume_above(domain: dlfx.mesh.Mesh, function: dlfx.fem.Function, threshhold: float, comm: MPI.Intercomm, dx : ufl.Measure):
    vol = volume_of_mesh(domain,comm,dx)
    vol_above = volume_of_mesh_above_threshhold(domain, function,threshhold, comm, dx)
    return comm.allreduce(vol_above / vol, MPI.MAX)



def reaction_force(sigma_func, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
    if n.ufl_shape[0] == 3:
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        Rz = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[2] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM), comm.allreduce(Rz,MPI.SUM)]
    elif n.ufl_shape[0] == 2:
        Rx = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[0] * ds))
        Ry = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.dot(sigma_func,n)[1] * ds))
        return [comm.allreduce(Rx,MPI.SUM), comm.allreduce(Ry,MPI.SUM)]
    else:
        raise NotImplementedError(f"dim {sigma_func.function_space.mesh.geometry.dim} not implemented")
    

def work_increment_external_forces(sigma_func, u: dlfx.fem.Function, um1: dlfx.fem.Function, n: ufl.FacetNormal, ds: ufl.Measure, comm: MPI.Intercomm,):
    du = u-um1
    t = ufl.dot(sigma_func,n)
    dW = dlfx.fem.assemble_scalar(dlfx.fem.form(ufl.inner(t,du)*ds))
    return comm.allreduce(dW,MPI.SUM)



def append_to_file(filename, parameters, comm=MPI.COMM_WORLD):
    if comm.Get_rank() == 0:
        with open(filename, 'a') as file:
            for key, value in parameters.items():
                file.write(f"{key}={value}\n")
                print(f"Appended to {filename}: {key}={value}")  # Debug print with file name

def write_to_file(filename, parameters, comm=MPI.COMM_WORLD):
    if comm.Get_rank() == 0:
        with open(filename, 'w') as file:
            for key, value in parameters.items():
                file.write(f"{key}={value}\n")
                print(f"Written to {filename}: {key}={value}")  # Debug print with file name

                
def read_parameters_file(file_path):
    """
    Reads a file with key-value pairs separated by '=' and returns a dictionary.
    
    Parameters:
    file_path (str): Path to the file containing key-value pairs.
    
    Returns:
    dict: A dictionary containing the key-value pairs from the file, with values as integers, floats, or strings.
    """
    data_dict = {}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Split the line into key and value
                key, value = line.split('=')
                
                # Remove any extraneous whitespace around key and value
                key = key.strip()
                value = value.strip()

                # Try to convert the value to an integer first, then a float, and finally leave as string if both conversions fail
                try:
                    data_dict[key] = int(value)
                except ValueError:
                    try:
                        data_dict[key] = float(value)
                    except ValueError:
                        data_dict[key] = value  # Keep as string if it cannot be converted to a number
    
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except ValueError:
        print("Error: Could not process the file. Check the file for proper formatting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return data_dict


def get_volume_average_of_field ( field,  vol,  dx: ufl.Measure, comm: MPI.Intercomm) -> float:
    fieldDv = dlfx.fem.assemble_scalar(dlfx.fem.form( 1.0 / vol * field * dx))
    return comm.allreduce(fieldDv,MPI.SUM)
