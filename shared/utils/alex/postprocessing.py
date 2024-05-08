from typing import Callable, Union
import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import pyvista
import dolfinx.plot as plot
import matplotlib.pyplot as plt
import glob 
import os
import sys

def write_mesh_and_get_outputfile_xdmf(domain: dlfx.mesh.Mesh,
                                       outputfile_xdmf_path: str,
                                       comm: MPI.Intercomm,
                                       meshtags: any = None):
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'w')
    xdmfout.write_mesh(domain)
    if( not meshtags is None):
         xdmfout.write_meshtags(meshtags, domain.geometry)
    xdmfout.close()
    return True

def write_phasefield_mixed_solution(domain: dlfx.mesh.Mesh,
                                    outputfile_xdmf_path: str,
                                    w: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    u, s = w.split() 
    
    u.name = "u"
    write_vector_field(domain,outputfile_xdmf_path,u,t,comm)
    
    s.name = "s"
    write_field(domain,outputfile_xdmf_path,s,t,comm)  
    # Ue = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    # Se = ufl.FiniteElement('CG', domain.ufl_cell(), 1)
    
    # U = dlfx.fem.FunctionSpace(domain, Ue)
    # S = dlfx.fem.FunctionSpace(domain, Se)
    # s_interp = dlfx.fem.Function(S)
    # u_interp = dlfx.fem.Function(U)
    
    # s_interp.interpolate(s)
    # u_interp.interpolate(u)
    # s_interp.name = 's'
    # u_interp.name = 'u'
    
    # # append xdmf-file
    # xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    # xdmfout.write_function(u_interp, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
    # xdmfout.write_function(s_interp, t)
    # xdmfout.close()
    # return xdmfout

def write_field(domain: dlfx.mesh.Mesh,
                                    outputfile_xdmf_path: str,
                                    field: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    Se = ufl.FiniteElement('CG', domain.ufl_cell(), 1)
    
    S = dlfx.fem.FunctionSpace(domain, Se)
    field_interp = dlfx.fem.Function(S)
    
    field_interp.interpolate(field)

    field_interp.name = field.name
    
    # append xdmf-file
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    xdmfout.write_function(field_interp, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
    xdmfout.close()
    return xdmfout

def write_vector_field(domain: dlfx.mesh.Mesh,
                                    outputfile_xdmf_path: str,
                                    field: dlfx.fem.Function,
                                    t: dlfx.fem.Constant,
                                    comm: MPI.Intercomm) :
    
    
    # split solution to displacement and crack field
    Se = ufl.VectorElement('CG', domain.ufl_cell(), 1)
    
    S = dlfx.fem.FunctionSpace(domain, Se)
    field_interp = dlfx.fem.Function(S)
    
    field_interp.interpolate(field)

    field_interp.name = field.name
    
    # append xdmf-file
    xdmfout = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    xdmfout.write_function(field_interp, t) # collapse reduces to subspace so one can work only in subspace https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2, only one component?
    xdmfout.close()
    return xdmfout


    
def write_tensor_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, tensor_fields_as_functions, tensor_field_names, outputfile_xdmf_path: str, t: float):
    TENe = ufl.TensorElement('DG', domain.ufl_cell(), 0)
    TEN = dlfx.fem.FunctionSpace(domain, TENe) 
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        for n  in range(0,len(tensor_fields_as_functions)):
            tensor_field_function = tensor_fields_as_functions[n]
            tensor_field_name = tensor_field_names[n]
            tensor_field_expression = dlfx.fem.Expression(tensor_field_function, 
                                                        TEN.element.interpolation_points())
            out_tensor_field = dlfx.fem.Function(TEN)
            out_tensor_field.interpolate(tensor_field_expression)
            out_tensor_field.name = tensor_field_name
            
            xdmf_out.write_function(out_tensor_field,t)

def write_vector_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm, vector_fields_as_functions, vector_field_names, outputfile_xdmf_path: str, t: float):
    Ve = ufl.VectorElement('CG', domain.ufl_cell(), 1)
    V = dlfx.fem.FunctionSpace(domain, Ve) 
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
    Se = ufl.FiniteElement('DG', domain.ufl_cell(), 0)
    S = dlfx.fem.FunctionSpace(domain, Se) 
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
    
def print_graphs_plot(output_file_path, print_path, legend_labels=None, default_label="Column"):
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
    plt.savefig(print_path + '/graphs.png') 
    plt.close()
    


def number_of_nodes(domain: dlfx.mesh.Mesh):
    return domain.topology.index_map(0).size_global

            
        
        
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