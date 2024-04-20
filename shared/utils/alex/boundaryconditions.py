#linear displacements
import dolfinx as dlfx
from typing import Callable
import numpy as np
from mpi4py import MPI

def linear_displacements_mixed(mixedFunctionSpace: dlfx.fem.FunctionSpace, 
                               eps_mac: dlfx.fem.Constant):
    # assume mixedFunctionSpace.sub(0) contains the displacementFields
    w_D = dlfx.fem.Function(mixedFunctionSpace)
    w_D.sub(0).sub(0).interpolate(lambda x: eps_mac.value[0, 0]*x[0] + eps_mac.value[0, 1]*x[1] + eps_mac.value[0, 2]*x[2] )
    w_D.sub(0).x.scatter_forward()
    w_D.sub(0).sub(1).interpolate(lambda x: eps_mac.value[1, 0]*x[0] + eps_mac.value[1, 1]*x[1] + eps_mac.value[1, 2]*x[2] )
    w_D.sub(0).x.scatter_forward()
    w_D.sub(0).sub(2).interpolate(lambda x: eps_mac.value[2, 0]*x[0] + eps_mac.value[2, 1]*x[1] + eps_mac.value[2, 2]*x[2] )
    w_D.sub(0).x.scatter_forward()
    return w_D

def define_dirichlet_bc_from_interpolated_function_mixed(domain: dlfx.mesh.Mesh,
                                                         desired_value_at_boundary_function: dlfx.fem.Function,
                                                         where_function: Callable,
                                                         mixedFunctionSpace: dlfx.fem.FunctionSpace,
                                                         subspace_idx: int) -> dlfx.fem.DirichletBC:
    fdim = domain.topology.dim-1
    facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
    dofs_at_boundary = dlfx.fem.locate_dofs_topological(mixedFunctionSpace.sub(subspace_idx), fdim, facets_at_boundary)
    bc = dlfx.fem.dirichletbc(desired_value_at_boundary_function,dofs_at_boundary)
    return bc

def get_dimensions(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm):
        x_min = np.min(domain.geometry.x[:,0]) 
        x_max = np.max(domain.geometry.x[:,0])   
        y_min = np.min(domain.geometry.x[:,1]) 
        y_max = np.max(domain.geometry.x[:,1])   
        z_min = np.min(domain.geometry.x[:,2]) 
        z_max = np.max(domain.geometry.x[:,2])

        # find global min/max over all mpi processes
        comm.Barrier()
        x_min_all = comm.allreduce(x_min, op=MPI.MIN)
        x_max_all = comm.allreduce(x_max, op=MPI.MAX)
        y_min_all = comm.allreduce(y_min, op=MPI.MIN)
        y_max_all = comm.allreduce(y_max, op=MPI.MAX)
        z_min_all = comm.allreduce(z_min, op=MPI.MIN)
        z_max_all = comm.allreduce(z_max, op=MPI.MAX)
        comm.Barrier()
        return [x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all]

def get_total_linear_displacement_boundary_condition_at_box(domain: dlfx.mesh.Mesh, 
                                                               comm: MPI.Intercomm,
                                                               mixedFunctionSpace: dlfx.fem.FunctionSpace,
                                                               subspace_idx: int, 
                                                               eps_mac: dlfx.fem.Constant):
    
    [x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all] = get_dimensions(domain, comm)
    
    # define top boundary
    def top(x):
        return np.isclose(x[1], y_max_all)


    # define bottom boundary
    def bottom(x):
        return np.isclose(x[1], y_min_all)

    def left(x):
        return np.isclose(x[0], x_min_all)

    def right(x):
        return np.isclose(x[0], x_max_all)


    def front(x):
        return np.isclose(x[2], z_max_all)

    def back(x):
        return np.isclose(x[2], z_min_all)
    
    w_D = linear_displacements_mixed(mixedFunctionSpace, eps_mac=eps_mac)
    
    bcs = []
    for where_function in [top, bottom,left, right, front, back]:
        bcs.append(define_dirichlet_bc_from_interpolated_function_mixed(domain,w_D,where_function,mixedFunctionSpace,subspace_idx))
        
    return bcs
        
    
    
    
    
    