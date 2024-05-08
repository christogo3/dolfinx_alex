import dolfinx as dlfx
import ufl as ufl
import numpy as np
from mpi4py import MPI
from dolfinx.cpp.la import InsertMode

from typing import List, Tuple

################# INTEGRALS of tensor fields from nodal forces ############################################
def get_volume_integral_of_div_of_tensors_from_nodal_forces(tensor: ufl.classes.ListTensor, W: dlfx.fem.FunctionSpace, dx: ufl.Measure = ufl.dx):
    V, _ = W.sub(0).collapse()
    J_nodal_vector_local_as_function = compute_nodal_forces_vector_from_locally_as_function(tensor, V, dx)

    J_nodal_local_sum = get_local_sum_of_nodal_forces(J_nodal_vector_local_as_function)
    
    J_nodal_global_sum = assemble_global_sum_3x1(J_nodal_local_sum)
    return J_nodal_global_sum

def assemble_global_sum_3x1(J_nodal_local_sum):
    J_nodal_global_sum = np.zeros(3, dtype=np.float64)
    for i in range(3):
        # J_sub = J_nodal_vector_local[i]
        J_nodal_global_sum[i] = MPI.COMM_WORLD.allreduce(J_nodal_local_sum[i], op=MPI.SUM)
    return J_nodal_global_sum

def get_local_sum_of_nodal_forces(J_nodal_vector_local_as_function: dlfx.fem.Function) -> Tuple[float, float, float]:
    J_nodal_local_sum = np.zeros(3, dtype=np.float64)
    for i in range(3):
        num_dofs_local = get_num_of_dofs_locally(J_nodal_vector_local_as_function[i])
        J_nodal_local_sum[i] = np.sum(J_nodal_vector_local_as_function[i].x.array[:num_dofs_local])
    return (J_nodal_local_sum[0], J_nodal_local_sum[1], J_nodal_local_sum[2])

def get_num_of_dofs_locally(local_function: dlfx.fem.Function) -> int:
    sub_imap = local_function.function_space.dofmap
    num_dofs_local = sub_imap.index_map.size_local * sub_imap.index_map_bs
    return num_dofs_local

def compute_nodal_forces_vector_from_locally_as_function(tensor: ufl.classes.ListTensor, V, dx: ufl.Measure) -> Tuple[dlfx.fem.Function, dlfx.fem.Function, dlfx.fem.Function]:
    '''
        returns a tuple containing functions for all three components of the field
    '''
    du = ufl.TestFunction(V)
    J_nodal_vector = dlfx.fem.Function(V)
    form_0 = dlfx.fem.form(ufl.dot(ufl.div(tensor), du) * dx)
    dlfx.fem.assemble_vector(J_nodal_vector.x.array, form_0)
    J_nodal_vector.x.scatter_reverse(InsertMode.add)
    J_nodal_vector.x.scatter_forward()
    return (J_nodal_vector.sub(0).collapse(), J_nodal_vector.sub(1).collapse(), J_nodal_vector.sub(2).collapse())

###################################################################################################################