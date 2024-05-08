import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
from dolfinx.cpp.la import InsertMode

import alex.tensor

script_path = os.path.dirname(__file__)
comm = MPI.COMM_WORLD

N = 16

domain = dlfx.mesh.create_unit_cube(
    comm, N, N, N, cell_type=dlfx.mesh.CellType.hexahedron
)


Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

w = dlfx.fem.Function(W)

w.sub(0).sub(0).interpolate(lambda x: x[0])
w.sub(0).sub(1).interpolate(lambda x: x[1])
w.sub(0).sub(2).interpolate(lambda x: x[2])
w.sub(1).interpolate(lambda x: x[0])

u, s = w.split()


def tensor_function_of_u_s(u, s):
    return ufl.as_tensor([[u[0] * s * s * s, 0, 0], [u[1], 0, 0], [u[2], 0, 0]])


tensor : ufl.classes.ListTensor = tensor_function_of_u_s(u, s)


##1. from surface integral ###############################
# returns the expected value 1 and is independent on number of processes
def surface_integral_of_tensor(tensor, n: ufl.FacetNormal, ds: ufl.Measure = ufl.ds):
    Jx = (tensor[0, 0] * n[0] + tensor[0, 1] * n[1] + tensor[0, 2] * n[2]) * ds
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
    Jy = (tensor[1, 0] * n[0] + tensor[1, 1] * n[1] + tensor[1, 2] * n[2]) * ds
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form(Jy))
    Jz = (tensor[2, 0] * n[0] + tensor[2, 1] * n[1] + tensor[2, 2] * n[2]) * ds
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form(Jz))
    return (
        MPI.COMM_WORLD.allreduce(Jxa, op=MPI.SUM),
        MPI.COMM_WORLD.allreduce(Jya, op=MPI.SUM),
        MPI.COMM_WORLD.allreduce(Jza, op=MPI.SUM),
    )


n = ufl.FacetNormal(domain)
Jx_s, Jy_s, Jz_s = surface_integral_of_tensor(tensor, n)


##2. from volume integral ###############################
# returns the expected value 1 and is independent on number of processes
def volume_integral_of_tensor(tensor, dx: ufl.Measure = ufl.dx):
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(((ufl.div(tensor)[0]) * dx)))
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form(((ufl.div(tensor)[1]) * dx)))
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form(((ufl.div(tensor)[2]) * dx)))
    return (
        MPI.COMM_WORLD.allreduce(Jxa, op=MPI.SUM),
        MPI.COMM_WORLD.allreduce(Jya, op=MPI.SUM),
        MPI.COMM_WORLD.allreduce(Jza, op=MPI.SUM),
    )


Jx_v, Jy_v, Jz_v = volume_integral_of_tensor(tensor)


##3. from nodal values ###############################
# fails to return the expected value 1 and is dependent on number of processes
def get_volume_integral_of_div_of_tensors_from_nodal_forces(tensor: ufl.classes.ListTensor, W: dlfx.fem.FunctionSpace):
    V, _ = W.sub(0).collapse()
    du = ufl.TestFunction(V)
    J_nodal_vector = dlfx.fem.Function(V)

# boundary_facets = dlfx.mesh.exterior_facet_indices(domain.topology)
# bc_dofs = dlfx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
# bc_zero = dlfx.fem.Function(V)
# bc = dlfx.fem.dirichletbc(bc_zero, bc_dofs)
# form_0 = dlfx.fem.form(ufl.inner(-tensor, ufl.grad(du)) * ufl.dx)
    form_0 = dlfx.fem.form(ufl.dot(ufl.div(tensor), du) * ufl.dx)
    dlfx.fem.assemble_vector(J_nodal_vector.x.array, form_0)
    J_nodal_vector.x.scatter_reverse(InsertMode.add)
    J_nodal_vector.x.scatter_forward()


    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    J_nodal = np.zeros(3, dtype=np.float64)
    for i in range(3):
        J_sub = J_nodal_vector.sub(i).collapse()
        sub_imap = J_sub.function_space.dofmap
        num_dofs_local = sub_imap.index_map.size_local * sub_imap.index_map_bs
        J_nodal[i] = MPI.COMM_WORLD.allreduce(
        np.sum(J_sub.x.array[:num_dofs_local]), op=MPI.SUM
    )
        
    return J_nodal

# J_nodal = get_volume_integral_of_div_of_tensors_from_nodal_forces(tensor, W)
J_nodal = alex.tensor.get_volume_integral_of_div_of_tensors_from_nodal_forces(tensor,W)
print(J_nodal)

###########################################################
print(
    "Jx from surface integral:  {0:.4e}\nJx from volume integral: {1:.4e}\nJx from nodal vectors: {2:.4e}".format(
        Jx_s, Jx_v, J_nodal[0]
    )
)