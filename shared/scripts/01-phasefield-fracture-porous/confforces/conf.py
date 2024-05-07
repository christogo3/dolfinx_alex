import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
from  dolfinx.cpp.la import InsertMode

script_path = os.path.dirname(__file__)
comm = MPI.COMM_WORLD

N = 16

domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)


Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) 
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) 
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

w =  dlfx.fem.Function(W)

w.sub(0).sub(0).interpolate(lambda x: x[0])
w.sub(0).sub(1).interpolate(lambda x: x[1])
w.sub(0).sub(2).interpolate(lambda x: x[2])
w.sub(1).interpolate(lambda x: x[0])

u,s = w.split()


def tensor_function_of_u_s(u,s):
    return ufl.as_tensor([[u[0]*s, 0, 0],
                          [u[1], 0, 0],
                          [u[2], 0, 0]
                          ])

tensor = tensor_function_of_u_s(u,s)

##1. from surface integral ###############################
# returns the expected value 1 and is independent on number of processes  
def surface_integral_of_tensor(tensor, n: ufl.FacetNormal, ds: ufl.Measure = ufl.ds):
    Jx = (tensor[0,0]*n[0]+tensor[0,1]*n[1]+tensor[0,2]*n[2])*ds
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
    Jy = (tensor[1,0]*n[0]+tensor[1,1]*n[1]+tensor[1,2]*n[2])*ds
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form(Jy))
    Jz = (tensor[2,0]*n[0]+tensor[2,1]*n[1]+tensor[2,2]*n[2])*ds
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form(Jz))
    return Jxa, Jya, Jza

n = ufl.FacetNormal(domain)
Jx_s, Jy_s, Jz_s = surface_integral_of_tensor(tensor,n)

##2. from volume integral ###############################
# returns the expected value 1 and is independent on number of processes 
def volume_integral_of_tensor(tensor, dx: ufl.Measure = ufl.dx):
    Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(tensor)[0] ) * dx ) ))
    Jya = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(tensor)[1] ) * dx ) ))
    Jza = dlfx.fem.assemble_scalar(dlfx.fem.form( ( ( ufl.div(tensor)[2] ) * dx )))
    return Jxa, Jya, Jza
Jx_v, Jy_v, Jz_v = volume_integral_of_tensor(tensor)


##3. from testfunctions ###############################
# fails to return the expected value 1 and is dependent on number of processes 
dw = ufl.TestFunction(W)
du, dv = ufl.split(dw)
J_nodal_vector : dlfx.la.Vector = dlfx.fem.assemble_vector(dlfx.fem.form( ufl.inner(-tensor, ufl.grad(du))*ufl.dx ))
# J_nodal_vector.scatter_reverse(InsertMode.add)

# cast into function
temp = dlfx.fem.Function(W)
temp.sub(0).x.array[:] = J_nodal_vector.array
J_nodal_function = temp.sub(0).collapse()
Jx_from_sum_of_nodal_vectors = np.sum(J_nodal_function.sub(0).x.array)

###########################################################
print('Jx from surface integral:  {0:.4e}\nJx from volume integral: {1:.4e}\nJx from nodal vectors: {2:.4e}'.format(Jx_s, Jx_v, Jx_from_sum_of_nodal_vectors))


# V = dlfx.fem.FunctionSpace(domain, Ve)
# u_out = dlfx.fem.Function(V)
# u_out.interpolate(u)
# u_out.name = "u"

# S = dlfx.fem.FunctionSpace(domain, Te)
# s_out = dlfx.fem.Function(S)
# s_out.interpolate(s)
# s_out.name = "s"


# xdmfout = dlfx.io.XDMFFile(comm, script_path+"/output.xdmf", 'w')
# xdmfout.write_mesh(domain)
# xdmfout.write_function(u_out, 0) 
# xdmfout.write_function(s_out, 0) 
# xdmfout.close()