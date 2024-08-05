import numpy as np
from dolfinx import fem
from dolfinx.mesh import create_rectangle
from mpi4py import MPI
import ufl

mesh = create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])], [32, 32])

Velem = ufl.VectorElement('Lagrange', mesh.ufl_cell(), degree=1, dim=2)
Felem = ufl.FiniteElement('Lagrange', mesh.ufl_cell(), degree=1)
V = fem.FunctionSpace(mesh, Velem)
F = fem.FunctionSpace(mesh, Felem)

u = fem.Function(V)
du = ufl.TestFunction(V)
ds = ufl.TestFunction(F)
grad_ds = ufl.grad(ds)

Ten = ufl.grad(u)

Gvec1 = fem.assemble_scalar(fem.form( (Ten[0,0]*grad_ds[0] + Ten[0,1]*grad_ds[1])*ufl.dx))
Gvec2 = fem.assemble_scalar(fem.form( (Ten[1,0]*grad_ds[0] + Ten[1,1]*grad_ds[1])*ufl.dx))

print("hi")