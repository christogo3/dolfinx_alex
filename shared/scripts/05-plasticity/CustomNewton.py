import dolfinx
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

def q(u):
    return 1 + u**2


domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = - ufl.div(q(u_ufl) * ufl.grad(u_ufl))


def u_exact(x):
    return eval(str(u_ufl))

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, fdim + 1)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
J = ufl.derivative(F, uh)
residual = dolfinx.fem.form(F)
jacobian = dolfinx.fem.form(J)







du = dolfinx.fem.Function(V)
import alex.solution as sol

def after_iteration():
    return True

sol.solve_with_custom_newton(jacobian, residual,uh,du,comm,[bc], after_iteration_hook=after_iteration)


# A = dolfinx.fem.petsc.create_matrix(jacobian)
# L = dolfinx.fem.petsc.create_vector(residual)
# solver = PETSc.KSP().create(comm)

# solver.setOperators(A)


# i = 0
# max_iterations = 25
# error = dolfinx.fem.form(ufl.inner(uh - u_ufl, uh - u_ufl) * ufl.dx(metadata={"quadrature_degree": 4}))
# L2_error = []
# du_norm = []
# while i < max_iterations:
#     # Assemble Jacobian and residual
#     with L.localForm() as loc_L:
#         loc_L.set(0)
#     A.zeroEntries()
#     dolfinx.fem.petsc.assemble_matrix(A, jacobian, bcs=[bc])
#     A.assemble()
#     dolfinx.fem.petsc.assemble_vector(L, residual)
#     L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     L.scale(-1)

#     # Compute b - J(u_D-u_(i-1))
#     dolfinx.fem.petsc.apply_lifting(L, [jacobian], [[bc]], x0=[uh.vector], scale=1)
#     # Set du|_bc = u_{i-1}-u_D
#     dolfinx.fem.petsc.set_bc(L, [bc], uh.vector, 1.0)
#     L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

#     # Solve linear problem
#     solver.solve(L, du.vector)
#     du.x.scatter_forward()

#     # Update u_{i+1} = u_i + delta u_i
#     uh.x.array[:] += du.x.array
#     i += 1

#     # Compute norm of update
#     correction_norm = du.vector.norm(0)

#     # Compute L2 error comparing to the analytical solution
#     L2_error.append(np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)))
#     du_norm.append(correction_norm)

#     print(f"Iteration {i}: Correction norm {correction_norm}, L2 error: {L2_error[-1]}")
#     if correction_norm < 1e-10:
#         break