import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
from dolfinx.fem.petsc import assemble_vector

script_path = os.path.dirname(__file__)

L=1
W= 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), 
                                          np.array([L, W, W])], 
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)

V = fem.VectorFunctionSpace(domain, ("Lagrange",1))

#boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0],0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, 
                     fem.locate_dofs_topological(V, fdim, boundary_facets), V)


# define bc only on a particular dof
def sliding_boundary(x):
    return np.isclose(x[0],L)
boundary_facets_sliding = mesh.locate_entities(domain,fdim,sliding_boundary)
u_D_sliding = np.array(0, dtype=default_scalar_type)
bc_sliding = fem.dirichletbc(u_D_sliding, 
                     fem.locate_dofs_topological(V.sub(0),
                                                 fdim, 
                                                 boundary_facets_sliding), 
                     V.sub(0))

# traction zero everywhere
T = fem.Constant(domain, default_scalar_type((0,0,0)))

ds = ufl.Measure("ds", domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

eta = ufl.TestFunction(V)

def test(u):
    out = u[0] * 1.0
    return out

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f,v) * ufl.dx + ufl.dot(T,v) * ds



# solving
problem = LinearProblem(a,L,bcs=[bc, bc_sliding],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()



G = (test(u)*ufl.grad(eta))[0,0] * ufl.dx
# G = ufl.inner(sigma(u),ufl.grad(eta)) * ufl.dx
G_out = assemble_vector(dlfx.fem.form(G,dtype=default_scalar_type))



# plotting
pyvista.start_xvfb()
p = pyvista.Plotter(off_screen=True)
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# attach vector values to grid and war grid by vector
#print((geometry.shape[0], 3))
grid["u"] = uh.x.array.reshape(((geometry.shape[0], 3))) # an array with 
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
figure_as_array = p.screenshot(script_path + "/deflection.png")
