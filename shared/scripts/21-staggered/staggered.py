import matplotlib.pyplot as plt
import numpy as np

import dolfinx
import dolfinx.plot
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import pyvista
from pyvista.utilities.xvfb import start_xvfb
start_xvfb(wait=0.5)

import sys
sys.path.append("../python/")
# from snes_problem import SNESProblem

# from plots import plot_damage_state


L = 1.; H = 0.3;
ell_ = 0.1
cell_size = ell_/6;

nx = int(L/cell_size)
ny = int(H/cell_size)

mesh = dolfinx.RectangleMesh(MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (L, H, 0.0)], [nx, ny])


ndim = mesh.geometry.dim


L = 1.; H = 0.3;
ell_ = 0.1
cell_size = ell_/6;

nx = int(L/cell_size)
ny = int(H/cell_size)

mesh = dolfinx.RectangleMesh(MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (L, H, 0.0)], [nx, ny])


ndim = mesh.geometry.dim


pyvista.OFF_SCREEN 
topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
plotter.add_axes()
plotter.set_scale(5,5)
#plotter.reset_camera(render=True, bounds=(-L/2, L/2, -H/2, H/2, 0, 0))
if not pyvista.OFF_SCREEN:
    plotter.show()

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)
figure = plotter.screenshot("output/mesh.png")

pyvista.OFF_SCREEN 
topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
plotter.add_axes()
plotter.set_scale(5,5)
#plotter.reset_camera(render=True, bounds=(-L/2, L/2, -H/2, H/2, 0, 0))
if not pyvista.OFF_SCREEN:
    plotter.show()

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)
figure = plotter.screenshot("output/mesh.png")


element_u = ufl.VectorElement('Lagrange',mesh.ufl_cell(),degree=1,dim=2)
V_u = dolfinx.FunctionSpace(mesh, element_u)

element_alpha = ufl.FiniteElement('Lagrange',mesh.ufl_cell(),degree=1)
V_alpha = dolfinx.FunctionSpace(mesh, element_alpha)

# Define the state
u = dolfinx.Function(V_u, name="Displacement")
alpha = dolfinx.Function(V_alpha, name="Damage")

state = {"u": u, "alpha": alpha}

# need upper/lower bound for the damage field
alpha_lb = dolfinx.Function(V_alpha, name="Lower bound")
alpha_ub = dolfinx.Function(V_alpha, name="Upper bound")

# Measures
dx = ufl.Measure("dx",domain=mesh)
ds = ufl.Measure("ds",domain=mesh)



def bottom(x):
    return np.isclose(x[1], 0.0)

def right(x):
    return np.isclose(x[0], L)

def top(x):
    return np.isclose(x[1], H)

def left(x):
    return np.isclose(x[0], 0.0)

blocked_dofs_left_u = dolfinx.fem.locate_dofs_geometrical((V_u.sub(0), V_u.sub(0).collapse()), left)
blocked_dofs_right_u = dolfinx.fem.locate_dofs_geometrical((V_u.sub(0), V_u.sub(0).collapse()), right)
blocked_dofs_bottom_u = dolfinx.fem.locate_dofs_geometrical((V_u.sub(1), V_u.sub(1).collapse()), bottom)
blocked_dofs_left_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, left)
blocked_dofs_right_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, right)

zero_u = dolfinx.Function(V_u.sub(0).collapse())

with zero_u.vector.localForm() as bc_local:
    bc_local.set(0.0)

nonzero_u = dolfinx.Function(V_u.sub(0).collapse())
with nonzero_u.vector.localForm() as bc_local:
    bc_local.set(1.0)
    
one_alpha = dolfinx.Function(V_alpha)
with one_alpha.vector.localForm() as bc_local:
    bc_local.set(1.0)
                 
zero_alpha = dolfinx.Function(V_alpha)
with zero_alpha.vector.localForm() as bc_local:
    bc_local.set(0.0)


bc_u0 = dolfinx.DirichletBC(zero_u, blocked_dofs_left_u, V_u.sub(1))
bc_u1 = dolfinx.DirichletBC(nonzero_u, blocked_dofs_right_u, V_u.sub(1))
bc_u2 = dolfinx.DirichletBC(zero_u, blocked_dofs_bottom_u, V_u.sub(0))

bc_alpha0 = dolfinx.DirichletBC(zero_alpha, blocked_dofs_left_alpha)
bc_alpha1 = dolfinx.DirichletBC(zero_alpha, blocked_dofs_right_alpha)

bcs_u = [bc_u0,bc_u1,bc_u2]
bcs_alpha = [bc_alpha0,bc_alpha1]

# setting the upper bound to 0 where BCs are applied
alpha_ub.interpolate(one_alpha)
dolfinx.fem.set_bc(alpha_ub.vector, bcs_alpha)


E, nu = dolfinx.Constant(mesh, 100.0), dolfinx.Constant(mesh, 0.3)
Gc = dolfinx.Constant(mesh, 1.0)
ell = dolfinx.Constant(mesh, ell_)

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha

def a(alpha, k_ell=1.e-6):
    """Stiffness modulation as a function of the damage """
    return (1 - alpha) ** 2 + k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    mu    = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / (1.0 - nu ** 2)
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(u)




import sympy 
z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
print("c_w = ",c_w)

c_1w = sympy.integrate(sympy.sqrt(1/w(z)),(z,0,1))
print("c_1/w = ",c_1w)

tmp = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp * Gc.value * E.value / (c_w * ell.value))
print("sigma_c = %2.3f"%sigma_c)

eps_c = float(sigma_c/E.value)
print("eps_c = %2.3f"%eps_c)


f = dolfinx.fem.Constant(mesh,(0,0))
elastic_energy = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
dissipated_energy = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx
external_work = ufl.dot(f, u) * dx 
total_energy = elastic_energy + dissipated_energy - external_work

f = dolfinx.fem.Constant(mesh,(0,0))
elastic_energy = 0.5 * ufl.inner(sigma(u,alpha), eps(u)) * dx 
dissipated_energy = Gc / float(c_w) * (w(alpha) / ell + ell * ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx
external_work = ufl.dot(f, u) * dx 
total_energy = elastic_energy + dissipated_energy - external_work

E_u = ufl.derivative(total_energy,u,ufl.TestFunction(V_u))
E_du = ufl.replace(E_u,{u: ufl.TrialFunction(V_u)})

problem_u = dolfinx.fem.LinearProblem(a=ufl.lhs(E_du), L=ufl.rhs(E_du), bcs=bcs_u, u=u,
                                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

problem_u.solve()

load = .6
with nonzero_u.vector.localForm() as bc_local:
        bc_local.set(load) 
        
problem_u.solve()

plot_damage_state(state,load=load)


E_alpha = ufl.derivative(total_energy,alpha,ufl.TestFunction(V_alpha))
E_alpha_alpha = ufl.derivative(E_alpha,alpha,ufl.TrialFunction(V_alpha))

damage_problem = SNESProblem(E_alpha, E_alpha_alpha, alpha, bcs_alpha)

b = dolfinx.cpp.la.create_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J = dolfinx.fem.create_matrix(damage_problem.a)

# Create Newton solver and solve
solver_alpha_snes = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b)
solver_alpha_snes.setJacobian(damage_problem.J, J)
solver_alpha_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_alpha_snes.getKSP().setType("preonly")
solver_alpha_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_alpha_snes.getKSP().getPC().setType("lu")

# We set the bound (Note: they are passed as reference and not as values)
solver_alpha_snes.setVariableBounds(alpha_lb.vector,alpha_ub.vector)

solver_alpha_snes.solve(None, alpha.vector)
plot_damage_state(state,load=load)

with alpha.vector.localForm() as alpha_local:
    alpha_local.set(0)

for i in range(10):
    print(f"iteration {i}")
    problem_u.solve()
    solver_alpha_snes.solve(None, alpha.vector)
    plot_damage_state(state,load)
