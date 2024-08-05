#https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html

from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar, assemble_matrix,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import vtk_mesh
import dolfinx as dlfx
import ufl

from ufl import (SpatialCoordinate, TestFunction, TrialFunction, as_vector,
                 dx, grad, inner)

from mpi4py import MPI
import sys

import meshio
import gmsh
import numpy as np
import pyvista

from dolfinx.common import Timer, TimingType, list_timings

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

pyvista.start_xvfb()

gmsh.initialize()
top_marker = 2
bottom_marker = 1
left_marker = 1
if rank == 0:
    # We create one rectangle for each subdomain
    # gmsh.model.occ.addRectangle(0, 0, 0, 1, 0.5, tag=1)
    # gmsh.model.occ.addRectangle(0, 0.5, 0, 1, 0.5, tag=2)
    # # We fuse the two rectangles and keep the interface between them
    # gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
    # gmsh.model.occ.synchronize()
    
    

    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
    gmsh.model.occ.addDisk(0.5,0.5,0,0.1,0.1, tag=2)
    
    gmsh.model.occ.cut([(2,1)], [(2, 2)],)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0)  # Set the minimum mesh size
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.02) 
    #gmsh.model.occ.cut()
    

    # Mark the top (2) and bottom (1) rectangle
    top, bottom = None, None
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0.5, 0.25, 0]):
            bottom = surface[1]
        else:
            top = surface[1]
   # gmsh.model.addPhysicalGroup(2, [bottom], bottom_marker)
    gmsh.model.addPhysicalGroup(2, [top], top_marker)
    # Tag the left boundary
    left = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 0):
            left.append(line[1])
    gmsh.model.addPhysicalGroup(1, left, left_marker)
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
gmsh.finalize()

domain, cell_markers, facet_markers = gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, gdim=2)
#domain = gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, gdim=2)
dim = domain.topology.dim

# macroscopic strain
# eps_mac = np.array([[0.1, 0.00, 0.0],
#                     [0.00, 0.0, 0.0],
#                     [0.0, 0.0, 0.0]])



# elastic parameters
Emod = 70000 # MPa, aluminium
nu = 0.25

lam = dlfx.fem.Constant(domain, Emod*nu/((1-2*nu)*(1+nu)))
mu = dlfx.fem.Constant(domain, Emod/(2*(1+nu)))

def eps(u):
    val = ufl.sym(ufl.grad(u))
    return val

def sig(eps):
    val = lam*ufl.tr(eps)*ufl.Identity(dim)+2*mu*eps
    return val


# define function space
Ve = ufl.VectorElement('CG', domain.ufl_cell(), 1)
V = dlfx.fem.FunctionSpace(domain, Ve) 




# calculate boundary conditions
x = ufl.SpatialCoordinate(domain)
#print(x.array.shape)



# periodic BCs
tol = 250 * np.finfo(default_scalar_type).resolution #https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_periodic_geometrical.py

def isCorner(x):
    return np.logical_or.reduce([
        np.logical_and(np.isclose(x[0], 0, atol=tol), np.isclose(x[1], 0, atol=tol)),
        np.logical_and(np.isclose(x[0], 1, atol=tol), np.isclose(x[1], 0, atol=tol)),
        np.logical_and(np.isclose(x[0], 1, atol=tol), np.isclose(x[1], 1, atol=tol)),
        np.logical_and(np.isclose(x[0], 0, atol=tol), np.isclose(x[1], 1, atol=tol))
    ])


# set bc on corner nodes
fdim = dim-2
boundary_vertices = dlfx.mesh.locate_entities_boundary(domain, fdim, isCorner)
boundary_dofs = dlfx.fem.locate_dofs_topological(V, fdim, boundary_vertices) 
bc = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, np.array([0.0, 0.0])), boundary_dofs, V)
bcs = [bc]

def topPeriodicBoundary(x):
    return np.logical_and(np.isclose(x[1],1),np.logical_not(isCorner(x)))

def topPeriodicRelation(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = 1-x[1]
    out_x[2] = x[2]
    return out_x
        
def rightPeriodicBoundary(x):
    return np.logical_and(np.isclose(x[0],1),np.logical_not(isCorner(x)))

def rightPeriodicRelation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 1-x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x
    
def periodicBoundary(x):
    return np.logical_or(np.isclose(x[1],1),np.isclose(x[0],1))

def periodicRelation(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = 1-x[1]
    out_x[2] = x[2]
    return out_x
    

mpc = MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, topPeriodicBoundary, topPeriodicRelation, bcs)
mpc.create_periodic_constraint_geometrical(V, rightPeriodicBoundary, rightPeriodicRelation, [])
mpc.finalize()
    
    
u_fluct = ufl.TrialFunction(V)
du = ufl.TestFunction(V)
u_fluct_h = dlfx.fem.Function(V)

# Lagrange Multiplier to enforce well-posed-ness, i.e. avoid rigid body translation by setting average fluctuation to zero
# need to do mixed function space?
#lamb = ufl.TrialFunction(V)
#dlamb = ufl.TestFunction(V)

def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return dlfx.fem.Constant(domain,np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.0, 0.0],
                    [Eps_Voigt[2]/2.0, Eps_Voigt[1], 0.0],
                    [0.0, 0.0, 0.0]]))
    
def stress2Voigt(s):
    return as_vector([s[0,0], s[1,1], s[0,1]])

# eps_mac = dlfx.fem.Constant(domain,np.array([[0.1, 0.00, 0.0],
#                     [0.00, 0.0, 0.0],
#                     [0.0, 0.0, 0.0]]))

def u_macro(x,i):
    u_x = macro_strain(i)[0, 0]*x[0] + macro_strain(i)[0, 1]*x[1] #+ eps_mac[0, 2]*x[2] 
    u_y = macro_strain(i)[1, 0]*x[0] + macro_strain(i)[1, 1]*x[1] #+ eps_mac[1, 2]*x[2]
    #u_z = eps_mac[2, 0]*x[0] + eps_mac[2, 1]*x[1] #+ eps_mac[2, 2]*x[2]
    return ufl.as_vector((u_x, u_y))

vol = 1 * 1 # volume of domain total
Chom = np.zeros((3, 3))

for (i, case) in enumerate(["Exx", "Eyy", "Exy"]):
    # compute macroscopic displacements
    u_mac = dlfx.fem.Function(V)
    u_mac_expr = dlfx.fem.Expression(u_macro(x,i), V.element.interpolation_points())
    u_mac.interpolate(u_mac_expr)

    # setup linear problem
    Fweak = ufl.inner(sig(eps(u_fluct)+eps(u_mac)), eps(du))*ufl.dx
    a, L = ufl.lhs(Fweak) , ufl.rhs(Fweak)
    #a += ufl.inner(lamb,du) * ufl.dx + ufl.inner(dlamb,u_fluct) * ufl.dx # https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html

    lin_problem = dolfinx_mpc.LinearProblem(a=a, L=L,mpc=mpc, bcs=bcs)

    # report on system
    num_dofs = np.shape(u_fluct_h.x.array[:])[0]
    comm.Barrier()
    num_dofs_all = comm.allreduce(num_dofs, op=MPI.SUM)
    comm.Barrier()

    if rank == 0:
        print('solving fem problem with', num_dofs_all,'dofs ...')
        sys.stdout.flush()

    # solve linear problem, need to read the solution with MPC.LinearProblem 
    u_fluct_h = lin_problem.solve()

    Sigma = np.zeros((3,))
    for k in range(3):
        Sigma[k] = assemble_scalar(form(stress2Voigt(sig(eps(u_fluct_h) + eps(u_mac)))[k]*ufl.dx))/vol
        # sig_xx_av = assemble_scalar(sig(eps(u_fluct_h)+eps(u_mac)))
    Chom[i,:] = Sigma

print(np.array_str(Chom, precision=2))
lmbda_hom = Chom[0, 1]
mu_hom = Chom[2, 2]
print(Chom[0, 0], lmbda_hom + 2*mu_hom)

E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2
print("Apparent Young modulus:", E_hom)
print("Apparent Poisson ratio:", nu_hom)

# postprocessing of fieldsu_total = dlfx.fem.Function(V)
u_total = dlfx.fem.Function(V)
u_total.x.array[:] = u_fluct_h.x.array[:]+u_mac.x.array[:]

# write results
u_total.name = "u"
u_fluct_h.name = "u_fluct"
u_mac.name = "u_macro"
with dlfx.io.XDMFFile(comm, "periodic_bc_2D_example/periodic_2D.xdmf", "w") as xdmf_out:
    xdmf_out.write_mesh(domain)
    xdmf_out.write_function(u_total)
    xdmf_out.write_function(u_fluct_h)
    xdmf_out.write_function(u_mac)




