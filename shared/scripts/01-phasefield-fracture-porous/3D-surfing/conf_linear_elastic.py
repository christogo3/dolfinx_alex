import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
from dolfinx.fem.petsc import assemble_vector
import gmsh
import sys

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

# domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), 
#                                           np.array([L, W, W])], 
#                          [20, 6, 6], cell_type=mesh.CellType.hexahedron)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

gmsh.initialize()
top_marker = 2
bottom_marker = 1
left_marker = 1
if rank == 0:
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
    # gmsh.model.occ.addWedge(-0.05,0.5,0,0.1,0.1,0.0, tag=2)
    gmsh.model.occ.addDisk(0.0,0.5,0,0.3, 0.01, tag=2)
    # gmsh.model.occ.addDisk()
    # el = gmsh.model.occ.addEllipse(0.0,0.5,0.0,0.2,0.1,tag=3)
    
    gmsh.model.occ.cut([(2,1)], [(2, 2)])
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
    gmsh.write(script_path + "/mesh.msh")
gmsh.finalize()

domain, cell_markers, facet_markers = gmshio.read_from_msh(script_path + "/mesh.msh", MPI.COMM_WORLD, gdim=2)

V = fem.VectorFunctionSpace(domain, ("Lagrange",1))
fdim = domain.geometry.dim

# define boundaries
def bot(x):
    return np.isclose(x[1],1.0)


# define boundaries
def top_b(x):
    return np.isclose(x[1],0.0)

bottom_facets = dlfx.mesh.locate_entities_boundary(domain, fdim-1, bot)   # check with x/y/z_min/max
boundary_dofs_bot_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim-1, bottom_facets)
boundary_dofs_bot_y = dlfx.fem.locate_dofs_topological(V.sub(1), fdim-1, bottom_facets)

top_facets = dlfx.mesh.locate_entities_boundary(domain, fdim-1, top_b)   # check with x/y/z_min/max
boundary_dofs_top_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim-1, top_facets)
boundary_dofs_top_y = dlfx.fem.locate_dofs_topological(V.sub(1), fdim-1, top_facets)


bc_top_y = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, 1.0), boundary_dofs_top_y, V.sub(1))  
bc_bot_y = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, -1.0), boundary_dofs_bot_y, V.sub(1)) 

bc_top_x = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, 1.0), boundary_dofs_top_x, V.sub(0))  
bc_bot_x = dlfx.fem.dirichletbc(dlfx.fem.Constant(domain, -1.0), boundary_dofs_bot_x, V.sub(0))
bcs = [bc_top_y, bc_bot_y,bc_top_x, bc_bot_x]


# bottom_facets = mesh.locate_entities(domain,fdim,bot)
# top_facets = mesh.locate_entities(domain,fdim,top_b)

# u_bot = np.array(-1.0, dtype=default_scalar_type)
# u_top = np.array(1.0, dtype=default_scalar_type)
# bc_bot = fem.dirichletbc(u_bot, 
#                      fem.locate_dofs_topological(V.sub(1),
#                                                  fdim, 
#                                                  bottom_facets), 
#                      V.sub(1))
# bc_top = fem.dirichletbc(u_top, 
#                      fem.locate_dofs_topological(V.sub(1),
#                                                  fdim, 
#                                                  top_facets), 
#                      V.sub(1))




# traction zero everywhere
T = fem.Constant(domain, default_scalar_type((0,0)))

ds = ufl.Measure("ds", domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def W(eps):
    val = 0.5*lambda_*ufl.tr(eps)**2+mu*ufl.inner(eps, eps)
    return val


def eshelby(u):
    Wenergy = W(epsilon(u))
    val = Wenergy*ufl.Identity(domain.geometry.dim)-ufl.grad(u).T*sigma(u)
    return val



u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f,v) * ufl.dx + ufl.dot(T,v) * ds



# solving
problem = LinearProblem(a,L,bcs=bcs,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.x.scatter_forward()

eta = ufl.TestFunction(V.sub(0))


# Gx = ufl.inner(eshelby(uh),ufl.grad(v)) * ufl.dx
Gx = ufl.dot(eshelby(uh),ufl.grad(eta))[0] * ufl.dx
# G = ufl.inner(sigma(u),ufl.grad(eta)) * ufl.dx
Gx_out = assemble_vector(dlfx.fem.form(Gx,dtype=default_scalar_type))

print(Gx_out.sum())

# G = ufl.inner(eshelby(uh),ufl.grad(eta)) * ufl.dx
Gy = ufl.dot(eshelby(uh),ufl.grad(eta))[1] * ufl.dx
# G = ufl.inner(sigma(u),ufl.grad(eta)) * ufl.dx
Gy_out = assemble_vector(dlfx.fem.form(Gy,dtype=default_scalar_type))

print(Gy_out.sum())

n = ufl.FacetNormal(domain)
Jx = (eshelby(uh)[0,0]*n[0] + eshelby(uh)[0,1]*n[1]) * ds
Jxa = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
print(Jxa)

JxVol = ufl.div(eshelby(uh))[0] * ufl.dx
JxVola = dlfx.fem.assemble_scalar(dlfx.fem.form(Jx))
print(JxVola)


# write results
uh.name = "u"
with dlfx.io.XDMFFile(comm, script_path + "/test.xdmf", "w") as xdmf_out:
    xdmf_out.write_mesh(domain)
    xdmf_out.write_function(uh)
