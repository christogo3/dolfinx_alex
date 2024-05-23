
# +
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

import gmsh
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc
from petsc4py import PETSc

import dolfinx

import alex.plasticity
import alex.solution as sol
import os 
import alex.postprocessing as pp
import sys

print(f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

hsize = 0.2

Re = 1.3
Ri = 1.0
# -

# We then model a quarter of cylinder using `Gmsh` similarly to the [](/tours/linear_problems/axisymmetric_elasticity/axisymmetric_elasticity.md) demo.

# + tags=["hide-input"]
gmsh.initialize()
gdim = 2
model_rank = 0
if MPI.COMM_WORLD.rank == 0:
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    gmsh.model.add("Model")

    geom = gmsh.model.geo
    center = geom.add_point(0, 0, 0)
    p1 = geom.add_point(Ri, 0, 0)
    p2 = geom.add_point(Re, 0, 0)
    p3 = geom.add_point(0, Re, 0)
    p4 = geom.add_point(0, Ri, 0)

    x_radius = geom.add_line(p1, p2)
    outer_circ = geom.add_circle_arc(p2, center, p3)
    y_radius = geom.add_line(p3, p4)
    inner_circ = geom.add_circle_arc(p4, center, p1)

    boundary = geom.add_curve_loop([x_radius, outer_circ, y_radius, inner_circ])
    surf = geom.add_plane_surface([boundary])

    geom.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    gmsh.model.addPhysicalGroup(gdim, [surf], 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [x_radius], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [y_radius], 2, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [inner_circ], 3, name="inner")

    gmsh.model.mesh.generate(gdim)

domain, _, facets = io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
)
gmsh.finalize()
# -

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)


E = fem.Constant(domain, 70e3)  # in MPa
nu = fem.Constant(domain, 0.3)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2.0 / (1 + nu)
sig0 = fem.Constant(domain, 250.0)  # yield strength in MPa
Et = E / 100.0  # tangent modulus
H = E * Et / (E - Et)  # hardening modulus


deg_u = 2
shape = (gdim,)
V = fem.functionspace(domain, ("P", deg_u, shape))
# Ve = ufl.VectorElement("P", domain.ufl_cell(), 2) # displacements
# V =

# +
Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
bottom_dofsy = fem.locate_dofs_topological((V.sub(1), Vy), gdim - 1, facets.find(1))
top_dofsx = fem.locate_dofs_topological((V.sub(0), Vx), gdim - 1, facets.find(2))


# used for post-processing
def bottom_inside(x):
    return np.logical_and(np.isclose(x[0], Ri), np.isclose(x[1], 0))


bottom_inside_dof = fem.locate_dofs_geometrical((V.sub(0), Vx), bottom_inside)[0]

u0x = fem.Function(Vx)
u0y = fem.Function(Vy)
bcs = [
    fem.dirichletbc(u0x, top_dofsx, V.sub(0)),
    fem.dirichletbc(u0y, bottom_dofsy, V.sub(1)),
]

n = ufl.FacetNormal(domain)
q_lim = float(2 / np.sqrt(3) * np.log(Re / Ri) * sig0)

loading = fem.Constant(domain, 0.0)
# -

# ### Internal state variables and `Quadrature` elements
#
# When dealing with nonlinear constitutive models, internal state variables such as plastic strains represent the history seen by the material and have to be stored in some way. We choose here to represent them using `Quadrature` elements. This choice will make it possible to express the complex non-linear material constitutive equation at the Gauss points only, without involving any interpolation of non-linear expressions throughout the element. It will ensure an optimal convergence rate for the Newton-Raphson method, see chap. 26 of {cite:p}`logg2012fenicsbook`. We will need `Quadrature` elements for 4-dimensional vectors and scalars, the number of Gauss points will be determined by the required degree `deg_quad` of the `Quadrature` element, see the [](/tips/quadrature_schemes/quadrature_schemes.md) tour for more details on the choice of quadrature rules.
#
# ```{note}
# We point out that, although the problem is 2D, plastic strain still occur in the transverse $zz$ direction. This will require us to keep track of the out-of-plane $zz$ components of stress/strain states.
# ```

deg_quad = 2  # quadrature degree for internal state variable representation
# W0e = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
# )
# We = basix.ufl.quadrature_element(
#     domain.basix_cell(), value_shape=(alex.plasticity.get_history_field_dimension_for_symmetric_second_order_tensor(gdim),), scheme="default", degree=deg_quad
# )

W0e = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=deg_quad, quad_scheme="default")
We = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=deg_quad,dim=4, quad_scheme="default")





W = fem.functionspace(domain, We)
W0 = fem.functionspace(domain, W0e)

# Various functions are defined to keep track of the current internal state and currently computed increments.

# +
sig_np1 = fem.Function(W)
sig_n = fem.Function(W)
N_np1 = fem.Function(W)
beta = fem.Function(W0)
alpha_n = fem.Function(W0, name="Cumulative_plastic_strain")
dGamma = fem.Function(W0)
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="Iteration_correction")
Du = fem.Function(V, name="Current_increment")
v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

# P0 = fem.functionspace(domain, ("DG", 0))
# p_avg = fem.Function(P0, name="Plastic_strain")



eps = alex.plasticity.eps_as_3D_tensor_function(gdim)



as_3D_tensor = alex.plasticity.from_history_field_to_3D_tensor_mapper(gdim)



# def to_vect(X):
#     return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])

to_vect = alex.plasticity.to_history_field_vector_mapper(gdim)


ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"},
)


# During the Newton-Raphson iterations, we will have to interpolate some `ufl` expressions at quadrature points to update the corresponding functions. We define the `interpolate_quadrature` function to do so. We first get the quadrature points location in the reference element and then use the `fem.Expression.eval` to evaluate the expression on all cells.

# +


quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)



Nitermax, tol = 200, 1e-6  # parameters of the Newton-Raphson procedure
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr + 1)[1:] ** 0.5


results = np.zeros((Nincr + 1, 3))


def after_last_time_step():
    if len(bottom_inside_dof) > 0:  # test if proc has dof
        plt.plot(results[:, 0], results[:, 1], "-oC3")
        plt.xlabel("Displacement of inner boundary")
        plt.ylabel(r"Applied pressure $q/q_{lim}$")

        # plt.show()
    # Save the figure to a file
        plt.savefig('/home/scripts/05-plasticity/plot.png')

    if len(bottom_inside_dof) > 0:
        plt.bar(np.arange(Nincr + 1), results[:, 2], color="C2")
        plt.xlabel("Loading step")
        plt.ylabel("Number of iterations")
        plt.xlim(0)

        # plt.show()
    # Save the figure to a file
        plt.savefig('/home/scripts/05-plasticity/plot2.png')


def get_residuum_and_tangent():
    return alex.plasticity.get_residual_and_tangent(n, loading, as_3D_tensor(sig_np1), u_, v, eps, ds(3), dx, lmbda,mu,as_3D_tensor(N_np1),beta,H)



def get_bcs(t):
    loading.value = t * q_lim
    return bcs
    

def before_each_time_step(t):
    Du.x.array[:] = 0
    
def before_first_time_step():
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
    
    # we set all functions to zero before entering the loop in case we would like to reexecute this code cell
    sig_np1.vector.set(0.0)
    sig_n.vector.set(0.0)
    alpha_n.vector.set(0.0)
    u.vector.set(0.0)
    N_np1.vector.set(0.0)
    beta.vector.set(0.0)
    
    return

def after_iteration():
    # Du.vector.axpy(1, du.vector)  # Du = Du + 1*du
    # Du.x.scatter_forward()

    dEps = eps(Du)
    sig_np1_expr, N_np1_expr, beta_expr, dGamma_expr = alex.plasticity.constitutive_update(dEps, as_3D_tensor(sig_n), alpha_n,sig0,H,lmbda,mu)
    sig_np1_expr = to_vect(sig_np1_expr)
    N_np1_expr = to_vect(N_np1_expr)

        # interpolate the new stresses and internal state variables
    sig_np1.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, sig_np1_expr)
    N_np1.x.array[:]  = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,N_np1_expr)
    beta.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,beta_expr)
    return dGamma_expr

def after_time_step_success(t, i, niter, parameter_after_last_iteration):
        dGamma_expr = parameter_after_last_iteration
    # Update the displacement with the converged increment
        u.vector.axpy(1, Du.vector)  # u = u + 1*Du
        u.x.scatter_forward()
        
        pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm)

    # Update the previous plastic strain
        # _ , _, _, dGamma_expr = alex.plasticity.constitutive_update(dEps, as_3D_tensor(sig_n), alpha_n,sig0,H,lmbda,mu)
        dGamma.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, dGamma_expr )
        alpha_n.vector.axpy(1, dGamma.vector)
        
        # pp.write_field(domain,outputfile_xdmf_path,alpha_n,t,comm)

    # Update the previous stress
        sig_n.x.array[:] = sig_np1.x.array[:]

        if len(bottom_inside_dof) > 0:  # test if proc has dof
            results[i + 1, :] = (u.x.array[bottom_inside_dof[0]], t, niter)
    


sol.solve_with_newton(domain, Du, du, Nitermax, tol, load_steps,  
                                  before_first_timestep_hook=before_first_time_step, 
                                  after_last_timestep_hook=after_last_time_step, 
                                  before_each_timestep_hook=before_each_time_step, 
                                  after_iteration_hook=after_iteration, 
                                  get_residuum_and_tangent=get_residuum_and_tangent, 
                                  get_bcs=get_bcs, 
                                  after_timestep_success_hook=after_time_step_success,
                                  comm=comm)

# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
