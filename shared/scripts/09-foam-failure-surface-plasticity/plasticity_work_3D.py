
# +
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

import alex.util
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
import alex.boundaryconditions as bc


# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# mesh 
N = 16 
domain = dolfinx.mesh.create_unit_cube(comm,N,N,N,cell_type=dolfinx.mesh.CellType.hexahedron)
gdim = 3


n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain)
loading = fem.Constant(domain, 0.0)

if rank == 0:
    alex.util.print_dolfinx_version()

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


deg_u = 1
# shape = (gdim,)
# V = fem.functionspace(domain, ("P", deg_u, shape))
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), deg_u) # displacements
V = dolfinx.fem.FunctionSpace(domain,Ve)


def left(x):
    return np.isclose(x[0],0)
    
def right(x):
    return np.isclose(x[0],1)
    
fdim = domain.geometry.dim-1
left_facets = dolfinx.mesh.locate_entities_boundary(domain,fdim,left)
right_facets = dolfinx.mesh.locate_entities_boundary(domain,fdim,right)


left_dofs_x = dolfinx.fem.locate_dofs_topological(V.sub(0),fdim,left_facets)
left_dofs_y = dolfinx.fem.locate_dofs_topological(V.sub(1),fdim,left_facets)
left_dofs_z = dolfinx.fem.locate_dofs_topological(V.sub(2),fdim,left_facets)
    
    
right_dofs_x = dolfinx.fem.locate_dofs_topological(V.sub(0),fdim,right_facets)
right_dofs_y = dolfinx.fem.locate_dofs_topological(V.sub(1),fdim,right_facets)
right_dofs_z = dolfinx.fem.locate_dofs_topological(V.sub(2),fdim,right_facets)
    
# bc_left_x = dolfinx.fem.dirichletbc(-0.1,left_dofs_x, V.sub(0))
bc_left_y = dolfinx.fem.dirichletbc(0.0,left_dofs_y, V.sub(1))
bc_left_z = dolfinx.fem.dirichletbc(0.0,left_dofs_z, V.sub(2))
     
# bc_right_x = dolfinx.fem.dirichletbc(0.1,right_dofs_x, V.sub(0))
bc_right_y = dolfinx.fem.dirichletbc(0.0,right_dofs_y, V.sub(1))
bc_right_z = dolfinx.fem.dirichletbc(0.0,right_dofs_z, V.sub(2))
    

u = fem.Function(V, name="Total_displacement")

du = fem.Function(V, name="Iteration_correction")
Du = fem.Function(V, name="Current_increment")
DuRestart =  fem.Function(V, name="Current_increment_restart")
v = ufl.TrialFunction(V)
u_tf = ufl.TestFunction(V)

deg_quad = 2  # quadrature degree for internal state variable representation
sig_np1, sig_n, N_np1, beta, alpha_n, dGamma = alex.plasticity.define_internal_state_variables(gdim, domain, deg_quad,quad_scheme="default")
dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)

eps = alex.plasticity.eps_as_3D_tensor_function(gdim)
as_3D_tensor = alex.plasticity.from_history_field_to_3D_tensor_mapper(gdim)
to_vect = alex.plasticity.to_history_field_vector_mapper(gdim)

# Nitermax, tol = 200, 1e-6  # parameters of the Newton-Raphson procedure
# Nincr = 200
# load_steps = np.linspace(0, 10.1, Nincr + 1)[1:] ** 0.5

# results = np.zeros((Nincr + 1, 3))
def after_last_time_step():
    return

def get_residuum_and_tangent(dt):
    return alex.plasticity.get_residual_and_tangent(n, loading, as_3D_tensor(sig_np1), u_tf, v, eps, ds, dx, lmbda,mu,as_3D_tensor(N_np1),beta,H)

def get_bcs(t):
    if rank == 0:
        print(f"Time: {t}")
        sys.stdout.flush()
    
    vertices_at_corner = dolfinx.mesh.locate_entities(domain,fdim-1,bc.get_corner_of_box_as_function(domain,comm))
    # dofs_at_corner_x = dolfinx.fem.locate_dofs_topological(V.sub(0),fdim-1,vertices_at_corner)
    # bc_corner_x = dolfinx.fem.dirichletbc(0.0,dofs_at_corner_x,V.sub(0))
    dofs_at_corner_y = dolfinx.fem.locate_dofs_topological(V.sub(1),fdim-1,vertices_at_corner)
    bc_corner_y = dolfinx.fem.dirichletbc(0.0,dofs_at_corner_y,V.sub(1))
    dofs_at_corner_z = dolfinx.fem.locate_dofs_topological(V.sub(2),fdim-1,vertices_at_corner)
    bc_corner_z = dolfinx.fem.dirichletbc(0.0,dofs_at_corner_z,V.sub(2))
    # bcs = [bc_corner_x,  bc_corner_y, bc_corner_z]    
    
    bc_left_x = dolfinx.fem.dirichletbc(-0.0001*t,left_dofs_x, V.sub(0))     
    bc_right_x = dolfinx.fem.dirichletbc(0.0001*t,right_dofs_x, V.sub(0))
    bcs = [bc_left_x, bc_right_x, bc_corner_y, bc_corner_z]
    return bcs
    
def before_each_time_step(t,dt):
    return
    Du.x.array[:] = 0 # TODO change maybe for better convergence
    
def before_first_time_step():
    DuRestart.x.array[:] = np.ones_like(DuRestart.x.array[:])
    
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
    
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    
    # we set all functions to zero before entering the loop in case we would like to reexecute this code cell
    sig_np1.vector.set(0.0)
    sig_n.vector.set(0.0)
    alpha_n.vector.set(0.0)
    u.vector.set(0.0)
    N_np1.vector.set(0.0)
    beta.vector.set(0.0)
    return

def after_iteration():
    dEps = eps(Du)
    sig_np1_expr, N_np1_expr, beta_expr, dGamma_expr = alex.plasticity.constitutive_update(dEps, as_3D_tensor(sig_n), alpha_n,sig0,H,lmbda,mu)
    sig_np1_expr = to_vect(sig_np1_expr)
    N_np1_expr = to_vect(N_np1_expr)

        # interpolate the new stresses and internal state variables
    sig_np1.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, sig_np1_expr)
    N_np1.x.array[:]  = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,N_np1_expr)
    beta.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points,beta_expr)
    return dGamma_expr

def after_time_step_success(t, dt,iters, parameter_after_last_iteration):
        if rank == 0:
            sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
        dGamma_expr = parameter_after_last_iteration
    # Update the displacement with the converged increment
        u.vector.axpy(1, Du.vector)  # u = u + 1*Du
        u.x.scatter_forward()
        
        # pp.write_vector_field(domain,outputfile_xdmf_path,u,t,comm) # buggy
        
        pp.write_vector_fields(domain,comm,[u], ["u"],outputfile_xdmf_path,t)

    # Update the previous plastic strain
        dGamma.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, dGamma_expr )
        dGamma.x.scatter_forward()
        alpha_n.vector.axpy(1, dGamma.vector)
        alpha_n.x.scatter_forward()
                
        pp.write_scalar_fields(domain, comm, [alpha_n], ["alpha_n"], outputfile_xdmf_path, t)

    # Update the previous stress
        sig_n.x.array[:] = sig_np1.x.array[:]
        
        DuRestart.x.array[:] = Du.x.array[:] 
        

def after_timestep_restart(t,dt,iters):
        Du.x.array[:] = DuRestart.x.array[:]
    
# sol.solve_with_newton(domain, Du, du, Nitermax, tol, load_steps,  
#                                   before_first_timestep_hook=before_first_time_step, 
#                                   after_last_timestep_hook=after_last_time_step, 
#                                   before_each_timestep_hook=before_each_time_step, 
#                                   after_iteration_hook=after_iteration, 
#                                   get_residuum_and_tangent=get_residuum_and_tangent, 
#                                   get_bcs=get_bcs, 
#                                   after_timestep_success_hook=after_time_step_success,
#                                   comm=comm,
#                                   print_bool=True)

# sol.solve_with_newton_b(domain, Du, du, Nitermax, tol, load_steps, 
#                                   Tend=10.1,
#                                   dt=0.05, 
#                                   before_first_timestep_hook=before_first_time_step, 
#                                   after_last_timestep_hook=after_last_time_step, 
#                                   before_each_timestep_hook=before_each_time_step, 
#                                   after_iteration_hook=after_iteration, 
#                                   get_residuum_and_tangent=get_residuum_and_tangent, 
#                                   get_bcs=get_bcs, 
#                                   after_timestep_success_hook=after_time_step_success,
#                                   comm=comm,
#                                   print_bool=True)

sol.solve_with_custom_newton_adaptive_time_stepping(domain=domain,
                                                    sol=Du,
                                                    dsol=du,
                                                    Tend=10.1,
                                                    dt=0.05,
                                                    before_first_timestep_hook=before_first_time_step,
                                                    after_last_timestep_hook=after_last_time_step,
                                                    after_timestep_restart_hook=after_timestep_restart,
                                                    after_iteration_hook=after_iteration,
                                                    get_residuum_and_gateaux=get_residuum_and_tangent,
                                                    get_bcs=get_bcs,
                                                    after_timestep_success_hook=after_time_step_success,
                                                    comm=comm,
                                                    print_bool=True
                                                    )
