import alex.linearelastic
import dolfinx as dlfx
import dolfinx.plot as plot
import pyvista
from mpi4py import MPI
from petsc4py import PETSc

import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# set FEniCSX log level
# dlfx.log.set_log_level(log.LogLevel.INFO)
# dlfx.log.set_output_file('xxx.log')

# set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# mesh 
N = 16 

# generate domain
#domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

Tend = 0.4
dt = 0.05

# elastic constants
lam = dlfx.fem.Constant(domain, 10.0)
mu = dlfx.fem.Constant(domain, 10.0)

# residual stiffness
eta = dlfx.fem.Constant(domain, 0.001)

# phase field parameters
Gc = dlfx.fem.Constant(domain, 1.0)
epsilon = dlfx.fem.Constant(domain, 0.05)
Mob = dlfx.fem.Constant(domain, 1.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)


# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 2) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.5), x[0]<0.25) 


# # define boundary condition on top and bottom
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

# def surfing_boundary(x):

E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
K1 = dlfx.fem.Constant(domain, 1.5 * math.sqrt(Gc.value*E_mod))
xtip = np.array([0.25, 0.5])
xK1 = dlfx.fem.Constant(domain, xtip)

bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
# bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
bcs.append(bccrack)


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

def before_first_time_step():
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
    # prepare xdmf output 
    pp.write_mesh_and_get_outputfile_xdmf(domain, outputfile_xdmf_path, comm,meshtags=cell_tags)
    # xdmfout.write_meshtags(cell_tags, domain.geometry)
    
    # if rank == 0:
    plot_vtk(0)


def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)

        
phaseFieldProblem = pf.StaticPhaseFieldProblem3D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf)

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=lam, mu = mu,
        Gc=Gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]
    
def get_bcs(t):
    v_crack = 0.75/0.4
    xtip = np.array([0.25 + v_crack * t, 0.5])
    xK1 = dlfx.fem.Constant(domain, xtip)

    bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
    # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
    bcs.append(bccrack)
    # can be updated here
    return bcs

n = ufl.FacetNormal(domain)
# from functools import reduce
# def locator(x):
#     tol = 4.0 * epsilon.value
#     return reduce(np.logical_and,[np.isclose(x[0], 0.25, atol=tol), np.isclose(x[1], 0.5, atol=tol)])
# facets = dlfx.mesh.locate_entities(domain, domain.topology.dim, locator)
# facet_indices, facet_markers = [], []
# facet_indices.append(facets)
# facet_markers.append(np.full_like(facets, 1))
# facet_indices = np.hstack(facet_indices).astype(np.int32)
# facet_markers = np.hstack(facet_markers).astype(np.int32)
# sorted_facets = np.argsort(facet_indices)
# facet_tag = dlfx.mesh.meshtags(domain, domain.topology.dim, facet_indices[sorted_facets], facet_markers[sorted_facets])

def in_cylinder_around_crack_tip(x):
        return np.array((x.T[0] - 0.25) ** 2 + (x.T[1] - 0.5) ** 2 < (epsilon.value*6)**2, dtype=np.int32)

# Create cell tags - if midpoint is inside circle, it gets value 1,
# otherwise 0
num_cells = domain.topology.index_map(domain.topology.dim).size_local
midpoints = dlfx.mesh.compute_midpoints(domain, domain.topology.dim, np.arange(num_cells, dtype=np.int32))
cell_tags = dlfx.mesh.meshtags(domain, domain.topology.dim, np.arange(num_cells), in_cylinder_around_crack_tip(midpoints))


def plot_vtk(t):
    pyvista.start_xvfb()
    
    # Create VTK mesh
    cells, types, x = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvista grid
    grid.cell_data["Marker"] = cell_tags.values
    grid.set_active_scalars("Marker")

    # Create a plotter with two subplots, and add mesh tag plot to the
    # first sub-window
    subplotter = pyvista.Plotter(off_screen=True, shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Mesh with markers", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.show_axes()
    
    # We can visualize subsets of data, by creating a smaller topology
    # (set of cells). Here we create VTK mesh data for only cells with
    # that tag '1'.
    cells, types, x = plot.vtk_mesh(domain, entities=cell_tags.find(1))

    # Add this grid to the second plotter window
    sub_grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 1)
    subplotter.add_text("Subset of mesh", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")
    subplotter.show_axes()
    subplotter.screenshot(
            script_path + "/2D_markers"+str(t)+".png", transparent_background=False, window_size=[2 * 800, 800]
    )
    

def after_timestep_success(t,dt,iters):
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
         
    # compute J-Integral
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    J3D_loc_x, J3D_loc_y, J3D_loc_z = alex.linearelastic.get_J_3D(eshelby, n)
    
    comm.Barrier()
    J3D_glob_x = comm.allreduce(J3D_loc_x, op=MPI.SUM)
    J3D_glob_y = comm.allreduce(J3D_loc_y, op=MPI.SUM)
    J3D_glob_z = comm.allreduce(J3D_loc_z, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        
    dxx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    J3D_loc_x_i, J3D_loc_y_i, J3D_loc_z_i = alex.linearelastic.get_J_3D_volume_integral(eshelby, dxx(1))
    
    comm.Barrier()
    J3D_glob_x_i = comm.allreduce(J3D_loc_x_i, op=MPI.SUM)
    J3D_glob_y_i = comm.allreduce(J3D_loc_y_i, op=MPI.SUM)
    J3D_glob_z_i = comm.allreduce(J3D_loc_z_i, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x_i, J3D_glob_y_i, J3D_glob_z_i))
        
    # # dxx = ufl.Measure("dx", domain=domain, subdomain_data=facet_tag)
    # J3D_loc_x_i, J3D_loc_y_i, J3D_loc_z_i = alex.linearelastic.get_J_3D_volume_integral(eshelby, dxx)
    
    # comm.Barrier()
    # J3D_glob_x_i = comm.allreduce(J3D_loc_x_i, op=MPI.SUM)
    # J3D_glob_y_i = comm.allreduce(J3D_loc_y_i, op=MPI.SUM)
    # J3D_glob_z_i = comm.allreduce(J3D_loc_z_i, op=MPI.SUM)
    # comm.Barrier()
    
    # if rank == 0:
    #     print(pp.getJString(J3D_glob_x_i, J3D_glob_y_i, J3D_glob_z_i))

    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
def after_timestep_restart(t,dt,iters):
    w.x.array[:] = wrestart.x.array[:]
    
    
def after_last_timestep():
    # stopwatch stop
    timer.stop()

    # report runtime to screen
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path,runtime)
    

sol.solve_with_newton_adaptive_time_stepping(
    domain,
    w,
    Tend,
    dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm
)

