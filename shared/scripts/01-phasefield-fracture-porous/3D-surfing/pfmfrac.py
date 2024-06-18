from typing import Callable, Union

import basix.ufl
import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
import dolfinx.plot as plot
import pyvista
from mpi4py import MPI
from petsc4py import PETSc as petsc


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

from  dolfinx.cpp.la import InsertMode

from dolfinx.fem.petsc import assemble_vector
import petsc4py
import basix


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
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

Tend = 1.0
dt = 0.2

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

# Ve = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
# Te = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(1,))
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))
# W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Te]))

# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.5), x[0]<0.25) 

# define boundary condition on top and bottom
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)

# V = W.sub(0).collapse()
# S = W.sub(1).collapse()
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)


# def crack_bounding_box_3D(domain: dlfx.mesh.Mesh, crack_locator_function: Callable):
#     '''
#     operates on nodes not on DOF locations
    
#     returns the bounding box in which all cracks are contained
#     '''
#     xx  = np.array(domain.geometry.x).T
#     crack_indices = crack_locator_function(xx)

#     crack_x = xx.T[crack_indices]

#     max_x = np.max(crack_x.T[0])
#     max_y = np.max(crack_x.T[1])
#     max_z = np.max(crack_x.T[2])

#     min_x = np.min(crack_x.T[0])
#     min_y = np.min(crack_x.T[1])
#     min_z = np.min(crack_x.T[2])
    
#     return max_x, max_y, max_z, min_x, min_y, min_z

# max_x, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, crack)

# coords = W.sub(1).tabulate_dof_coordinates()[crackdofs]
# print(np.max(coords))

bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

E_mod = alex.linearelastic.get_emod(lam.value, mu.value)
K1 = dlfx.fem.Constant(domain, 1.5 * math.sqrt(Gc.value*E_mod))
xtip = np.array([0.25, 0.5])
xK1 = dlfx.fem.Constant(domain, xtip)

bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
# bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
bcs.append(bccrack)                   


# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

def before_first_time_step():
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm,meshtags=cell_tags)
    # xdmfout.write_meshtags(cell_tags, domain.geometry)
    
    if rank == 0:
        pp.screenshot_of_subdomain(script_path, domain, cell_tags, 0)


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


s_zero_for_tracking = pp.get_s_zero_field_for_tracking(domain)

def get_bcs(t):
    v_crack = 0.75/0.4
    xtip = np.array([0.25 + v_crack * t, 0.5])
    xK1 = dlfx.fem.Constant(domain, xtip)

    if(t <= 0.5):
        bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,lam,mu,epsilon.value)
    else:
        bcs = []
    # bcs = bc.get_total_linear_displacement_boundary_condition_at_box(domain, comm, W,0,eps_mac)
    
    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
        
    # initial conditions    
    bcs.append(bccrack)
    # can be updated here
    return bcs

n = ufl.FacetNormal(domain)
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm),5)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)

def in_cylinder_around_crack_tip(x):
        return np.array((x.T[0] - 0.5) ** 2 + (x.T[1] - 0.5) ** 2 < (epsilon.value*6)**2, dtype=np.int32)
dx_in_cylinder, cell_tags = pp.ufl_integration_subdomain(domain, in_cylinder_around_crack_tip)




def after_timestep_success(t,dt,iters):
    
    
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
             
    eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    
    divEshelby = ufl.div(eshelby)
    pp.write_vector_fields(domain=domain,comm=comm,vector_fields_as_functions=[divEshelby],
                            vector_field_names=["Ge"], 
                            outputfile_xdmf_path=outputfile_xdmf_path,t=t)
    
    J3D_glob_x, J3D_glob_y, J3D_glob_z = alex.linearelastic.get_J_3D(eshelby, ds=ds(5), n=n, comm=comm)
    
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
    
    J_from_nodal_forces = alex.linearelastic.get_J_from_nodal_forces(eshelby,W,ufl.dx,comm)
    
    J3D_glob_x_ii, J3D_glob_y_ii, J3D_glob_z_ii = alex.linearelastic.get_J_3D_volume_integral(eshelby, ufl.dx,comm)
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x_ii, J3D_glob_y_ii, J3D_glob_z_ii))
       
    
    cohesiveConfStress = alex.phasefield.getCohesiveConfStress(s,Gc,epsilon)
    G_ad_x_glob, G_ad_y_glob, G_ad_z_glob = alex.phasefield.get_G_ad_3D_volume_integral(cohesiveConfStress, ufl.dx,comm)
    
    
    if rank == 0:
        print(pp.getJString(G_ad_x_glob, G_ad_y_glob, G_ad_z_glob))
        
    dissipativeConfForce = alex.phasefield.getDissipativeConfForce(s,sm1,Mob,dt)
    G_dis_x_glob, G_dis_y_glob, G_dis_z_glob = alex.phasefield.getDissipativeConfForce_volume_integral(dissipativeConfForce,ufl.dx,comm)
        
    if rank == 0:
        print(pp.getJString(G_dis_x_glob, G_dis_y_glob, G_dis_z_glob))
        pp.write_to_graphs_output_file(outputfile_graph_path,t,J3D_glob_x, J3D_glob_x_ii, J_from_nodal_forces[0], G_dis_x_glob, G_ad_x_glob)


    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
   
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking.interpolate(s)
    max_x, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking),comm)
    if rank == 0:
        print("Crack tip position x: " + str(max_x))
    
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
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx_surf", "Ge_x_div", "Jx_nodal_forces", "Gad_x", "Gdis_x"])
        
        # cleanup only necessary on cluster
        # results_folder_path = alex.os.create_results_folder(script_path)
        # alex.os.copy_contents_to_results_folder(script_path,results_folder_path)

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
    comm=comm,
    print=True
)


