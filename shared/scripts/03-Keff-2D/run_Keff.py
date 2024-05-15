import dolfinx as dlfx
import os
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import numpy as np
from array import array
import ufl

import alex.heterogeneous as het
import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic
import math

import sys



script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)

# set MPI environment
comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)

with dlfx.io.XDMFFile(comm, os.path.join(script_path,'Keff_mesh.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh(name="Grid")
    mesh_tags = mesh_inp.read_meshtags(domain,name="Grid")
    
# mesh, cell_markers, facet_markers = gmshio.read_from_msh(os.path.join(script_path,'Keff_mesh.msh'), MPI.COMM_WORLD, gdim=2)
    

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)


# Material definition ##################################################
inclusion_marker = 1
matrix_marker = 0

la_inclusion = 2.0
la_matrix = 1.0
mu_inclusion = 2.0
mu_matrix = 1.0
gc_inclusion = 1.1
gc_matrix = 1.0

inclusion_cells = mesh_tags.find(inclusion_marker)
matrix_cells = mesh_tags.find(matrix_marker)

la = het.set_cell_function_heterogeneous_material(domain,la_inclusion, la_matrix, inclusion_cells, matrix_cells)
mu = het.set_cell_function_heterogeneous_material(domain,mu_inclusion, mu_matrix, inclusion_cells, matrix_cells)
gc = het.set_cell_function_heterogeneous_material(domain,gc_inclusion, gc_matrix, inclusion_cells, matrix_cells)

eta = dlfx.fem.Constant(domain, 0.001)
epsilon = dlfx.fem.Constant(domain, (y_max_all-y_min_all)*0.04)
Mob = dlfx.fem.Constant(domain, 10000.0)
iMob = dlfx.fem.Constant(domain, 1.0/Mob.value)

# Simulation parameters ####
Tend = 10.0
dt = 0.05

# Function space and FE functions ########################################################
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1,dim=2) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(),1) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

# define solution, restart, trial and test space
w =  dlfx.fem.Function(W)
u,s = w.split()
wrestart =  dlfx.fem.Function(W)
wm1 =  dlfx.fem.Function(W) # trial space
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

# Prepare boundary conditions ##########################################
crack_tip_start_location_x = 0.05*(x_max_all-x_min_all) + x_min_all
crack_tip_start_location_y = ((y_max_all-y_min_all) / 2.0 + y_min_all)
def crack(x):
    return np.logical_and(np.isclose(x[1], crack_tip_start_location_y,
                                     atol=(0.02*((y_max_all-y_min_all)))), x[0]<0.25) 

## define boundary conditions 
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

E_mod_matrix = alex.linearelastic.get_emod(la_matrix, mu_matrix)
K1 = dlfx.fem.Constant(domain, 1.5 * ufl.sqrt(gc_matrix*E_mod_matrix))
xtip = np.array([crack_tip_start_location_x, crack_tip_start_location_y])
xK1 = dlfx.fem.Constant(domain, xtip)
# wrapping as constant so it can be used in existing implemtation -> use matrix material here
la_matrix_constant = dlfx.fem.Constant(domain,la_matrix)
mu_matrix_constant = dlfx.fem.Constant(domain,mu_matrix)
bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,la_matrix_constant,mu_matrix_constant,epsilon.value)
bcs.append(bccrack)


phaseFieldProblem = pf.StaticPhaseFieldProblem2D(degradationFunction=pf.degrad_quadratic,
                                                   psisurf=pf.psisurf_from_function)

timer = dlfx.common.Timer()
def before_first_time_step():
    timer.start()
    
    # initialize s=1 
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    # prepare newton-log-file
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
    # prepare xdmf output 
    pp.write_mesh_and_get_outputfile_xdmf(domain, outputfile_xdmf_path, comm)

def before_each_time_step(t,dt):
    # report solution status
    if rank == 0:
        sol.print_time_and_dt(t,dt)
        
def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w,wm1=wm1,dw=dw,ddw=ddw,lam=la, mu = mu,
        Gc=gc,epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]


def top(x):
    return np.isclose(x[1],y_max_all)

def bottom(x):
    return np.isclose(x[1],y_min_all)

fdim = domain.topology.dim-1
top_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, top)
bottom_facets = dlfx.mesh.locate_entities_boundary(domain, fdim, bottom)

top_dofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0),fdim,top_facets)
bottom_dofs_x = dlfx.fem.locate_dofs_topological(W.sub(0).sub(0),fdim,bottom_facets)

top_dofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1),fdim,top_facets)
bottom_dofs_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1),fdim,bottom_facets)

def all(x):
    return np.full_like(x[0],True)

all_boundary_facets = dlfx.mesh.locate_entities_boundary(domain,fdim,all)

w_D = dlfx.fem.Function(W)


def get_bcs(t):
    v = (x_max_all-x_min_all) / Tend
    xK1 = dlfx.fem.Constant(domain,np.array([crack_tip_start_location_x + float(t) * v,0.0]))
    
    def get_carthesian_crack_tip_coordinates(x):
        x_tip = x[0] - xK1.value[0]
        y_tip = x[1] - xK1.value[1]
        return x_tip, y_tip

    def get_polar_coordinates(x_tip, y_tip):
        r = np.hypot(x_tip, y_tip)
        theta = np.arctan2(y_tip, x_tip)
        return r, theta

    nu = alex.linearelastic.get_nu(lam=la_matrix, mu=mu_matrix)
    K1 = 1.0 

    def u_x(x):
        x_tip, y_tip = get_carthesian_crack_tip_coordinates(x)
        r, theta = get_polar_coordinates(x_tip, y_tip)
        u_x = K1/ (2.0 * mu_matrix * math.sqrt(2.0 * math.pi)) * np.sqrt(r) * (3.0 - 4.0 * nu  -np.cos(theta)) * np.cos(0.5*theta)
        return u_x

    def u_y(x):
        x_tip, y_tip = get_carthesian_crack_tip_coordinates(x)
        r, theta = get_polar_coordinates(x_tip, y_tip)
        u_y = K1/ (2.0 * mu_matrix * math.sqrt(2.0 * math.pi)) * np.sqrt(r) * (3.0 - 4.0 * nu  -np.cos(theta)) * np.sin(0.5*theta)
        return u_y
    
    w_D.sub(0).sub(0).interpolate(u_x)
    w_D.sub(0).x.scatter_forward()
    w_D.sub(0).sub(1).interpolate(u_y)
    w_D.sub(0).x.scatter_forward()
    
    dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, all_boundary_facets)
    bc = dlfx.fem.dirichletbc(w_D,dofs_at_boundary)
    bcs = [bc]
    
    # bc_top_x = dlfx.fem.dirichletbc(0.0,top_dofs_x,W.sub(0).sub(0))
    # bc_bottom_x = dlfx.fem.dirichletbc(0.0,bottom_dofs_x,W.sub(0).sub(0))
    # bc_top_y = dlfx.fem.dirichletbc(float(t),top_dofs_y,W.sub(0).sub(1))
    # bc_bottom_y = dlfx.fem.dirichletbc(float(-t),bottom_dofs_y,W.sub(0).sub(1))
    # bcs = [bc_top_x, bc_bottom_x,bc_top_y, bc_bottom_y]
    
    
    # v_crack = (x_max_all - x_min_all)/Tend
    # xtip = np.array([crack_tip_start_location_x + v_crack * t, crack_tip_start_location_y])
    # alex.os.mpi_print("Position K-Field: x: {0:.2e} y: {1:.2e}".format(xtip[0], xtip[1]),rank)
    # xK1 = dlfx.fem.Constant(domain, xtip)
    # bcs = bc.get_total_surfing_boundary_condition_at_box(domain,comm,W,0,K1,xK1,la_matrix_constant,mu_matrix_constant,0.0*epsilon.value)
    # irreversibility
    if(abs(t)> sys.float_info.epsilon*5): # dont do before first time step
        bcs.append(pf.irreversibility_bc(domain,W,wm1))
    
    # initial bc for crack        
    bcs.append(bccrack)
    return bcs

n = ufl.FacetNormal(domain)
s_zero_for_tracking = pp.get_s_zero_field_for_tracking(domain)
def after_timestep_success(t,dt,iters):
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
    
    eshelby = phaseFieldProblem.getEshelby(w,eta,la,mu)
    Jx, Jy = alex.linearelastic.get_J_2D(eshelby,n,ufl.ds,comm)
    Jx_vol, Jy_vol = alex.linearelastic.get_J_2D_volume_integral(eshelby,ufl.dx,comm)
    
    alex.os.mpi_print(pp.getJString2D(Jx,Jy),rank)
    
    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    
   
    # s_zero_for_tracking.x.array[:] = s.collapse().x.array[:]
    s_zero_for_tracking.interpolate(s)
    max_x, max_y, min_x, min_y  = pp.crack_bounding_box_2D(domain, pf.get_dynamic_crack_locator_function(wm1,s_zero_for_tracking),comm)
    if rank == 0:
        x_ct = max_x
        print("Crack tip position x: " + str(x_ct))
        pp.write_to_graphs_output_file(outputfile_graph_path,t,Jx, Jy, Jx_vol, Jy_vol, x_ct)

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
        pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx", "Jy", "Jx_vol", "Jy_vol", "x_ct"])
        
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
    comm=comm
)