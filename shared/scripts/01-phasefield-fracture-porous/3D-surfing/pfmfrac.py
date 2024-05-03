from typing import Callable, Union
import alex.linearelastic
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


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_J_path = alex.os.outputfile_J_full_path(script_path,script_name_without_extension)
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

Tend = 0.2
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
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

# define crack by boundary
def crack(x):
    return np.logical_and(np.isclose(x[1], 0.25), x[0]<0.5) 

# define boundary condition on top and bottom
fdim = domain.topology.dim -1
crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
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
        pp.prepare_J_output_file(outputfile_J_path)
    # prepare xdmf output 
    pp.write_mesh_and_get_outputfile_xdmf(domain, outputfile_xdmf_path, comm,meshtags=cell_tags)
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
external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm),5)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)

def in_cylinder_around_crack_tip(x):
        return np.array((x.T[0] - 0.5) ** 2 + (x.T[1] - 0.5) ** 2 < (epsilon.value*6)**2, dtype=np.int32)
dxx, cell_tags = pp.ufl_integration_subdomain(domain, in_cylinder_around_crack_tip)




def after_timestep_success(t,dt,iters):
    w.x.scatter_forward() # synchronization between processes? 
    pp.write_phasefield_mixed_solution(domain,outputfile_xdmf_path, w, t, comm)
    
    
    def getEshelby( u: any, s: any, eta: dlfx.fem.Constant, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
        
        def deg(s):
            return pf.degrad_quadratic(s,eta)
        def sigma_wo_deg(u):
            return lam.value * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(3) + 2.0 * mu.value * ufl.sym(ufl.grad(u)) 
        
        def psiel(u):
            return 0.5*ufl.inner(sigma_wo_deg(u), ufl.sym(ufl.grad(u)))
        # u,s = ufl.split(w)
        return deg(s) * psiel(u) * ufl.Identity(3) - ufl.dot(ufl.grad(u).T, deg(s) * sigma_wo_deg(u))
    
    
    # write to newton-log-file
    if rank == 0:
        sol.write_to_newton_logfile(logfile_path,t,dt,iters)
         
    # compute J-Integral
    # eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    u,s = ufl.split(w)
    du, dv = ufl.split(dw)
    
    F = dlfx.fem.FunctionSpace(domain, Te)
    # sg = dlfx.fem.Function(F)
    # sg.interpolate(s)
    dv = ufl.TestFunction(F)
    
    V = dlfx.fem.FunctionSpace(domain, Ve)
    # ug = dlfx.fem.Function(V)
    # ug.interpolate(u)
    
    # u, s = w.split()   
    # Ue = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    # Se = ufl.FiniteElement('CG', domain.ufl_cell(), 1)
    
    # U = dlfx.fem.FunctionSpace(domain, Ue)
    # S = dlfx.fem.FunctionSpace(domain, Se)
    # sg = dlfx.fem.Function(S)
    # ug = dlfx.fem.Function(U)
    
    # sg.interpolate(s)
    # ug.interpolate(u)
    
    eshelby = getEshelby(u,s,eta,lam,mu)
    
    
    
    # eshelby = phaseFieldProblem.getEshelby(w,eta,lam,mu)
    
    divEshelby = ufl.div(eshelby)
    pp.write_vector_fields(domain=domain,comm=comm,vector_fields_as_functions=[divEshelby],
                            vector_field_names=["Ge"], 
                            outputfile_xdmf_path=outputfile_xdmf_path,t=t)
    
    J3D_loc_x, J3D_loc_y, J3D_loc_z = alex.linearelastic.get_J_3D(eshelby, ds=ds(5), outer_normal=n)
    
    comm.Barrier()
    J3D_glob_x = comm.allreduce(J3D_loc_x, op=MPI.SUM)
    J3D_glob_y = comm.allreduce(J3D_loc_y, op=MPI.SUM)
    J3D_glob_z = comm.allreduce(J3D_loc_z, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x, J3D_glob_y, J3D_glob_z))
        pp.write_to_J_output_file(outputfile_J_path,t,J3D_glob_x,J3D_glob_y,J3D_glob_z)
        
    
    
    # DGTe = ufl.FiniteElement('DG', domain.ufl_cell(), 0)
    # DGT = dlfx.fem.FunctionSpace(domain, DGTe) 
    # esh_exp = dlfx.fem.Expression(eshelby, DGT.element.interpolation_points())
    # esh_fun = dlfx.fem.Function(DGT,name="Esh")
    # esh_fun.interpolate(esh_exp)
    
    grad_v = ufl.grad(dv)
    Gvec = dlfx.fem.assemble_vector(dlfx.fem.form( (eshelby[0,0]*grad_v[0] + eshelby[0,1]*grad_v[1] + eshelby[0,2]*grad_v[2]) * ufl.dx))
    Gvec.scatter_reverse(InsertMode.add)
    
    Gw = dlfx.fem.Function(F)
    Gw.x.array[:] = Gvec.array
    Gw.name = 'Gel'
    
    # Gvec1 = dlfx.fem.assemble_scalar(dlfx.fem.form( (eshelby[0,0]*grad_v[0] + eshelby[0,1]*grad_v[1] + eshelby[0,2]*grad_v[2])*ufl.dx))
    
    Gvec2 = dlfx.fem.assemble_vector(dlfx.fem.form( ufl.inner(eshelby, ufl.grad(du))*ufl.dx ))
    Gvec2.scatter_reverse(InsertMode.add)
    Gw2 = dlfx.fem.Function(W)
    Gw2.sub(0).x.array[:] = Gvec2.array
    Gel = Gw2.sub(0).collapse()
    Gel.name = 'Gel2'
        
    G1_loc = np.sum(Gw.x.array)
    G2_loc = np.sum(Gel.sub(0).x.array)
    
    comm.Barrier()
    G1_glob = comm.allreduce(G1_loc, op=MPI.SUM)
    G2_glob = comm.allreduce(G2_loc, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(G1_glob)
        print(G2_glob)
    
    # pp.write_scalar_fields(domain,comm,[Gw], ["Gel"], outputfile_xdmf_path, t)
    pp.write_field(domain,outputfile_xdmf_path,Gw,t,comm=comm)
    
    pp.write_vector_field(domain,outputfile_xdmf_path,Gel,t,comm=comm)
    # with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
    
    # # xdmf_out = dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a')
    #     xdmf_out.write_function(Gw)
    # xdmf_out.close()
    
    # J3D_loc_x_i, J3D_loc_y_i, J3D_loc_z_i = alex.linearelastic.get_J_3D_volume_integral_tf(eshelby, dv ,ufl.dx) # dxx(1)
    
    # comm.Barrier()
    # J3D_glob_x_i = comm.allreduce(J3D_loc_x_i, op=MPI.SUM)
    # J3D_glob_y_i = comm.allreduce(J3D_loc_y_i, op=MPI.SUM)
    # J3D_glob_z_i = comm.allreduce(J3D_loc_z_i, op=MPI.SUM)
    # comm.Barrier()
    
    # if rank == 0:
    #     print(pp.getJString(J3D_glob_x_i, J3D_glob_y_i, J3D_glob_z_i))
    #     pp.write_to_J_output_file(outputfile_J_path,t, J3D_glob_x, J3D_glob_y, J3D_glob_z)
        
    J3D_loc_x_ii, J3D_loc_y_ii, J3D_loc_z_ii = alex.linearelastic.get_J_3D_volume_integral(eshelby, ufl.dx)
    
    comm.Barrier()
    J3D_glob_x_ii = comm.allreduce(J3D_loc_x_ii, op=MPI.SUM)
    J3D_glob_y_ii = comm.allreduce(J3D_loc_y_ii, op=MPI.SUM)
    J3D_glob_z_ii = comm.allreduce(J3D_loc_z_ii, op=MPI.SUM)
    comm.Barrier()
    
    if rank == 0:
        print(pp.getJString(J3D_glob_x_ii, J3D_glob_y_ii, J3D_glob_z_ii))

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
        pp.print_J_plot(outputfile_J_path,script_path)
        
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


