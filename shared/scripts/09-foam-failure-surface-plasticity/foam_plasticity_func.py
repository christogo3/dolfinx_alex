import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
from mpi4py import MPI


import ufl 
import numpy as np
import os 
import sys
import math

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
import alex.plasticity

def run_simulation(scal,eps_mac_param, comm: MPI.Intercomm):

    script_path = os.path.dirname(__file__)
    script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
    logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
    outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
    outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)
    # outputfile_vtk_path = alex.os.outputfile_vtk_full_path(script_path,script_name_without_extension)

    # set FEniCSX log level
    # dlfx.log.set_log_level(log.LogLevel.INFO)
    # dlfx.log.set_output_file('xxx.log')

    # set and start stopwatch
    timer = dlfx.common.Timer()
    timer.start()

    # set MPI environment
    # comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # size = comm.Get_size()
    # print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
    # sys.stdout.flush()


    # generate domain
    #domain = dlfx.mesh.create_unit_square(comm, N, N, cell_type=dlfx.mesh.CellType.quadrilateral)
    # domain = dlfx.mesh.create_unit_cube(comm,N,N,N,cell_type=dlfx.mesh.CellType.hexahedron)

    with dlfx.io.XDMFFile(comm, os.path.join(script_path,'msh2xdmf.xdmf'), 'r') as mesh_inp: 
        domain = mesh_inp.read_mesh()
        
    gdim = domain.geometry.dim
        
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain)
    loading = dlfx.fem.Constant(domain, 0.0)
    
    
    E = dlfx.fem.Constant(domain, 70e3)  # in MPa
    nu = dlfx.fem.Constant(domain, 0.3)
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2.0 / (1 + nu)
    sig0 = dlfx.fem.Constant(domain, 250.0)  # yield strength in MPa
    Et = E / 100.0  # tangent modulus
    H = E * Et / (E - Et)  # hardening modulus
    

    dt = 0.2
    Tend = 1.0

    # function space using mesh and degree
    Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
    V = dlfx.fem.FunctionSpace(domain, Ve)
                
    u = dlfx.fem.Function(V, name="Total_displacement")

    du = dlfx.fem.Function(V, name="Iteration_correction")
    Du = dlfx.fem.Function(V, name="Current_increment")
    DuRestart =  dlfx.fem.Function(V, name="Current_increment_restart")
    v = ufl.TrialFunction(V)
    u_tf = ufl.TestFunction(V)
    
    
    deg_quad = 2  # quadrature degree for internal state variable representation
    sig_np1, sig_n, N_np1, beta, alpha_n, dGamma = alex.plasticity.define_internal_state_variables(gdim, domain, deg_quad,quad_scheme="default")
    dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
    quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)

    eps = alex.plasticity.eps_as_3D_tensor_function(gdim)
    as_3D_tensor = alex.plasticity.from_history_field_to_3D_tensor_mapper(gdim)
    to_vect = alex.plasticity.to_history_field_vector_mapper(gdim)


    def get_residuum_and_tangent(dt):
        return alex.plasticity.get_residual_and_tangent(n, loading, as_3D_tensor(sig_np1), u_tf, v, eps, ds, dx, lmbda,mu,as_3D_tensor(N_np1),beta,H)


    def before_each_time_step(t,dt):
        return
        Du.x.array[:] = 0 # TODO change maybe for better convergence
        

    


    def get_bcs(t):
        x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = bc.get_dimensions(domain,comm)
        
        eps_mac = dlfx.fem.Constant(domain, eps_mac_param * scal*t)
        # def left(x):
        #     return np.isclose(x[0], x_min_all,atol=0.01)
        
        # leftfacets = dlfx.mesh.locate_entities_boundary(domain, fdim, left)
        # leftdofs_x = dlfx.fem.locate_dofs_topological(V.sub(0), fdim, leftfacets)
        # bcleft_x = dlfx.fem.dirichletbc(1.0, leftdofs_x, V.sub(0))
        
        # v_crack = (x_max_all-x_min_all)/Tend
        # xtip = np.array([crack_tip_start_location_x + v_crack * t, crack_tip_start_location_y])
        # xtip = np.array([ v_crack * t, (y_max_all+y_min_all)/2.0],dtype=dlfx.default_scalar_type)
        # xK1 = dlfx.fem.Constant(domain, xtip)

        # bcs = bc.get_total_surfing_boundary_condition_at_box(domain=domain,comm=comm,functionSpace=V,subspace_idx=-1,K1=K1,xK1=xK1,lam=lam,mu=mu,epsilon=0.0, atol=0.01)
        
        
        bcs = bc.get_total_linear_displacement_boundary_condition_at_box_for_incremental_formulation(
            domain=domain, w_n=u, functionSpace=V, comm=comm,eps_mac=eps_mac,subspace_idx=-1,atol=0.01)
        # bcs = [bcleft_x]
        return bcs
    
    def before_first_time_step():
        DuRestart.x.array[:] = np.zeros_like(DuRestart.x.array[:]) 
    
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

    
    simulation_result = np.array([0.0])    
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
        
        sig_vm = le.sigvM(as_3D_tensor(sig_n))
        simulation_result[0] = pp.percentage_of_volume_above(domain,sig_vm,0.9*sig0,comm,ufl.dx)
        
       
                
        
        
    def after_timestep_restart(t,dt,iters):
        Du.x.array[:] = DuRestart.x.array[:]
        return
        
        
    def after_last_timestep():
        # stopwatch stop
        timer.stop()

        # report runtime to screen
        if rank == 0:
            runtime = timer.elapsed()
            # sol.print_runtime(runtime)
            # sol.write_runtime_to_newton_logfile(logfile_path,runtime)
            # pp.print_graphs_plot(outputfile_graph_path,script_path,legend_labels=["Jx_surf", "Ge_x_div", "Jx_nodal_forces", "Gad_x", "Gdis_x"])
            
            # cleanup only necessary on cluster
            # results_folder_path = alex.os.create_results_folder(script_path)
            # alex.os.copy_contents_to_results_folder(script_path,results_folder_path)
    
    
    Nitermax, tol = 200, 1e-6  # parameters of the Newton-Raphson procedure
    Nincr = 10
    load_steps = np.linspace(0, 1.0, Nincr + 1)[1:] ** 2.0
       
    # sol.solve_with_newton(domain, Du, du, Nitermax, tol, load_steps,  
    #                               before_first_timestep_hook=before_first_time_step, 
    #                               after_last_timestep_hook=after_last_timestep, 
    #                               before_each_timestep_hook=before_each_time_step, 
    #                               after_iteration_hook=after_iteration, 
    #                               get_residuum_and_tangent=get_residuum_and_tangent, 
    #                               get_bcs=get_bcs, 
    #                               after_timestep_success_hook=after_time_step_success,
    #                               comm=comm,
    #                               print_bool=True)


    sol.solve_with_custom_newton_adaptive_time_stepping(domain=domain,
                                                    sol=Du,
                                                    dsol=du,
                                                    Tend=Tend,
                                                    dt=dt,
                                                    before_first_timestep_hook=before_first_time_step,
                                                    after_last_timestep_hook=after_last_timestep,
                                                    after_timestep_restart_hook=after_timestep_restart,
                                                    after_iteration_hook=after_iteration,
                                                    get_residuum_and_gateaux=get_residuum_and_tangent,
                                                    get_bcs=get_bcs,
                                                    after_timestep_success_hook=after_time_step_success,
                                                    comm=comm,
                                                    print_bool=True
                                                    )
    return comm.allreduce(simulation_result,MPI.MAX)

