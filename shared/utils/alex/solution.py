import sys
import glob
import os
import dolfinx as dlfx
from typing import Callable
from mpi4py import MPI

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import numpy as np

def print_time_and_dt(t: float, dt: float):
    print(' ')
    print('==================================================')
    print('Computing solution at time = {0:.4e}'.format(t))
    print('==================================================')
    print('Current time step dt = {0:.4e}'.format(dt))
    print('==================================================')
    print(' ')
    sys.stdout.flush()
    return True
    

def print_no_convergence(dt: float):
    print('-----------------------------')
    print('!!! NO CONVERGENCE => dt: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_increasing_dt(dt: float):
    print('-----------------------------')
    print('!!! Increasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_decreasing_dt(dt: float):
    print('-----------------------------')
    print('!!! Decreasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True
    
def print_timestep_overview(iters: int, converged: bool, restart_solution: bool):
    print('-----------------------------')
    print(' No. of iterations: ', iters)
    print(' Converged:         ', converged)
    print(' Restarting:        ', restart_solution)
    print('-----------------------------')
    sys.stdout.flush()
    return True

def print_runtime(runtime: float):
    print('') 
    print('-----------------------------')
    print('elapsed time:', runtime)
    print('-----------------------------')
    print('') 
    sys.stdout.flush()
    return True
        
def prepare_newton_logfile(logfile_path: str):
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')  
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()
    return True
    
def write_to_newton_logfile(logfile_path: str, t: float, dt: float, iters: int):
    logfile = open(logfile_path, 'a')
    logfile.write(str(t)+'  '+str(dt)+'  '+str(iters)+'\n')
    logfile.close()
    return True

def write_runtime_to_newton_logfile(logfile_path: str, runtime: float):
    logfile = open(logfile_path, 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  '+str(runtime)+'\n')
    logfile.write('# \n')
    logfile.close()
    return True
    
def solve_with_newton_adaptive_time_stepping(domain: dlfx.mesh.Mesh,
                                             w: dlfx.fem.Function, 
                                             Tend: float,
                                             dt: float,
                                             before_first_timestep_hook: Callable,
                                             after_last_timestep_hook: Callable,
                                             before_each_timestep_hook: Callable, 
                                             get_residuum_and_gateaux: Callable,
                                             get_bcs: Callable,
                                             after_timestep_success_hook: Callable,
                                             after_timestep_restart_hook: Callable,
                                             comm: MPI.Intercomm):
    rank = comm.Get_rank()
    
    # time stepping
    max_iters = 8
    min_iters = 4
    dt_scale_down = 0.5
    dt_scale_up = 2.0
    
    t = 0
    trestart = 0
    delta_t = dlfx.fem.Constant(domain, dt)

    before_first_timestep_hook()

    while t < Tend:
        delta_t.value = dt

        before_each_timestep_hook(t,dt)
            
        [Res, dResdw] = get_residuum_and_gateaux(delta_t)
        
        bcs = get_bcs(t)
        
        # define nonlinear problem and solver
        problem = NonlinearProblem(Res, w, bcs, dResdw)
        solver = NewtonSolver(comm, problem)
        solver.report = True
        solver.max_it = max_iters
        
        # control adaptive time adjustment
        restart_solution = False
        converged = False
        iters = max_iters + 1 # iters aalways needs to be defined
        try:
            (iters, converged) = solver.solve(w)
        except RuntimeError:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0:
                print_no_convergence(dt)
        
        if converged and iters < min_iters and t > np.finfo(float).eps:
            dt = dt_scale_up*dt
            if rank == 0:
                print_decreasing_dt(dt)
        if iters > max_iters:
            dt = dt_scale_down*dt
            restart_solution = True
            if rank == 0:
                print_decreasing_dt(dt)

        if rank == 0:    
            print_timestep_overview(iters, converged, restart_solution)

        if not(restart_solution):
            after_timestep_success_hook(t,dt,iters)
            trestart = t
            t = t+dt
        else:
            t = trestart+dt
            after_timestep_restart_hook(t,dt,iters)
    after_last_timestep_hook()