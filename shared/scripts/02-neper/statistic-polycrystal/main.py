import dolfinx as dlfx
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
import alex.linearelastic

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path,script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path,script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path,script_name_without_extension)


def mpi_print(output):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return
# set FEniCSX log level
# dlfx.log.set_log_level(dlfx.log.LogLevel.INFO)
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

with dlfx.io.XDMFFile(comm, os.path.join(alex.os.resources_directory,'polycrystal_cube.xdmf'), 'r') as mesh_inp: 
    domain = mesh_inp.read_mesh(name="Grid")

Tend = 5.0
dt = 0.0001

# function space using mesh and degree
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) # displacements
Te = ufl.FiniteElement("Lagrange", domain.ufl_cell(),2) # fracture fields
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))

# get dimension and bounds for each mpi process
dim = domain.topology.dim
x_min = np.min(domain.geometry.x[:,0]) 
x_max = np.max(domain.geometry.x[:,0])   
y_min = np.min(domain.geometry.x[:,1]) 
y_max = np.max(domain.geometry.x[:,1])   
z_min = np.min(domain.geometry.x[:,2]) 
z_max = np.max(domain.geometry.x[:,2])

# find global min/max over all mpi processes
comm.Barrier()
x_min_all = comm.allreduce(x_min, op=MPI.MIN)
x_max_all = comm.allreduce(x_max, op=MPI.MAX)
y_min_all = comm.allreduce(y_min, op=MPI.MIN)
y_max_all = comm.allreduce(y_max, op=MPI.MAX)
z_min_all = comm.allreduce(z_min, op=MPI.MIN)
z_max_all = comm.allreduce(z_max, op=MPI.MAX)
comm.Barrier()

mpi_print('spatial dimensions: '+str(dim))
mpi_print('x_min, x_max: '+str(x_min_all)+', '+str(x_max_all))
mpi_print('y_min, y_max: '+str(y_min_all)+', '+str(y_max_all))
mpi_print('z_min, z_max: '+str(z_min_all)+', '+str(z_max_all))