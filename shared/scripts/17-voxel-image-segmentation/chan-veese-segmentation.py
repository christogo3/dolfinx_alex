# TODO does not work in parallel
import os
import alex.os
import numpy as np
import alex.postprocessing
import dolfinx as dlfx
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt

import alex.imageprocessing as img
import alex.solution as sol


# set up MPI parallel processing
comm = MPI.COMM_WORLD

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(alex.os.scratch_directory,script_name_without_extension)


scan_intensity_as_array = np.load(os.path.join(script_path, "voxel_data.npy"))
# scan_intensity_as_array = scan_intensity_as_array[:512,:512, : 512]


# This is where the fem field is written back to voxel to test if original voxel file is recaptured
fem2voxel_file_path = os.path.join(script_path,"output.dat")

# The voxel data as a 3D array
voxel_data_3d_array = scan_intensity_as_array
voxel_number_along_each_dimension = len(voxel_data_3d_array[0])

# creating fem mesh
domain : dlfx.mesh.Mesh = dlfx.mesh.create_unit_cube(comm,voxel_number_along_each_dimension,
                                                     voxel_number_along_each_dimension,
                                                     voxel_number_along_each_dimension,
                                                     cell_type=dlfx.mesh.CellType.hexahedron)

# creating fem functions
U = dlfx.fem.functionspace(domain,("CG",1))
u_h = dlfx.fem.Function(U)
u_restart = dlfx.fem.Function(U)
u_n = dlfx.fem.Function(U)
u_n.x.array[:] = np.full_like(u_n.x.array,0.0,dtype=dlfx.default_scalar_type)
du = ufl.TestFunction(U)
ddu = ufl.TrialFunction(U)

# defining parameters, and functions for chan veese
def W(xi):
    return ufl.as_ufl(0.25*xi**2*(1.0-xi)**2)

# reading the voxel data to fem in two ways
I : dlfx.fem.Function = img.set_cell_field_from_voxel_data(voxel_data_3d_array, domain) # cell wise constant field
# cell_tags = img.voxel_data_as_cell_tags(voxel_data_3d_array, domain) # as cell tags

epsilon = dlfx.fem.Constant(domain, 0.05)
lam = dlfx.fem.Constant(domain, 0.5)
alpha = dlfx.fem.Constant(domain, 0.25)
c1 = dlfx.fem.Constant(domain, 2500.0) # peaks from histogram
c2 = dlfx.fem.Constant(domain, 10000.0)

dt = dlfx.fem.Constant(domain, 1.0)
iter_cv = 5 # number iterations chan veese 
Tend = (2.0**iter_cv - 1.0) * dt.value


# preparing solution with newton
def before_first_timestep():
    alex.postprocessing.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    return
   

# for output of different parts
Q = dlfx.fem.functionspace(domain, ("DG", 0))
segmentation_as_cell_field = dlfx.fem.Function(Q)
segmentation_as_cell_field.name = "segmentation"

def after_last_timestep():
    # write segmented voxel data
    # if comm.Get_rank() == 0:
    #     img.write_field_to_voxel_data_leS(domain, fem2voxel_file_path, voxel_number_along_each_dimension, segmentation_as_cell_field)
    return
    

def after_timestep_success(t,dt,iters):
    c1.value = dlfx.fem.assemble_scalar(dlfx.fem.form((u_h*I)*ufl.dx)) / dlfx.fem.assemble_scalar(dlfx.fem.form(u_h*ufl.dx))
    c2.value = dlfx.fem.assemble_scalar(dlfx.fem.form(((1.0 -u_h) * I) * ufl.dx)) / dlfx.fem.assemble_scalar(dlfx.fem.form((1.0-u_h)*ufl.dx))
    img.clip(u_h, segmentation_as_cell_field, 0.5 ,1,0) 
    
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        xdmf_out.write_function(segmentation_as_cell_field,t)
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        xdmf_out.write_function(I,t)
# alex.postprocessing.write_scalar_fields(domain,comm,scalar_fields_as_functions=[phase],scalar_field_names=["phase"],outputfile_xdmf_path=outputfile_xdmf_path,t=0.0)
    alex.postprocessing.write_scalar_fields(domain,comm,[u_h],["u"],outputfile_xdmf_path,t)
   
    u_restart.x.array[:] = u_h.x.array[:]
    
def after_timestep_restart(t,dt,iters):
    u_h.x.array[:] = u_restart.x.array[:]

def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    # this defines the chan veese problem
    dWdxi = ufl.diff(W(u_n),u_n)
    Res = ( 1.0/delta_t * ufl.inner((u_h-u_n),du) + \
    epsilon ** 2 * ufl.inner(ufl.grad(u_h),ufl.grad(du)) + ufl.inner(dWdxi,du) + \
    + 2.0 * lam * ufl.inner(u_h * ((I - c1) ** 2 + (I - c2) ** 2), du ) - \
        2.0 * lam * ufl.inner((I - c2) ** 2, du) + \
        alpha * ufl.inner(ufl.div(ufl.grad(I)), du) ) * ufl.dx

    dResDu = ufl.derivative(Res,u_h,ddu)
    return [Res, dResDu]

def get_bcs(t):
    return []


sol.solve_with_newton_adaptive_time_stepping(domain,w=u_h, Tend=Tend,dt=dt,
                                             before_first_timestep_hook=before_first_timestep,
                                             after_last_timestep_hook=after_last_timestep,
                                             get_residuum_and_gateaux=get_residuum_and_gateaux,
                                             get_bcs=get_bcs,
                                             after_timestep_success_hook=after_timestep_success,
                                             after_timestep_restart_hook=after_timestep_restart,
                                             comm=comm,
                                             print_bool=True)


