import numpy as np
import foam_linear_elastic_func as sim
from mpi4py import MPI
import sys
import os
import itertools

import alex.parameterstudy as ps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]





def simulation_wrapper(scal_param):
    return sim.run_simulation(eps_mac_param=tensor_param, scal=scal_param, comm=comm)



desired_simulation_result = 0.1

# scal_at_failure = bisection_method(simulation_wrapper, desired_simulation_result, 0.001, 1.0, tol=0.02 * desired_simulation_result)

if rank == 0:
    # Remove the file if it exists
    file_path = os.path.join(script_path, 'failure_surface.csv')
    if os.path.exists(file_path):
        os.remove(file_path)

# Define arrays
n_values = 3
values = np.linspace(-1.0, 1.0, n_values)

# Initialize arrays
sxx = values
sxy = values
sxz = values
syz = values
szz = values
syy = values

# Define the ranges for each list (start and end indices)
range_sxx = (0, len(sxx))  # Replace with actual start and end indices
range_sxy = (0, len(sxy))  # Replace with actual start and end indices
range_sxz = (0, len(sxz))  # Replace with actual start and end indices
range_syz = (0, len(syz))  # Replace with actual start and end indices
range_szz = (0, len(szz))  # Replace with actual start and end indices
range_syy = (0, len(syy))  # Replace with actual start and end indices

# Slice the lists according to the specified ranges
sxx_range = sxx[range_sxx[0]:range_sxx[1]]
sxy_range = sxy[range_sxy[0]:range_sxy[1]]
sxz_range = sxz[range_sxz[0]:range_sxz[1]]
syz_range = syz[range_syz[0]:range_syz[1]]
szz_range = szz[range_szz[0]:range_szz[1]]
syy_range = syy[range_syy[0]:range_syy[1]]

total_iterations = len(sxx_range) * len(sxy_range) * len(sxz_range) * len(syz_range) * len(szz_range) * len(syy_range)
computation = 1

# Combine the sliced lists into one list of lists
input_lists = [sxx_range, sxy_range, sxz_range, syz_range, szz_range, syy_range]

# Generate all combinations using itertools.product
all_combinations = list(itertools.product(*input_lists))

# Split combinations among MPI ranks
# chunk_size = len(all_combinations) // size
# remainder = len(all_combinations) % size

# start_idx = rank * chunk_size + min(rank, remainder)
# end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

# local_combinations = all_combinations[start_idx:end_idx]

for vals in all_combinations:
    val_sxx, val_sxy, val_sxz, val_syz, val_szz, val_syy = vals

    tensor_param = np.array([[val_sxx, val_sxy, val_sxz],
                             [val_sxy, val_syy, val_syz],
                             [val_sxz, val_syz, val_szz]])
    if np.linalg.norm(tensor_param) < 1.0e-3:  # if all entries are zero then no ev
        continue

    try:
        comm.barrier()
        scal_at_failure = ps.bisection_method(simulation_wrapper, desired_simulation_result, 0.001, 1.0, tol=0.05 * desired_simulation_result, comm=comm)
        
        principal_tensor_values_at_failure = np.linalg.eigvals(tensor_param * scal_at_failure)  # does not make sense to do this in main stresses since not isotropic?
        tensor_critical_value_hom_material = 1.0

        if rank == 0:
            with open(os.path.join(script_path, 'failure_surface.csv'), 'a') as file:
                if np.linalg.norm(principal_tensor_values_at_failure) < 5.0 * tensor_critical_value_hom_material:
                    file.write(','.join(map(str, principal_tensor_values_at_failure)) + '\n')
            print("Running computation {} of {} total".format(computation, total_iterations))
            sys.stdout.flush()

        computation += 1
        comm.barrier()
    except Exception as e:
        if rank == 0:
            print("An error occurred in computation " + str(computation) + " message:", e)
            print("tensor value is: \n")
            print(tensor_param)
            sys.stdout.flush()
        computation += 1

# Optional: gather results at root process (rank 0) if needed




