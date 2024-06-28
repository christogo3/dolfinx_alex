import pfmfrac_function_test as sim
from mpi4py import MPI
comm = MPI.COMM_WORLD


# Run the simulation
sim.run_simulation("coarse_pores",
                   10.0,
                   10.0,
                   0.5,
                   25.0,
                   1,
                   comm)