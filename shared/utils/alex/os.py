import os
import sys

scripts_directory = os.path.join('/home','scripts')
resources_directory = os.path.join('/home','resources')

def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_log.txt")

def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".xdmf")

def mpi_print(output, rank=0):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return