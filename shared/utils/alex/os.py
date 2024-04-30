import os
import sys
import shutil
from datetime import datetime

scripts_directory = os.path.join('/home','scripts')
resources_directory = os.path.join('/home','resources')

def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_log.txt")

def outputfile_J_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_J.txt")

def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".xdmf")

def mpi_print(output, rank=0):
    if rank == 0:
        print(output)
        sys.stdout.flush
    return


def create_timestamp_string():
    now = datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_string

def create_results_folder(source_folder):
    timestamp = create_timestamp_string()
    results_folder = os.path.join(source_folder, timestamp + "_results")
    os.makedirs(results_folder)
    return results_folder

def copy_contents_to_results_folder(source_folder, results_folder):
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        if os.path.isfile(source_item):
            shutil.copy(source_item, results_folder)
        elif os.path.isdir(source_item) and not item.endswith("_results"):
            shutil.copytree(source_item, os.path.join(results_folder, item))
