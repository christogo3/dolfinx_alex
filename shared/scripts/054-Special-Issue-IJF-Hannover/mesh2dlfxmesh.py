import meshio
import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
import sys
import re
import copy
import pandas as pd

def scale_points_to_target_domain(points,
                                  x_min_target, x_max_target,
                                  y_min_target, y_max_target):
    # Find current bounds of points
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # Compute scale factors for each axis
    scale_x = (x_max_target - x_min_target) / (x_max - x_min)
    scale_y = (y_max_target - y_min_target) / (y_max - y_min)

    # Scale points
    points_scaled = np.empty_like(points)
    points_scaled[:, 0] = x_min_target + (points[:, 0] - x_min) * scale_x
    points_scaled[:, 1] = y_min_target + (points[:, 1] - y_min) * scale_y

    return points_scaled


def read_active_cells_mapping(mapping_file_path):
    """
    Reads the active_cells_mapping file.
    Expected format:
        # cell_data_X active_cells_to_be_meshed
        19 1,2,3
        20 4
        ...
    Returns a dictionary {x_value: [active_cell_ids]}.
    """
    mapping = {}
    with open(mapping_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid mapping line: {line}")
            x_val = int(parts[0])
            # Split by comma, strip spaces, convert to int
            active_ids = [int(v.strip()) for v in parts[1].split(",") if v.strip()]
            mapping[x_val] = active_ids
    return mapping

def read_target_domain_from_csv(folder, x_value):
    csv_path = os.path.join(folder, f"node_coords_{x_value}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Node coords file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {"Points_0", "Points_1"}.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} missing required columns Points_0, Points_1")
    x_min, x_max = df["Points_0"].min(), df["Points_0"].max()
    y_min, y_max = df["Points_1"].min(), df["Points_1"].max()
    return x_min, x_max, y_min, y_max

def process_mesh_file(comm, mesh_path, output_folder, x_value, active_ids,
                      x_min_target=0, x_max_target=2.0,
                      y_min_target=0, y_max_target=1.0):
    print(f"[INFO] Processing mesh file: {mesh_path}")
    data = meshio.read(mesh_path)
    points_tmp = data.points[:, 0:2]
    points = copy.deepcopy(points_tmp)
    # Swap x and y as in original code
    points[:, 0] = points_tmp[:, 1]
    points[:, 1] = points_tmp[:, 0]
    
    
    # Read scaling domain from node_coords CSV
    folder = os.path.dirname(mesh_path)
    x_min_target, x_max_target, y_min_target, y_max_target = read_target_domain_from_csv(folder, x_value)

    # Override target bounds
    # x_min_target, x_max_target = 0, 10.0
    # y_min_target, y_max_target = 0, 1.0

    points = scale_points_to_target_domain(points,
                                           x_min_target, x_max_target,
                                           y_min_target, y_max_target)

    cells = data.cells_dict['triangle']
    cells_id = data.cell_data_dict['physical']['triangle']

    # Select cells where cell ID is in active_ids list
    active_cells = [cell for idx, cell in enumerate(cells) if cells_id[idx] in active_ids]

    if not active_cells:
        print(f"[WARNING] No cells found with IDs {active_ids} for x_value={x_value}")

    cell = ufl.Cell('triangle', geometric_dimension=2)
    element = ufl.VectorElement('Lagrange', cell, 1, dim=2)
    mesh = ufl.Mesh(element)
    domain = dlfx.mesh.create_mesh(comm, active_cells, points, mesh)

    output_path = os.path.join(output_folder, f'dlfx_mesh_{x_value}.xdmf')
    print(f"[INFO] Writing scaled mesh to: {output_path}")
    with dlfx.io.XDMFFile(comm, output_path, 'w') as xdmf:
        xdmf.write_mesh(domain)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 1:
        if rank == 0:
            print("[ERROR] This script requires to be run with a single MPI process.")
        sys.exit(1)

    script_path = os.path.dirname(__file__)
    default_folder = os.path.join(script_path, "resources", "250925_TTO_mbb_festlager_var_a_E_var_min_max","mbb_festlager_var_a_E_var")

    # Arguments:
    #   sys.argv[1] -> optional folder path
    #   sys.argv[2] -> optional mesh number to process
    folder_path = default_folder
    mesh_number_to_process = None

    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            folder_path = sys.argv[1]
            if len(sys.argv) > 2:
                try:
                    mesh_number_to_process = int(sys.argv[2])
                except ValueError:
                    raise ValueError(f"Invalid mesh number: {sys.argv[2]}")
        else:
            # If first arg is not a folder, treat it as the mesh number
            try:
                mesh_number_to_process = int(sys.argv[1])
            except ValueError:
                raise NotADirectoryError(f"Provided folder path does not exist: {sys.argv[1]}")

    print(f"[INFO] Using folder: {folder_path}")
    if mesh_number_to_process is not None:
        print(f"[INFO] Only processing mesh number: {mesh_number_to_process}")

    # Read active_cells_mapping
    mapping_file_path = os.path.join(folder_path, "active_cells_mapping")
    if not os.path.isfile(mapping_file_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")

    active_cells_mapping = read_active_cells_mapping(mapping_file_path)

    # Find mesh files
    pattern = re.compile(r"mesh_(\d+)\.xdmf$")
    mesh_files = []
    for filename in os.listdir(folder_path):
        m = pattern.match(filename)
        if m:
            x_val = int(m.group(1))
            mesh_files.append((x_val, filename))
    mesh_files.sort(key=lambda t: t[0])

    if not mesh_files:
        print("[WARNING] No mesh_x.xdmf files found in folder.")
        return

    output_folder = folder_path

    for x_val, fname in mesh_files:
        if mesh_number_to_process is not None and x_val != mesh_number_to_process:
            continue
        if x_val not in active_cells_mapping:
            print(f"[WARNING] No mapping entry for x_value={x_val}, skipping.")
            continue
        active_ids = active_cells_mapping[x_val]
        full_path = os.path.join(folder_path, fname)
        process_mesh_file(comm, full_path, output_folder, x_val, active_ids, x_max_target=10.0)


if __name__ == "__main__":
    main()



