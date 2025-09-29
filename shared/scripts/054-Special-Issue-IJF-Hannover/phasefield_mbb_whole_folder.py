import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ufl
import basix.ufl
import basix
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc
from scipy.interpolate import griddata

import alex.homogenization
import alex.linearelastic as le
import alex.phasefield as pf
import alex.postprocessing as pp
import alex.util
import alex.os
import alex.boundaryconditions as bc
import alex.solution as sol
import json


# ---------------------------
# CLI INPUT HANDLING
# ---------------------------
DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), "resources", "250923_TTO_var_a_E_var_min_max","mbb_var_a_E_var")
VALID_CASES = {"vary", "min", "max", "all"}

def parse_args(argv):
    """
    Accepted forms:
      python script.py
      python script.py FOLDER
      python script.py FOLDER CASE
      python script.py FOLDER INDEX
      python script.py FOLDER START END
      python script.py FOLDER START END CASE
      python script.py FOLDER INDEX CASE
    CASE in {vary|min|max|all}
    """
    folder = DEFAULT_FOLDER
    ds_start = None
    ds_end = None
    case = None

    if len(argv) >= 2:
        folder = argv[1]
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Provided folder path does not exist: {folder}")

    # Collect remaining tokens and try to interpret ints vs case
    tokens = argv[2:]
    ints = []
    others = []
    for t in tokens:
        try:
            ints.append(int(t))
        except ValueError:
            others.append(t.lower())

    if len(ints) == 1:
        ds_start = ds_end = ints[0]
    elif len(ints) >= 2:
        ds_start, ds_end = ints[0], ints[1]

    if others:
        # last textual token wins
        last = others[-1]
        if last in VALID_CASES:
            case = last
        else:
            raise ValueError(f"Unknown case '{last}'. Valid: {sorted(VALID_CASES)}")

    return folder, ds_start, ds_end, case

folder_path, dataset_start, dataset_end, case_param = parse_args(sys.argv)
print(f"[INFO] Using folder: {folder_path}")




# ---------------------------
# AUTO-DETECT INTEGER SUFFIXES
# ---------------------------
available_files = os.listdir(folder_path)
pattern = re.compile(r"cell_data_(\d+)\.csv")
all_x_candidates = sorted([int(pattern.match(f).group(1)) for f in available_files if pattern.match(f)])

if not all_x_candidates:
    raise FileNotFoundError(f"No 'cell_data_x.csv' file found in {folder_path}")

# Filter if range is specified
if dataset_start is not None and dataset_end is not None:
    x_candidates = [x for x in all_x_candidates if dataset_start <= x <= dataset_end]
else:
    x_candidates = all_x_candidates

if not x_candidates:
    raise ValueError(f"No dataset indices found in the specified range: {dataset_start} to {dataset_end}")

print(f"[INFO] Detected dataset indices to process: {x_candidates}")

# ---------------------------
# MPI INITIALIZATION
# ---------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

convergence_log_path = os.path.join(folder_path, "convergence_log.txt")
if rank == 0:
    # Start fresh for each run
    with open(convergence_log_path, "w") as f:
        f.write("x_value,case,status\n")
        
def log_convergence_status(x_value, case, status):
    if rank == 0:
        with open(convergence_log_path, "a") as f:
            f.write(f"{x_value},{case},{status}\n")

# ---------------------------
# MAIN LOOP OVER SELECTED x_candidates
# ---------------------------
for x_value in x_candidates:
    print(f"[INFO] Processing dataset index: {x_value}")

    # ---------------------------
    # BUILD FILE PATHS
    # ---------------------------
    node_file = os.path.join(folder_path, f"node_coords_{x_value}.csv")
    point_data_file = os.path.join(folder_path, f"points_data_{x_value}.csv")
    cell_data_file = os.path.join(folder_path, f"cell_data_{x_value}.csv")
    connectivity_file = os.path.join(folder_path, f"connectivity_{x_value}.csv")
    mesh_file = os.path.join(folder_path, f"dlfx_mesh_{x_value}.xdmf")

    # case-agnostic figure (E-distribution); case-specific outputs will be below
    base_results_xdmf_path = os.path.join(folder_path, f"results_{x_value}.xdmf")  # kept for mesh write convenience
    base_output_graph_path = os.path.join(folder_path, f"result_graphs_{x_value}.txt")  # not used directly, but kept

    # ---------------------------
    # VALIDATE FILES
    # ---------------------------
    for fpath in [node_file, point_data_file, cell_data_file, connectivity_file, mesh_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    print(f"[INFO] All required files found for dataset {x_value}.")

    # ---------------------------
    # HELPER FUNCTIONS
    # ---------------------------
    def load_data(file_path):
        return pd.read_csv(file_path)

    def infer_mesh_dimensions_from_nodes(nodes_df):
        unique_y_coords = nodes_df['Points_1'].unique()
        unique_x_coords = nodes_df['Points_0'].unique()
        unique_y_coords.sort()
        unique_x_coords.sort()
        return len(unique_y_coords) - 1, len(unique_x_coords) - 1

    def arrange_cells_2D(connectivity_df, mesh_dims):
        cell_grid = np.zeros(mesh_dims, dtype=int)
        for index, row in connectivity_df.iterrows():
            cell_id = index #row['Cell ID']
            row_idx = index // mesh_dims[1]
            col_idx = index % mesh_dims[1]
            cell_grid[row_idx, col_idx] = cell_id
        return cell_grid

    def map_E_to_grid(cell_id_grid, cell_data_df):
        E_Grid = np.full(cell_id_grid.shape, np.nan)
        E_values = cell_data_df['E-Modul'].values
        for row in range(cell_id_grid.shape[0]):
            for col in range(cell_id_grid.shape[1]):
                cell_id = cell_id_grid[row, col]
                if cell_id < len(E_values):
                    E_Grid[row, col] = E_values[cell_id]
                else:
                    E_Grid[row, col] = np.nan
        return E_Grid

    def calculate_element_size(nodes_df):
        x1, y1 = nodes_df.iloc[0]['Points_0'], nodes_df.iloc[0]['Points_1']
        x2, y2 = nodes_df.iloc[1]['Points_0'], nodes_df.iloc[1]['Points_1']
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def interpolate_pixel_data(data, element_size, x_coords, y_coords, method='linear'):
        grid_x, grid_y = np.meshgrid(
            (np.arange(data.shape[1]) + 0.5) * element_size, 
            (np.arange(data.shape[0]) + 0.5) * element_size
        )
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = data.ravel()
        query_points = np.column_stack((x_coords, y_coords))
        interpolated_values = griddata(points, values, query_points, method=method)
        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            interpolated_values[nan_mask] = griddata(points, values, query_points[nan_mask], method='nearest')
        return interpolated_values

    def create_emodulus_interpolator(nodes_df, E_grid):
        return lambda x: interpolate_pixel_data(E_grid, calculate_element_size(nodes_df), x[0], x[1])

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    nodes_df = load_data(node_file)
    point_data_df = load_data(point_data_file)
    cell_data_df = load_data(cell_data_file)
    connectivity_df = load_data(connectivity_file)

    mesh_dims = infer_mesh_dimensions_from_nodes(nodes_df)
    cell_id_grid = arrange_cells_2D(connectivity_df, mesh_dims)
    E_grid = map_E_to_grid(cell_id_grid, cell_data_df)
    E_max, E_min = 210000, 130000.0 #np.min(E_grid) #np.max(E_grid)

    # Plot E distribution (once per dataset index)
    plt.figure(figsize=(10, 8))
    plt.imshow(E_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='E')
    plt.title(f'E Distribution for dataset {x_value}')
    plt.savefig(os.path.join(folder_path, f'E_distribution_{x_value}.png'), dpi=300)
    plt.close()

    # ---------------------------
    # MPI + MESH LOADING
    # ---------------------------
    with dlfx.io.XDMFFile(comm, mesh_file, 'r') as mesh_inp:
        domain = mesh_inp.read_mesh()
        
    # with dlfx.io.XDMFFile(comm, os.path.join("/home/scripts/052-Special-Issue-IJF-Hannover/resources/310125_var_bcpos_rho_10_120_004","dlfx_mesh_20.xdmf"), 'r') as mesh_inp:
    #     domain = mesh_inp.read_mesh()

    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
    if rank == 0:
        pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

    Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(2,))
    Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    W = dlfx.fem.FunctionSpace(domain, basix.ufl.mixed_element([Ve, Se]))
    S = dlfx.fem.FunctionSpace(domain, Se)

    # ---------------------------
    # CASE LOOP
    # ---------------------------
    available_cases = ["vary", "min", "max"]
    if case_param is None:
        cases_to_run = ["min", "max", "vary"]    
    elif case_param == "all":
        cases_to_run = available_cases
    else:
        if case_param not in available_cases:
            if rank == 0:
                print(f"[WARNING] Unknown case '{case_param}'. Falling back to all cases {available_cases}.")
            cases_to_run = available_cases
        else:
            cases_to_run = [case_param]

    for case in cases_to_run:
        print(f"[INFO] Running case '{case}' for dataset index {x_value}")

        # ---- Case-specific output paths to avoid overwrites
        results_xdmf_path = os.path.join(folder_path, f"results_{x_value}_{case}.xdmf")
        outputfile_graph_path = os.path.join(folder_path, f"result_graphs_{x_value}_{case}.txt")

        # ---- Material fields
        E = dlfx.fem.Function(S)
        nu = dlfx.fem.Constant(domain=domain, c=0.3)

        if case == "vary":
            E.interpolate(create_emodulus_interpolator(nodes_df, E_grid))
        elif case == "min":
            E.x.array[:] = np.full_like(E.x.array[:], E_min)
        elif case == "max":
            E.x.array[:] = np.full_like(E.x.array[:], E_max)

        lam = le.get_lambda(E, nu)
        mue = le.get_mu(E, nu)
        dim = domain.topology.dim
        alex.os.mpi_print('spatial dimensions: ' + str(dim), rank)

        # ---- Boundary dofs (top boundary, u_y)
        fdim = domain.topology.dim - 1
        atol = 1e-12
        
        atol_bc = (x_max_all - x_min_all) * 0.000
            
        width_applied_load = 0.2
        increment_a = 0.5
        facets_at_boundary = dlfx.mesh.locate_entities_boundary(
            domain, fdim, bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc) 
        )
        dofs_at_boundary_y = dlfx.fem.locate_dofs_topological(W.sub(0).sub(1), fdim, facets_at_boundary)

        # ---- Simulation parameters
        dt_start = 0.001
        dt_global = dlfx.fem.Constant(domain, dt_start)
        t_global = dlfx.fem.Constant(domain, 0.0)
        trestart_global = dlfx.fem.Constant(domain, 0.0)
        Tend = 100.0 * dt_global.value
        gc = dlfx.fem.Constant(domain, 1.0)
        eta = dlfx.fem.Constant(domain, 0.00001)
        epsilon = dlfx.fem.Constant(domain, 0.05)
        Mob = dlfx.fem.Constant(domain, 1000.0)
        iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)

        # ---- Solution fields
        w = dlfx.fem.Function(W)
        u, s = w.split()
        wrestart = dlfx.fem.Function(W)
        wm1 = dlfx.fem.Function(W)
        um1, sm1 = ufl.split(wm1)
        dw = ufl.TestFunction(W)
        ddw = ufl.TrialFunction(W)

        phaseFieldProblem = pf.StaticPhaseFieldProblem2D_split(
            degradationFunction=pf.degrad_quadratic,
            psisurf=pf.psisurf_from_function,
            split= "volumetric"#"volumetric"
        )

        timer = dlfx.common.Timer()

        # ---- Logs
        script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
        logfile_path = alex.os.logfile_full_path(folder_path, f"{script_name_without_extension}_{x_value}_{case}")

        # ---- Hooks
        def before_first_time_step():
            timer.start()
            wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
            wrestart.x.array[:] = wm1.x.array[:]
            if rank == 0:
                pp.prepare_graphs_output_file(outputfile_graph_path)
            # write mesh container once so XDMF exists
            pp.write_meshoutputfile(domain, results_xdmf_path, comm)

        def before_each_time_step(t, dt):
            if rank == 0:
                sol.print_time_and_dt(t, dt)

        def get_residuum_and_gateaux(delta_t):
            return phaseFieldProblem.prep_newton(
                w=w, wm1=wm1, dw=dw, ddw=ddw, lam=lam, mu=mue,
                Gc=gc, epsilon=epsilon, eta=eta, iMob=iMob, delta_t=delta_t
            )

        n = ufl.FacetNormal(domain)
        external_surface_tag = 5
        external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
        ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)
        
        
        
        top_surface_tag = 9
        top_surface_tags = pp.tag_part_of_boundary(
            domain, bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), top_surface_tag
        )
        ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags)

        success_timestep_counter = dlfx.fem.Constant(domain, 0.0)
        postprocessing_interval = dlfx.fem.Constant(domain, 300.0)

        def get_bcs(t):
            atol_bc = (x_max_all - x_min_all) * 0.000
            
            width_applied_load = 0.2
            increment_a = 0.5
            
            
            
            
            bcs = [
                bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                                                   bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                   bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
                                                  bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                  bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), W, 0),
                # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                #                                   bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                #                                   bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0)
                
                # bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                #                                   bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                # bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
                #                                   bc.get_bottom_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                  bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                #                                   bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0)
            ]
            if abs(t) > sys.float_info.epsilon * 5:
                bcs.append(pf.irreversibility_bc(domain, W, wm1))
            return bcs

        Work = dlfx.fem.Constant(domain,0.0)
        
       
        dx = ufl.Measure("dx", domain=domain)
        vol = alex.homogenization.get_filled_vol(dx=dx,comm=comm)
        
        
        def after_timestep_success(t, dt, iters):
            sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
            # Reaction force at top boundary
            Rx_top, Ry_top = pp.reaction_force(sigma, n=n, ds=ds_top_tagged(top_surface_tag), comm=comm)

            # Get vertical displacement u_y at top boundary dofs
            if len(w.x.array[dofs_at_boundary_y]) > 0:
                u_y_top_local = w.x.array[dofs_at_boundary_y][0]
            else:
                u_y_top_local = 1e10

            comm.barrier()
            u_y_top = comm.allreduce(u_y_top_local, MPI.MIN)
            comm.barrier()

            
            # dW = pp.work_increment_external_forces(sigma,u,um1,n,ds_top_tagged(top_surface_tag),comm=comm)
            dW = pp.work_increment_external_forces(sigma,u,um1,n,ds,comm=comm)
            Work.value = Work.value + dW
    
            A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
            E_el = phaseFieldProblem.get_E_el_global(s,eta,u,lam,mue,dx=ufl.dx,comm=comm)
            
            # E_s = phaseFieldProblem.getGlobalFractureSurface(s,gc,epsilon,dx=ufl.dx)
    
            if rank == 0:
                pp.write_to_graphs_output_file(outputfile_graph_path, t, u_y_top, Ry_top, dW, Work.value, A, E_el)

            if rank == 0:
                sol.write_to_newton_logfile(logfile_path, t, dt, iters)
                
            wm1.x.array[:] = w.x.array[:]
            wrestart.x.array[:] = w.x.array[:]
                

            success_timestep_counter.value = success_timestep_counter.value + 1.0
            if int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0:
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t)
                pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t)

        def after_timestep_restart(t, dt, iters):
            # If global dt has shrunk beyond tolerance -> write what we have and skip this case
            if dt_global.value < 10.0 ** (-14):
                sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t)
                pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t)
                if rank == 0:
                    print(f"[WARNING] NO CONVERGENCE (dt too small) in case '{case}' for dataset {x_value}. Skipping to next case.")
                # Signal to outer try/except to continue with next case
                raise RuntimeError("ConvergenceFailure")
            # Otherwise: restore previous state and let the solver retry with smaller dt
            w.x.array[:] = wrestart.x.array[:]

        def after_last_timestep():
            timer.stop()
            if rank == 0:
                runtime = timer.elapsed()
                sol.print_runtime(runtime)
                sol.write_runtime_to_newton_logfile(logfile_path, runtime)
                pp.print_graphs_plot(outputfile_graph_path, print_path=folder_path, legend_labels=["u_y_top", "R_y_top", "dW", "W","A", "E_el"])

            
                vol_path = os.path.join(folder_path, f"vol_{x_value}_{case}.json")
                volumes_data = {
                    "vol": vol,
                }
                with open(vol_path, "w") as f:
                    json.dump(volumes_data, f, indent=4)
                print(f"Saved volume info to: {vol_path}")
        
            sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
            pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t_global.value, comm)
            E.name = "E"
            pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t_global.value)
            pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t_global.value)

        try:
            sol.solve_with_newton_adaptive_time_stepping(
                domain,
                w,
                Tend,
                dt_global,
                before_first_timestep_hook=before_first_time_step,
                after_last_timestep_hook=after_last_timestep,
                before_each_timestep_hook=before_each_time_step,
                get_residuum_and_gateaux=get_residuum_and_gateaux,
                get_bcs=get_bcs,
                after_timestep_restart_hook=after_timestep_restart,
                after_timestep_success_hook=after_timestep_success,
                comm=comm,
                print_bool=True,
                t=t_global,
                trestart=trestart_global,
            )
            log_convergence_status(x_value, case, "OK")
        except RuntimeError as e:
            if "ConvergenceFailure" in str(e):
                log_convergence_status(x_value, case, f"ConvergenceFailure at time {t_global.value}")
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t_global.value+0.0001, comm)
                continue  # skip to next case
            else:
                log_convergence_status(x_value, case, f"RuntimeError: {str(e)}")
                raise



