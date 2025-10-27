import os
import glob
import re
import argparse
import json
import numpy as np
import alex.evaluation as ev  # plotting utilities


def extract_number(filename):
    match = re.search(r"result_graphs_(\d+)_", filename)
    if not match:
        match = re.search(r"result_graphs_(\d+)_fromfile", filename)
    return int(match.group(1)) if match else float('inf')


def collect_data(files, stride=1, min_index=0):
    data_to_plot = []
    legend_entries = []
    file_indices = []
    max_values = {"Ry": [], "Work": [], "Fracture": [], "Elastic": []}

    for f in sorted(files, key=extract_number)[::stride]:
        idx = extract_number(f)
        if idx < min_index:
            continue

        with open(f, "r") as infile:
            lines = [l for l in infile if not l.startswith("#")]
        if not lines:
            continue

        raw_data = np.loadtxt(lines)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        u_y_abs = np.abs(raw_data[:, 1])
        R_y_abs = np.abs(raw_data[:, 2])
        Work = np.abs(raw_data[:, 3])
        Fracture = np.abs(raw_data[:, 4])
        Elastic = np.abs(raw_data[:, 5])

        arr = np.full((raw_data.shape[0], 6), np.nan)
        arr[:, 1] = u_y_abs
        arr[:, 2] = R_y_abs
        arr[:, 3] = Work
        arr[:, 4] = Fracture
        arr[:, 5] = Elastic

        data_to_plot.append(arr.T)

        legend_entries.append(f"{idx}")
        file_indices.append(idx)

        max_values["Ry"].append(np.max(R_y_abs))
        max_values["Work"].append(np.max(Work))
        max_values["Fracture"].append(np.max(Fracture))
        max_values["Elastic"].append(np.max(Elastic))

    return data_to_plot, legend_entries, file_indices, max_values


def collect_volumes(base_folders, types, min_index=0):
    vol_data = {}
    for key, folder in base_folders.items():
        vol_data[key] = {"indices": [], "vols": []}
        for f in glob.glob(os.path.join(folder, "vol_*_" + types[key] + ".json")):
            idx_match = re.search(r"vol_(\d+)_", f)
            if not idx_match:
                continue
            idx = int(idx_match.group(1))
            if idx < min_index:
                continue
            with open(f, "r") as infile:
                js = json.load(infile)
                vol = js.get("vol", None)
            if vol is not None:
                vol_data[key]["indices"].append(idx)
                vol_data[key]["vols"].append(vol)
    return vol_data


def filter_indices_and_values(indices, values, min_index):
    new_idx, new_vals = [], []
    for i, v in zip(indices, values):
        if i >= min_index:
            new_idx.append(i)
            new_vals.append(v)
    return new_idx, new_vals


def main():
    parser = argparse.ArgumentParser(
        description="Plot energy-related quantities, maxima, and volumes vs index."
    )
    parser.add_argument(
        "--base_folder",
        default="/home/scripts/054-Special-Issue-IJF-Hannover/resources/250925_TTO_mbb_festlager_var_a_E_var_min_max_volumetric/",
        help="Base folder containing min/max/vary subfolders."
    )
    parser.add_argument(
        "--ext",
        default="volumetric",
        help="Optional filename extension for output plots (default: '')."
    )
    parser.add_argument(
        "--min_index",
        type=int,
        default=5,
        help="Exclude all data below this index (default: 0)."
    )
    args = parser.parse_args()

    base_folder = args.base_folder
    folders = {
        "vary": os.path.join(base_folder, "mbb_festlager_var_a_E_var"),
        "min": os.path.join(base_folder, "mbb_festlager_var_a_E_min"),
        "max": os.path.join(base_folder, "mbb_festlager_var_a_E_max"),
        "avg": os.path.join(base_folder, "mbb_festlager_var_a_E_var"),
    }

    patterns = {
        "vary": "result_graphs_*_vary.txt",
        "min": "result_graphs_*_min.txt",
        "max": "result_graphs_*_max.txt",
        "avg": "result_graphs_*_fromfile.txt"
    }

    types = {
        "vary": "vary",
        "min": "min",
        "max": "max",
        "avg": "fromfile"
    }

    all_indices = {}
    all_max = {"Ry": {}, "Work": {}, "Fracture": {}, "Elastic": {}}

    script_path = os.path.dirname(os.path.abspath(__file__))
    xlabel = "$u_y$ / mm"

    stride_for_curves = 2
    stride_for_max = 1

    for key in ["vary", "min", "max", "avg"]:
        folder = folders[key]
        pattern = patterns[key]

        if not os.path.isdir(folder):
            print(f"Error: folder '{folder}' not found. Skipping {key}.")
            continue

        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            print(f"No files found for pattern {pattern} in folder {folder}")
            continue

        data_to_plot, legend_entries, file_indices, max_values = collect_data(
            files, stride=stride_for_curves, min_index=args.min_index
        )
        if not data_to_plot:
            print(f"No usable data for {key}.")
            continue

        if key == "avg":
            legend_entries = [f"avg {i}" for i in file_indices]
        else:
            legend_entries = [f"{key} {i}" for i in file_indices]

        _, _, file_indices_all, max_values_all = collect_data(
            files, stride=stride_for_max, min_index=args.min_index
        )
        all_indices[key] = file_indices_all
        for qty in ["Ry", "Work", "Fracture", "Elastic"]:
            all_max[qty][key] = max_values_all[qty]

        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=2,
            output_filename=os.path.join(script_path, f"Ry_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="$R_y$ / (N/mm)",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=3,
            output_filename=os.path.join(script_path, f"Work_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Work $G_c$ / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=4,
            output_filename=os.path.join(script_path, f"FractureEnergy_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Fracture Energy $G_c$ / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=5,
            output_filename=os.path.join(script_path, f"ElasticEnergy_vs_uy_{key}{args.ext}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Elastic Energy / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )

    # === Combined max plots (energies) ===
    script_out = os.path.join(script_path, f"max_values_vs_index{args.ext}.png")
    x_values, y_values, labels = [], [], []

    for qty, label in zip(["Work", "Fracture", "Elastic"],
                          ["Max Work", "Max Fracture Energy", "Max Elastic Energy"]):
        for key in all_indices:
            indices, vals = filter_indices_and_values(all_indices[key], all_max[qty][key], args.min_index)
            if indices:
                x_values.append(indices)
                y_values.append(vals)
                labels.append(f"{label} ({key})")

    if x_values:
        ev.plot_multiple_lines(
            x_values=x_values, y_values=y_values,
            title="Max values vs BC position",
            x_label="BC position", y_label="$G_c$ / mm",
            legend_labels=labels,
            output_file=script_out,
            figsize=(12, 8), usetex=True, show_markers=True, use_colors=True, bold_text=True
        )

    # === Max Reaction Force vs index ===
    ry_plot_file = os.path.join(script_path, f"max_Ry_vs_index{args.ext}.png")
    x_vals, y_vals, labels = [], [], []
    for key in all_indices:
        indices, vals = filter_indices_and_values(all_indices[key], all_max["Ry"][key], args.min_index)
        if indices:
            x_vals.append(indices)
            y_vals.append(vals)
            labels.append(f"Max R_y ({key})")

    if x_vals:
        ev.plot_multiple_lines(
            x_values=x_vals, y_values=y_vals,
            title="Max Reaction Force vs BC position",
            x_label="BC position", y_label="$R_y$ / (N/mm)",
            legend_labels=labels,
            output_file=ry_plot_file,
            figsize=(12, 8), usetex=True, show_markers=True, use_colors=True, bold_text=True
        )

    # === Max Work vs index ===
    work_plot_file = os.path.join(script_path, f"max_Work_vs_index{args.ext}.png")
    x_vals, y_vals, labels = [], [], []
    for key in all_indices:
        indices, vals = filter_indices_and_values(all_indices[key], all_max["Work"][key], args.min_index)
        if indices:
            x_vals.append(indices)
            y_vals.append(vals)
            labels.append(f"Max Work ({key})")

    if x_vals:
        ev.plot_multiple_lines(
            x_values=x_vals, y_values=y_vals,
            title="Max Work vs BC position",
            x_label="BC position", y_label="Work $G_c$ / mm",
            legend_labels=labels,
            output_file=work_plot_file,
            figsize=(12, 8), usetex=True, show_markers=True, use_colors=True, bold_text=True
        )

    # === Max Fracture Energy vs index ===
    fracture_plot_file = os.path.join(script_path, f"max_FractureEnergy_vs_index{args.ext}.png")
    x_vals, y_vals, labels = [], [], []
    for key in all_indices:
        indices, vals = filter_indices_and_values(all_indices[key], all_max["Fracture"][key], args.min_index)
        if indices:
            x_vals.append(indices)
            y_vals.append(vals)
            labels.append(f"Max Fracture Energy ({key})")

    if x_vals:
        ev.plot_multiple_lines(
            x_values=x_vals, y_values=y_vals,
            title="Max Fracture Energy vs BC position",
            x_label="BC position", y_label="Fracture Energy $G_c$ / mm",
            legend_labels=labels,
            output_file=fracture_plot_file,
            figsize=(12, 8), usetex=True, show_markers=True, use_colors=True, bold_text=True
        )

    # === Volumes vs index ===
    vol_data = collect_volumes(folders, types, min_index=args.min_index)
    vol_plot_file = os.path.join(script_path, f"volumes_vs_index{args.ext}.png")

    x_vals, y_vals, labels = [], [], []
    for key, dct in vol_data.items():
        indices, vals = filter_indices_and_values(dct["indices"], dct["vols"], args.min_index)
        if indices:
            x_vals.append(indices)
            y_vals.append(vals)
            labels.append(f"Volume ({key})")

    if x_vals:
        ev.plot_multiple_lines(
            x_values=x_vals, y_values=y_vals,
            title="Volumes vs BC position",
            x_label="BC position", y_label="Volume",
            legend_labels=labels,
            output_file=vol_plot_file,
            figsize=(12, 8), usetex=True, show_markers=True, use_colors=True, bold_text=True,
            markers_only=True
        )


if __name__ == "__main__":
    main()
