import os
import sys
import glob
import numpy as np
import re
import argparse
import alex.evaluation as ev  # plotting utilities with plot_multiple_columns & plot_multiple_lines


def extract_number(filename):
    match = re.search(r"result_graphs_(\d+)_", filename)
    return int(match.group(1)) if match else float('inf')


def collect_data(files, stride=1):
    data_to_plot = []
    legend_entries = []
    file_indices = []
    max_values = {"Ry": [], "Work": [], "Fracture": [], "Elastic": []}

    for f in sorted(files, key=extract_number)[::stride]:
        with open(f, "r") as infile:
            lines = [l for l in infile if not l.startswith("#")]
        if not lines:
            continue

        raw_data = np.loadtxt(lines)

        # columns: time, u_y, R_y, Work, FractureEnergy, ElasticEnergy
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

        idx = extract_number(f)
        legend_entries.append(f"BC position {idx}")
        file_indices.append(idx)

        max_values["Ry"].append(np.max(R_y_abs))
        max_values["Work"].append(np.max(Work))
        max_values["Fracture"].append(np.max(Fracture))
        max_values["Elastic"].append(np.max(Elastic))

    return data_to_plot, legend_entries, file_indices, max_values


def main():
    parser = argparse.ArgumentParser(description="Plot energy-related quantities vs u_y and their maxima.")
    parser.add_argument(
        "folder", nargs="?", default="/home/scripts/054-Special-Issue-IJF-Hannover/resources/250925_TTO_mbb_festlager_var_a_E_var_min_max/mbb_festlager_var_a_E_var",
        help="Folder containing result_graphs_*.txt files."
    )
    args = parser.parse_args()
    folder = args.folder

    if not os.path.isdir(folder):
        print(f"Error: folder '{folder}' not found.")
        sys.exit(1)

    patterns = {
        "vary": "result_graphs_*_vary.txt",
        "min": "result_graphs_*_min.txt",
        "max": "result_graphs_*_max.txt",
    }

    all_indices = {}
    all_max = {"Ry": {}, "Work": {}, "Fracture": {}, "Elastic": {}}

    script_path = os.path.dirname(os.path.abspath(__file__))
    xlabel = "$u_y$ / mm"

    stride_for_curves = 2
    stride_for_max = 1

    for key, pattern in patterns.items():
        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            print(f"No files found for pattern {pattern}")
            continue

        data_to_plot, legend_entries, file_indices, max_values = collect_data(files, stride=stride_for_curves)
        if not data_to_plot:
            print(f"No usable data for {key}.")
            continue

        # Full stride=1 for maxima
        _, _, file_indices_all, max_values_all = collect_data(files, stride=stride_for_max)
        all_indices[key] = file_indices_all
        for qty in ["Ry", "Work", "Fracture", "Elastic"]:
            all_max[qty][key] = max_values_all[qty]

        # === Plots ===
        # R_y
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=2,
            output_filename=os.path.join(script_path, f"Ry_vs_uy_{key}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="$R_y$ / (N/mm)",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        # Work
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=3,
            output_filename=os.path.join(script_path, f"Work_vs_uy_{key}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Work $G_c$ / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        # Fracture Energy
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=4,
            output_filename=os.path.join(script_path, f"FractureEnergy_vs_uy_{key}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Fracture Energy $G_c$ / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )
        # Elastic Energy
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1, col_y=5,
            output_filename=os.path.join(script_path, f"ElasticEnergy_vs_uy_{key}.png"),
            legend_labels=legend_entries,
            xlabel=xlabel, ylabel="Elastic Energy / mm",
            usetex=True, use_colors=True, legend_outside=True, figsize=(15, 7), vary_linestyles=True
        )

    # === Combined max plots ===
    script_out = os.path.join(script_path, "max_values_vs_index.png")

    x_values = []
    y_values = []
    labels = []

    for qty, label in zip(["Work", "Fracture", "Elastic"],
                          ["Max Work", "Max Fracture Energy", "Max Elastic Energy"]):
        if "vary" in all_indices:  # use same indices from vary for combined plot
            x_values.append(all_indices["vary"])
            y_values.append(all_max[qty]["vary"])
            labels.append(label)

    if x_values:
        ev.plot_multiple_lines(
            x_values=x_values, y_values=y_values,
            title="Max values vs BC position",
            x_label="BC position", y_label="$G_c$ / mm",
            legend_labels=labels,
            output_file=script_out,
            figsize=(10, 7), usetex=True, show_markers=True, use_colors=True, bold_text=True
        )
        print(f"Combined energy plot saved to {script_out}")


if __name__ == "__main__":
    main()


