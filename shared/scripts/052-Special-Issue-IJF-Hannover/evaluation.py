import os
import sys
import glob
import numpy as np
import re
import argparse
import alex.evaluation as ev  # plotting utilities with plot_multiple_columns & plot_multiple_lines


def extract_number(filename):
    """Extract integer X from result_graphs_X_*.txt filename for sorting."""
    match = re.search(r"result_graphs_(\d+)_", filename)
    return int(match.group(1)) if match else float('inf')


def collect_data(files, stride=1):
    """
    Read data from a list of files, return arrays for plotting and max R_y values.
    Optionally skip files according to stride (e.g. stride=2 plots every second file).
    """
    data_to_plot = []
    legend_entries = []
    file_indices = []
    max_Ry_values = []

    # Apply stride after sorting
    for f in sorted(files, key=extract_number)[::stride]:
        with open(f, "r") as infile:
            lines = [l for l in infile if not l.startswith("#")]

        if not lines:
            continue

        raw_data = np.loadtxt(lines)  # columns: time, u_y, R_y
        u_y_abs = np.abs(raw_data[:, 1])
        R_y_abs = np.abs(raw_data[:, 2])

        arr = np.full((raw_data.shape[0], 3), np.nan)
        arr[:, 1] = u_y_abs
        arr[:, 2] = R_y_abs

        data_to_plot.append(arr.T)

        # Use BC position instead of filename
        idx = extract_number(f)
        legend_entries.append(f"BC position {idx}")

        file_indices.append(idx)
        max_Ry_values.append(np.max(R_y_abs))

    return data_to_plot, legend_entries, file_indices, max_Ry_values


def main():
    parser = argparse.ArgumentParser(description="Plot R_y vs u_y and max R_y vs BC position.")
    parser.add_argument(
        "folder", nargs="?", default="/home/scripts/052-Special-Issue-IJF-Hannover/resources/310125_var_bcpos_rho_10_120_004",
        help="Folder containing result_graphs_*.txt files."
    )
    args = parser.parse_args()

    folder = args.folder

    if not os.path.isdir(folder):
        print(f"Error: folder '{folder}' not found.")
        sys.exit(1)

    # File patterns
    patterns = {
        "vary": "result_graphs_*_vary.txt",
        "min": "result_graphs_*_min.txt",
        "max": "result_graphs_*_max.txt",
    }

    # Storage for max Ry vs index
    all_indices = {}
    all_max_Ry = {}

    # Process each file type
    script_path = os.path.dirname(os.path.abspath(__file__))
    xlabel = "$u_y$ / mm"
    ylabel = "$R_y$ / (N/mm)"

    # === First plots (every 2nd file) ===
    stride_for_curves = 2
    stride_for_max = 1

    for key, pattern in patterns.items():
        files = glob.glob(os.path.join(folder, pattern))
        if not files:
            print(f"No files found for pattern {pattern}")
            continue

        # First plots use stride = 2
        data_to_plot, legend_entries, file_indices, max_Ry_values = collect_data(files, stride=stride_for_curves)

        if not data_to_plot:
            print(f"No usable data for {key} (after stride filtering).")
            continue

        # Save ALL data (stride=1) for the combined max plot
        _, _, file_indices_all, max_Ry_values_all = collect_data(files, stride=stride_for_max)
        all_indices[key] = file_indices_all
        all_max_Ry[key] = max_Ry_values_all

        # === Plot Ry vs uy ===
        output_file = os.path.join(script_path, f"Ry_vs_uy_{key}.png")
        ev.plot_multiple_columns(
            data_objects=data_to_plot,
            col_x=1,
            col_y=2,
            output_filename=output_file,
            legend_labels=legend_entries,  # now shows BC position
            xlabel=xlabel,
            ylabel=ylabel,
            usetex=True,
            use_colors=True,
            legend_outside=True,
            figsize=(15, 7),
            vary_linestyles=True
        )
        print(f"Plot saved to {output_file}")

    # === Combined Max Ry vs Index plot (stride=1) ===
    output_file_max = os.path.join(script_path, "max_Ry_vs_index.png")

    x_values = []
    y_values = []
    labels = []

    for key in ["vary", "min", "max"]:
        if key in all_indices:
            x_values.append(all_indices[key])
            y_values.append(all_max_Ry[key])
            labels.append(f"Max $R_y$ ({key})")

    if x_values:
        ev.plot_multiple_lines(
            x_values=x_values,
            y_values=y_values,
            title="Maximum $R_y$ vs BC position",
            x_label="BC position",
            y_label="Max $R_y$ / (N/mm)",
            legend_labels=labels,
            output_file=output_file_max,
            figsize=(10, 7),
            usetex=True,
            show_markers=True,
            use_colors=True,
            bold_text=True
        )
        print(f"Combined plot saved to {output_file_max}")


if __name__ == "__main__":
    main()

