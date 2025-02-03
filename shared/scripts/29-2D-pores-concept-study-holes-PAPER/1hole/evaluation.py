import alex.evaluation as ev
import os as os
import pandas as pd

def find_simulation_by_wsteg(path, wsteg_value_in):
    # Iterate over all directories starting with "simulation_"
    for directory in os.listdir(path):
        dir_path = os.path.join(path, directory)
        
        # Check if it is a directory and starts with "simulation_"
        if os.path.isdir(dir_path) and directory.startswith("simulation_"):
            parameters_file = os.path.join(dir_path, "parameters.txt")
            
            # Check if parameters.txt exists in the directory
            if os.path.isfile(parameters_file):
                # Open and read the file line by line
                with open(parameters_file, "r") as file:
                    for line in file:
                        # Look for the line that starts with "wsteg="
                        if line.startswith("wsteg="):
                            # Extract the value and convert to float
                            try:
                                wsteg_value = float(line.split("=")[1].strip())
                                # Compare with the given value
                                if wsteg_value == wsteg_value_in:
                                    return dir_path
                            except ValueError:
                                continue  # Skip line if conversion fails

    # Return None if no directory with the matching wsteg value is found
    return None

script_path = os.path.dirname(__file__)
data_directory = os.path.join(script_path)
simulation_data_folder = find_simulation_by_wsteg(data_directory,wsteg_value_in=16.0)
#simulation_data_folder= os.path.join(script_path,"simulation_20241205_065319")

data_path = os.path.join(simulation_data_folder, 'run_simulation_graphs.txt')
parameter_path = os.path.join(simulation_data_folder,"parameters.txt")
circular_label = "circ"
J_x_label = "Jx"

# Load the data from the text file, skipping the first row
data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)

output_file = os.path.join(script_path, 'Jx')
ev.plot_multiple_columns([data],3,1,output_file,
                         legend_labels=[circular_label],usetex=True,xlabel="$x_{ct} / L$",ylabel=J_x_label,
                         y_range=[0.0, 1.5])