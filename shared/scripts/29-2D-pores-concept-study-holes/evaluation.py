import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Define the path to the file based on the script directory
script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, 'simulation_20241010_172413', 'run_simulation_graphs.txt')

# Load the data from the text file, skipping the first row
data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)

# Display the first few rows of the data to understand its structure
print(data.head())

# Function to plot any two columns against each other and save as a file
def plot_columns(data, col_x, col_y, output_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data[col_x], data[col_y], marker='o', linestyle='-')
    plt.xlabel(f'Column {col_x}')
    plt.ylabel(f'Column {col_y}')
    plt.title(f'Column {col_x} vs Column {col_y}')
    
    # Set the maximum number of ticks on each axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit x-axis to 10 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit y-axis to 10 ticks
    
    # Save the plot as a PNG file
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

# Specify the output file path
output_file = os.path.join(script_path, 'plot_column_1_vs_3.png')

# Call the function with the column indices you want to plot and the output filename
plot_columns(data, 3, 1, output_file)

