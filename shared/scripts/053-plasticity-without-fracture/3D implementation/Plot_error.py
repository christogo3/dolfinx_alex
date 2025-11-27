import numpy as np
import matplotlib.pyplot as plt

# Input files
path1 = '/home/scripts/053-plasticity-without-fracture/3D implementation/run_simulation_3d_graphs_hex_q2.txt'
path2 = '/home/scripts/053-plasticity-without-fracture/3D implementation/run_simulation_3d_graphs_hex_q3.txt'
save_path = '/home/scripts/053-plasticity-without-fracture/3D implementation/'
# Formatting: Time u_y R_y

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and convert data to a NumPy array
    data = [list(map(float, line.split())) for line in lines if not line.startswith('#')]
    data = np.array(data)
    return data[:, 2], data[:, 1]  # Return first column as x, second as y


def plot_error(data1, data2, title_addon):

    Error_List = []
    for i, _ in enumerate(data1):
        Error = abs(data1[i]-data2[i])
        Error_List.append(Error)
    
    plt.figure(title_addon)
    plt.plot(Error_List)
    plt.xlabel("Timestep")
    plt.ylabel("Linear error")
    plt.title(title_addon + " error for 1 quadrature and 2 quadrature elements")
    plt.savefig(save_path + f'error_plot_{title_addon}')

u_y_1, R_y_1 = read_data(path1)
u_y_2, R_y_2 = read_data(path2)

#plot_error(u_y_1,u_y_2,'u_y') # since u_y is defined by trhe current timestep, it will always be the same
plot_error(R_y_1,R_y_2,'hex_q2_q3')