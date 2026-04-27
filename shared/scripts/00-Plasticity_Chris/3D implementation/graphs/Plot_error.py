import numpy as np
import matplotlib.pyplot as plt
import os.path

script_path = os.path.dirname(__file__)

# Input files

path1 = '/home/scripts/00-Plasticity_Chris/3D implementation/run_simulation_3D_graphs.txt'
path2 = '/home/scripts/00-Plasticity_Chris/Large deformations/run_simulation_large_deformation_graphs.txt'
label_list = ['hex q=1', 'hex q=2', 'tet q=1', 'tet q=2']
'''name_list = ['graphs_hex_q1.txt',
             'graphs_hex_q2.txt',
             'graphs_tet_q1.txt',
             'graphs_tet_q1.txt']
name_componentwise = ['graphs_component_hex_q1.txt',
                      'graphs_component_hex_q2.txt',
                      'graphs_component_tet_q1.txt',
                      'graphs_component_tet_q2.txt']'''

save_path = script_path
# Formatting: Time u_y R_y


def read_data(filename):
    """
    Read the data from a single path
    Returns:
        displacement, reaction force, timestep
    """
    path = os.path.join(script_path,filename)
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and convert data to a NumPy array
    data = [list(map(float, line.split())) for line in lines if not line.startswith('#')]
    data = np.array(data)
    return data[:, 2], data[:, 1], data[:,0]  # Return first column as x, second as y

def multi_read_data(name_list): 
    """
    Read multiple datasets from a list of paths
    """
    R_y_list = []
    for filename in name_list:
        path = os.path.join(script_path,filename)
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Skip comments and convert data to a NumPy array
        data = [list(map(float, line.split())) for line in lines if not line.startswith('#')]
        data = np.array(data)
        R_y_list.append(data[:,1])
    return R_y_list # Return first column as x, second as y

def plot_error(path1, path2, title, filename, correction=1):
    """
    Plot the absolute error of the reaction force in path2 with respect to path1
    Correction is applied to data in path 2
    """

    u_y_1, R_y_1, t = read_data(path1)
    u_y_2, R_y_2, t = read_data(path2)

    Error_List = []
    for i, _ in enumerate(R_y_1):

        Error = abs(R_y_1[i]-R_y_2[i]*correction)
        Error_List.append(Error)
    
    plt.figure(filename)
    plt.plot(Error_List, label='error')
    #plt.plot(data1, label='data')
    plt.xlabel("Timestep")
    plt.ylabel("Linear error")
    plt.title(title)
    plt.savefig(save_path + f'/error_plot_{filename}')

def error_multiplot(path_list, label_list, title, filename, plot_force=False):
    """
    Plot the absolute error of all datasets within a path list with respect to the first entry.
    """
    data_list = multi_read_data(path_list)
    ref = data_list[0]
    linestyle = ['-', '--', '-', '--', '-', '--', '-', '--']

    plt.figure(filename)
    max_errors = []
    for i, _ in enumerate(data_list):
        #if i==0: continue
        Error_List = []
        data = data_list[i]
        
        for j, _ in enumerate(data):
            Error = abs(data[j]-ref[j])
            Error_List.append(Error)

        plt.plot(Error_List, label=label_list[i], linestyle=linestyle[i], alpha=0.7)
        max_errors.append(np.max(Error_List))
    
    if plot_force == True: 
        # rescale to fit data
        error_max = np.max(max_errors)
        if error_max == 0: error_max = 1
        rescaled_force = (ref/np.max(ref))*error_max
        plt.plot(rescaled_force, label='Reaction force magnitude', alpha=0.3)

    plt.xlabel("Timestep")
    plt.ylabel("Linear error")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path + f'/error_multiplot_{filename}')

def stress_strain_multiplot(path_list, label_list, title, filename, print_reference=False, scaling=False):
    """
    Plot the Force-Displacement (or Stress-Strain) diagram of all datasets from a list for comparison.
    """
    
    plt.figure(filename)
    
    for i, _ in enumerate(path_list):
        u_y, R_y, _ = read_data(path_list[i])
        if scaling == True:
            R_y = R_y/np.max(R_y)

        plt.plot(u_y,R_y, label=label_list[i]) # alpha=0.7

    if print_reference == True:
        # parameters
        sol_ref = []
        mu = 1
        la = 1
        sig_y = 1
        E = mu * (3*la + 2*mu) / (la + mu)
        hardening = 0.6
        u_yield = sig_y / E
        E_hardening = hardening*E / (hardening+E)

        for disp in u_y:
            abs_disp = abs(disp)

            
            if abs_disp<=u_yield: # linear elastic
                R_ref = E * disp
            else: # plastic effects
                sign_u = 1.0 if disp >= 0 else -1.0
                R_ref = sign_u * (sig_y + E_hardening * (abs_disp - u_yield))
                
            sol_ref.append(R_ref)
        
        plt.plot(u_y, sol_ref, label='Analytical reference', linestyle='--', alpha=0.6)

    plt.xlabel("Displacement")
    plt.ylabel("Reaction force")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path + f'/stress_strain_multiplot_{filename}')




stress_strain_multiplot([path1, path2], ["small deformation","large deformation"], 'Stress-Strain Relation Nanomesh', 'nanomesh', print_reference=True)
#plot_error(path1, path2,title='R_y error for small and large deformations', filename='large_defo_comparison',correction=(1.5026e-02)/(8.4034e-01))
#error_multiplot([path2, path1],["large deformation","small deformation"],title='Reaction force for nanomesh foam',filename='large_defo_comparison', plot_force=True)
