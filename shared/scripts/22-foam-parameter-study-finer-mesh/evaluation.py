import os
import re
import numpy as np
import math
import matplotlib.colors as mcolors

import alex.evaluation as ev


import pandas as pd
import statsmodels.api as sm

from typing import Callable, List, Dict, Tuple


### PROBLEM DEFINITION
# height / width of domain
reference_L_global = 1.2720814740168862

# MESH SIZES
h_coarse_mean =  0.024636717648428213 #0.040704583134024946
h_all = {
    "coarse_pores": h_coarse_mean,
    "medium_pores": h_coarse_mean/2.0,
    "fine_pores": h_coarse_mean/4.0,
}
# PORE SIZES
pore_size_coarse =  0.183
pore_size_all = {
    "coarse_pores": pore_size_coarse,
    "medium_pores": pore_size_coarse/2.0,
    "fine_pores": pore_size_coarse/4.0,
}

# Extract script path and name
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

# Directory containing the simulation folders
directory_path = os.path.join(script_path, "data_not_tracked", "finer_original_dir")

# Regular expression pattern to extract the required values from the outer folder names
outer_pattern = re.compile(
    r"simulation_\d+_\d{6}_(?P<mesh_name>[a-zA-Z0-9_]+)_lam(?P<lam_value>\d+\.\d+)_mue(?P<mue_value>\d+\.\d+)_Gc(?P<Gc_value>\d+\.\d+)_eps(?P<eps_value>\d+\.\d+)_order(?P<order_value>\d+)"
)

# Dictionary to store the data
results_dict = {}
first_level_keys = set()


# Iterate over all the directories
for folder_name in os.listdir(directory_path):
    outer_match = outer_pattern.match(folder_name)
    if outer_match:
        # Extract the values from the outer directory name
        mesh_name = outer_match.group("mesh_name")
        lam_value = float(outer_match.group("lam_value"))
        mue_value = float(outer_match.group("mue_value"))
        Gc_value = float(outer_match.group("Gc_value"))
        eps_value = float(outer_match.group("eps_value"))
        order_value = int(outer_match.group("order_value"))

        # Get the path of the outer folder
        outer_folder_path = os.path.join(directory_path, folder_name)

        # Initialize variables
        data_file_path = None
        first_level_found = False

        # Check for any file ending with *_graphs.txt at the first level
        for item in os.listdir(outer_folder_path):
            if item.endswith("_graphs.txt") and os.path.isfile(os.path.join(outer_folder_path, item)):
                data_file_path = os.path.join(outer_folder_path, item)
                first_level_found = True
                break

        # If not found at the first level, check within the inner folder
        if not first_level_found:
            inner_folder_name = None
            for item in os.listdir(outer_folder_path):
                if item.startswith("simulation_") and os.path.isdir(os.path.join(outer_folder_path, item)):
                    inner_folder_name = item
                    break

            if inner_folder_name:
                inner_folder_path = os.path.join(outer_folder_path, inner_folder_name)
                for item in os.listdir(inner_folder_path):
                    if item.endswith("_graphs.txt") and os.path.isfile(os.path.join(inner_folder_path, item)):
                        data_file_path = os.path.join(inner_folder_path, item)
                        break

        # If a data file was found, process it
        if data_file_path and os.path.isfile(data_file_path):
            # Read the data from the file
            data = []
            with open(data_file_path, 'r') as file:
                for line in file:
                    # Skip lines that start with #
                    if line.startswith('#'):
                        continue
                    # Split the line into columns and convert to float
                    data.append(list(map(float, line.split())))

            # Store the data in the dictionary
            key = f"{mesh_name}_lam{lam_value}_mue{mue_value}_Gc{Gc_value}_eps{eps_value}_order{order_value}"
            results_dict[key] = np.array(data)

            # If the file was found at the first level, add the key to first_level_keys
            if first_level_found:
                first_level_keys.add(key)
            
           



# Example usage
# param_string = "coarse_pores_lam10.0_mue10.0_Gc0.5_eps50.0_order2"
# param_names = [['mesh_name'], ['lam', 'mue'], ['Gc']]
# new_param_string = remove_parameter(param_string, param_names)
# print(new_param_string)  # Output: "lam10.0_mue10.0_eps50.0_order2"


# 0.

keys = [ "medium_pores_lam1.0_mue1.0_Gc0.4842_eps51.6336_order1" ]
# parameter_names = ['Gc']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"TEST" + ".png")
ev.plot_results_2yaxis(results_dict, keys, [1,4], save_path,y_labels=["Jx","crack length"],show_legend=False)

# 1. Gc
keys = ["coarse_pores_lam1.0_mue1.0_Gc0.9684_eps25.8168_order1", 
        "coarse_pores_lam1.0_mue1.0_Gc1.1444_eps21.8454_order1",
        "coarse_pores_lam1.0_mue1.0_Gc1.3987_eps17.8739_order1",
        "coarse_pores_lam1.0_mue1.0_Gc1.7982_eps13.9025_order1", 
        "coarse_pores_lam1.0_mue1.0_Gc1.9367_eps25.8168_order1",
        "coarse_pores_lam1.0_mue1.0_Gc2.2888_eps21.8454_order1",
        "coarse_pores_lam1.0_mue1.0_Gc3.5965_eps13.9025_order1", 
        "coarse_pores_lam1.0_mue1.0_Gc2.7974_eps17.8739_order1"]
parameter_names = ['Gc']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
ev.plot_results(results_dict, keys, column_index, save_path)

# 2. Stiffness
keys = ["coarse_pores_lam1.0_mue1.0_Gc1.7982_eps13.9025_order1", 
         "coarse_pores_lam1.5_mue1.0_Gc1.7982_eps13.9025_order1",
         "coarse_pores_lam1.0_mue1.5_Gc1.7982_eps13.9025_order1"]
parameter_names = ['lam','mue']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
ev.plot_results(results_dict, keys, column_index, save_path)


# 3. Mesh
keys = ["coarse_pores_lam1.0_mue1.0_Gc1.7982_eps13.9025_order1", 
        "medium_pores_lam1.0_mue1.0_Gc1.7982_eps27.805_order1",
        "fine_pores_lam1.0_mue1.0_Gc0.8991_eps55.61_order1"] 
parameter_names = ['mesh_name']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
ev.plot_results(results_dict, keys, column_index, save_path)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + "_crack_tip_position.png")
ev.plot_results(results_dict, keys, column_index=4, save_path=save_path)

# 4. Epsilon
# TODO: try to show that maximum energy release rate is the same if results are normalized by effective stress in phase field model

keys = ["coarse_pores_lam1.0_mue1.0_Gc1.9367_eps25.8168_order1",
        "coarse_pores_lam1.0_mue1.0_Gc2.2888_eps21.8454_order1",
        "coarse_pores_lam1.0_mue1.0_Gc3.5965_eps13.9025_order1", 
        "coarse_pores_lam1.0_mue1.0_Gc2.7974_eps17.8739_order1"]
parameter_names = ['eps']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + ".png")
ev.plot_results(results_dict, keys, column_index, save_path)


# 4a. Epsilon
# With normalization w.r.t effective stress, 
keys = ["coarse_pores_lam1.0_mue1.0_Gc1.9367_eps25.8168_order1",
        "coarse_pores_lam1.0_mue1.0_Gc2.2888_eps21.8454_order1",
        "coarse_pores_lam1.0_mue1.0_Gc3.5965_eps13.9025_order1", 
        "coarse_pores_lam1.0_mue1.0_Gc2.7974_eps17.8739_order1"]

i = 0


sig_c = ev.get_sig_c(ev.extract_parameters,keys,reference_L_global=reference_L_global)  
scaling_factors = 1.0 / (sig_c *  reference_L_global)

parameter_names = ['eps']
column_index = 1  # Column index to plot (e.g., the second column in the results)
save_path = os.path.join(directory_path,"_".join(parameter_names) + "_" +  ev.remove_parameter(keys[0],
                        param_names=[parameter_names]) + "_TEST.png")
ev.plot_results(results_dict, keys, column_index, save_path, scaling_factors=scaling_factors)











max_Jx_dict = ev.create_max_dict(results_dict, column_index=1)
# Compute sig_c for all keys in max_Jx_dict
keys = list(max_Jx_dict.keys())



mesh_colors = {
        "fine_pores": mcolors.to_rgba('darkred'),
        "medium_pores": mcolors.to_rgba('red'),
        "coarse_pores": mcolors.to_rgba('lightcoral')
    }





# def gc_num(mesh_name,gc,eps):
#     return gc * (1.0 + h_all(mesh_name)/eps)

mesh_all = ["fine_pores","medium_pores", "coarse_pores"]


values_of_params_coarse_mesh = ev.get_values_for_prefix(keys, "coarse_pores")
values_of_params_medium_mesh = ev.get_values_for_prefix(keys, "medium_pores")
values_of_params_fine_mesh = ev.get_values_for_prefix(keys, "fine_pores")

values_of_params_all = ev.merge_dicts([values_of_params_coarse_mesh, values_of_params_medium_mesh, values_of_params_fine_mesh])



gc_all = np.array(values_of_params_all["Gc"])
eps_all = np.array(values_of_params_all["eps"])
lam_all = np.array(values_of_params_all["lam"])
mue_all = np.array(values_of_params_all["mue"])

keys_to_plot = keys # ['fine_pores_lam1.0_mue1.0_Gc1.0_eps1.0_order1', 'medium_pores_lam10.0_mue10.0_Gc2.0_eps0.5_order2']
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "max_Jx_vs_sig_c.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path,reference_L_global=reference_L_global,pore_size_all=pore_size_all, special_keys=first_level_keys)

# 1. fixed Gc
target_Gc_values = np.array([1.3987])  
target_eps_values = np.array(np.array([values_of_params_all["eps"]])  )  
target_lam_values = np.array([0.6667,1.0, 1.5])  
target_mue_values = np.array([1.0])   
target_mesh_types = ["fine_pores", "medium_pores", "coarse_pores"]  

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "001_max_Jx_vs_sig_c_Gc_fixed.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all, special_keys=first_level_keys)


# 2. varying stiffness
target_Gc_values = np.array([0.9684])  
target_eps_values = np.array(values_of_params_medium_mesh["eps"])  
target_lam_values = lam_all  
target_mue_values = mue_all
target_mesh_types = ["medium_pores"] 

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "002_max_Jx_vs_sig_c_varying_stiffness.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all,special_keys=first_level_keys)

# 3. varying Gc
target_Gc_values = gc_all 
target_eps_values = [21.8454, 43.6908, 87.3816]  
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["coarse_pores", "medium_pores", "fine_pores" ]  

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "003_max_Jx_vs_sig_c_varying_Gc.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all,special_keys=first_level_keys)

# 4. varying eps
intersect_medium_coarse = ev.intersect_dicts(values_of_params_coarse_mesh,values_of_params_medium_mesh)
intersect_medium_fine = ev.intersect_dicts(values_of_params_fine_mesh,values_of_params_medium_mesh)
merged_data = ev.merge_dicts([intersect_medium_coarse,intersect_medium_fine])

target_Gc_values = np.array(merged_data["Gc"])  
target_eps_values = eps_all  
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores", "coarse_pores" ,"fine_pores"]  

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "004_max_Jx_vs_sig_c_varying_eps.png")
# ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all, special_keys=first_level_keys)
ev.plot_max_Jx_vs_pore_size_eps_ratio(results_dict,keys_to_plot,plot_title,save_path, pore_size_all, reference_L_global=reference_L_global, special_keys=first_level_keys)


# 5. varying eps fixed sig_c
target_Gc_values = gc_all 
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores"]  

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "005_max_Jx_vs_sig_c_varying_eps_fixed_sigc.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all, special_keys=first_level_keys)



# 6. plot all gc_num
target_Gc_values = gc_all 
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["coarse_pores"]
filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)

save_path = os.path.join(directory_path, "006_gc_num_vs_gc.png")
ev.plot_gc_num_vs_gc(results_dict, filtered_keys, h_all, save_path, reference_L_global=reference_L_global, mesh_colors=mesh_colors)

#7. plot ratio eps / h 
target_Gc_values =  np.array(merged_data["Gc"])  
target_eps_values = eps_all 
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["coarse_pores", "medium_pores", "fine_pores"]
filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)

save_path = os.path.join(directory_path, "007_eps_h_ratio.png")
ev.plot_eps_h_ratio(filtered_keys, h_all, reference_L_global, mesh_colors, output_file=save_path, lower_limit_eps_h=2.0, lower_limit_pore_size_eps=2.0, pore_size_all=pore_size_all)
# plot_eps_h_ratio(filtered_keys,output_file=save_path)


# 8. plot ratio pore size vs eps and pore size vs h
# Assuming results_dict is already defined and populated
save_path_ratio_eps = os.path.join(directory_path, "008_ratio_pore_size_eps.png")
ev.plot_ratio_pore_size_eps(results_dict, save_path_ratio_eps,reference_L_global,pore_size_all,mesh_colors)

save_path_ratio_h = os.path.join(directory_path, "009_ratio_pore_size_h.png")
ev.plot_ratio_pore_size_h(results_dict, save_path_ratio_h,h_all,pore_size_all,mesh_colors)




## 10. varying eps (with different function)

# get datasets with same Gc
intersect_medium_coarse = ev.intersect_dicts(values_of_params_coarse_mesh,values_of_params_medium_mesh)
intersect_medium_fine = ev.intersect_dicts(values_of_params_fine_mesh,values_of_params_medium_mesh)
merged_data = ev.merge_dicts([intersect_medium_coarse,intersect_medium_fine])

target_Gc_values = np.array(merged_data["Gc"])  
target_eps_values = eps_all  
target_lam_values = np.array([1.0])  
target_mue_values = np.array([1.0])  
target_mesh_types = ["medium_pores", "coarse_pores" ,"fine_pores"]  

filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=target_Gc_values,
                            target_eps=target_eps_values,
                            target_lam=target_lam_values,
                            target_mue=target_mue_values,
                            target_mesh_types=target_mesh_types)


keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "010_max_Jx_vs_sig_c_varying_eps.png")

xdata_label = "eps_vs_h_ratio"
def xdata_function(key):
    params = ev.extract_parameters(key)
    if params:
        mesh_type = params[0]
        eps_value = ev.get_eps(reference_L_global, params[4])
        xdata = eps_value / h_all[mesh_type]
        return xdata

def data_label_function1(key):
    params = ev.extract_parameters(key)
    if params:
        mesh_type = params[0]
        eps_value = ev.get_eps(reference_L_global, params[4])
        pore_size = pore_size_all[mesh_type]
        return params[3], pore_size / eps_value

ev.plot_max_Jx_vs_data(results_dict, filtered_keys,
                       xdata_function=xdata_function,
                       xdata_label=xdata_label, data_label_function=data_label_function1,
                       plot_title=plot_title, save_path=save_path, special_keys=first_level_keys)



### 12. Const sig_c

def group_by_sig_c(parameters: Tuple[str, float, float, float, float, int]):
    
        return round(ev.sig_c_quadr_deg(parameters[3],parameters[2],ev.get_eps(reference_L_global,parameters[4])),2)
    
keys_grouped_by_sigc = ev.group_by_function(keys,group_by_sig_c)


#### do regression analysis in order to find trends, BAD dataset sig_c is almost const.
filtered_keys = ev.filter_keys(results_dict,
                            target_Gc=[0.5722,0.6993,1.3987, 1.7982],
                            target_eps=[43.6908, 35.7478, 27.8050],
                            target_lam=np.array([1.0]),
                            target_mue=np.array([1.0]),
                            target_mesh_types=["medium_pores"])

max_dict = ev.create_max_dict(results_dict=results_dict,column_index=1)
sig_c_all = ev.get_sig_c(extract_parameters=ev.extract_parameters, keys=filtered_keys,reference_L_global=reference_L_global)

keys_to_plot = filtered_keys 
plot_title = "Max Jx vs sig_c"
save_path = os.path.join(directory_path, "011_Test.png")
ev.plot_max_Jx_vs_sig_c(results_dict, keys_to_plot, plot_title, save_path, reference_L_global=reference_L_global,pore_size_all=pore_size_all,special_keys=first_level_keys)


maxJx = []
sig_c = []
Gc = []
for key in filtered_keys:
    maxJx.append(max_dict[key])
    
   
    
    params = ev.extract_parameters(key)
    mesh_name = params[0]
    
    mesh_to_number = {
        "coarse_pores": 1,
        "medium_pores": 2,
        "fine_pores": 4
    }
    mesh_value = mesh_to_number[mesh_name]
    
    lam_value = params[1]
    mue_value = params[2]
    Gc_value = params[3]
    eps_value = params[4]
    order_value = params[5]
    
    Gc.append(Gc_value)
    sig_c.append(ev.sig_c_quadr_deg(Gc_value,mue_value,eps_value))
    
data = {
    #"sig_c" : sig_c,
    # "Gc" : Gc,
    "mesh": mesh_value,
    "lam": lam_value,
    "mue": mue_value,
    "Gc": Gc_value,
    "eps": eps_value,
    "maxJx": maxJx
}

df = pd.DataFrame(data)

X = df[["mesh", "lam", "mue", "Gc", "eps"]]
Y = df["maxJx"]

# Konstante hinzufügen (intercept)
X = sm.add_constant(X)

# Regression durchführen
model = sm.OLS(Y, X).fit()

# Zusammenfassung des Modells ausgeben
print(model.summary())

    
        

