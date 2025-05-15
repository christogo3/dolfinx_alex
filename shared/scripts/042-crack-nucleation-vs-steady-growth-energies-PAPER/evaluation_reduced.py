import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter


import alex.postprocessing as pp
import alex.phasefield as pf
import alex.homogenization as hom
import alex.linearelastic as le
import alex.evaluation as ev
import math

from scipy.signal import savgol_filter
import numpy as np


import seaborn as sns
from matplotlib.lines import Line2D

from scipy.optimize import fsolve


script_path = os.path.dirname(__file__)
initial_crack_file = os.path.join(script_path, "simulation_20250305_195427","run_simulation_graphs.txt")
crack_nucleation_file = os.path.join(script_path, "simulation_20250127_131913", "run_simulation_graphs.txt")

data_initial_crack = pd.read_csv(initial_crack_file, delim_whitespace=True, header=None, skiprows=1)
data_crack_nucleation = pd.read_csv(crack_nucleation_file, delim_whitespace=True, header=None, skiprows=1)

time_initial_crack = data_initial_crack.values[:,0]
A_initial_crack = data_initial_crack.values[:,9]
Eel_initial_crack = data_initial_crack.values[:,11]
Total_energy_initial_crack = data_initial_crack.values[:,9] + data_initial_crack.values[:,11]

time_crack_nucleation = data_crack_nucleation.values[:,0]
A_crack_nucleation = data_crack_nucleation.values[:,9]
Eel_crack_nucleation = data_crack_nucleation.values[:,11]
Total_energy_crack_nucleation = data_crack_nucleation.values[:,9] + data_crack_nucleation.values[:,11]

A_label = "fracture"
E_label = "elastic"
Total_label = "total"

initial_crack_label = " prescribed crack"
crack_nucleation_label = " nucleation"
velocity_bc_label = "$\dot{x}^{bc}$"


output_file = os.path.join(script_path,"energies.png")
ev.plot_multiple_lines([time_initial_crack, time_initial_crack, time_initial_crack, time_crack_nucleation, time_crack_nucleation, time_crack_nucleation], 
                       [A_initial_crack, Eel_initial_crack,Total_energy_initial_crack,
                        A_crack_nucleation, Eel_crack_nucleation,Total_energy_crack_nucleation],
                       legend_labels=[A_label+initial_crack_label, E_label+initial_crack_label, Total_label+initial_crack_label,
                                      A_label + crack_nucleation_label,E_label+crack_nucleation_label,Total_label+crack_nucleation_label],
                       x_label="$t / ( L / $" + velocity_bc_label + ")",y_label="energy"+" / ($G_c \cdot {L}$)",
                       output_file=output_file,
                       line_colors=["blue","red","black","blue","red","black"],
                       line_styles=["--","--", "--", "-","-","-"],
                       x_range=[1.2,1.6],
                       y_range=[0.0, 3.0],
                       usetex=True,
                       legend_fontsize=18)